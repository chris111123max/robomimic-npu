"""CPU regression tests for pipeline, RNG isolation, and block snapshots."""
import copy
import json
import random
import tempfile
import time
import unittest
import multiprocessing as mp
import io
import contextlib
import threading
import types
from unittest.mock import patch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from stage3_v5_pipeline import TransitionCredit, overlap_burst, interval_overlap
from stage3_v5_rollout import BoundarySnapshotExecutor
from stage3_v5_diagnostics import isolated_training_rng, frozen_set
from stage3_v5_replay import OnlineSequenceReplay, BalancedOfflineDemonstrations, source_sample_metrics
from stage3_v5_execution import prepare_round_batches
from stage3_v5_readiness import stability, replay_metrics
from stage3_v5_agent import RecurrentGMMTD3
from stage3_v5_actor import flat_to_obs
from test_stage3_v5_replay import make_file

CONFIG = json.loads(Path(__file__).with_name("stage3_v5_config.json").read_text())


def simulation_worker(connection):
    connection.send("READY")
    connection.recv()
    start = time.perf_counter()
    time.sleep(.25)
    connection.send(("OK", None, 0., False, False,
                     {"_stage3_env_started":start, "_stage3_env_finished":time.perf_counter()}))
    connection.close()


class ToyActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.low_noise_eval = True

    def _distribution(self, value):
        means = value[..., None, None].expand(*value.shape, 1, 14)
        return torch.distributions.MixtureSameFamily(
            torch.distributions.Categorical(logits=torch.zeros(*value.shape, 1)),
            torch.distributions.Independent(torch.distributions.Normal(means, torch.ones_like(means)*1e-7), 1))

    def forward_train_step(self, obs, rnn_state=None):
        batch = len(obs["robot0_eef_pos"])
        state = torch.zeros(1, batch, 1) if rnn_state is None else rnn_state
        state = state + self.weight
        return self._distribution(state[0, :, 0]), state

    def forward_train(self, obs, rnn_init_state=None, return_state=False):
        shape = obs["robot0_eef_pos"].shape[:2]
        return self._distribution(self.weight.expand(*shape))


class ConstantTwin(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, states, actions):
        value = self.value.expand(len(states), 1)
        return value, value + 1.0


def episode(length=22, success=False):
    return {"observations": np.zeros((length, 59), np.float32),
            "next_observations": np.ones((length, 59), np.float32),
            "actions": np.zeros((length, 14), np.float32),
            "rewards": np.ones((length, 1), np.float32),
            "terminals": np.zeros((length, 1), np.float32),
            "episode_steps": np.arange(length), "success": success}


class TestRNGIsolation(unittest.TestCase):
    def test_all_training_rng_restored_on_exception(self):
        online = OnlineSequenceReplay(2000, 7)
        random.seed(9); np.random.seed(9); torch.manual_seed(9)
        states = (copy.deepcopy(random.getstate()), copy.deepcopy(np.random.get_state()),
                  torch.get_rng_state().clone(), copy.deepcopy(online.rng.bit_generator.state))
        with self.assertRaises(ValueError):
            with isolated_training_rng(online):
                random.random(); np.random.rand(); torch.rand(4); online.rng.random()
                raise ValueError("diagnostic failed")
        self.assertEqual(states[0], random.getstate())
        np.testing.assert_array_equal(states[1][1], np.random.get_state()[1])
        self.assertTrue(torch.equal(states[2], torch.get_rng_state()))
        self.assertEqual(states[3], online.rng.bit_generator.state)

    def test_frozen_sample_survives_replay_growth_eviction_and_resume(self):
        replay = OnlineSequenceReplay(500, 7)
        for i in range(150):
            replay.add_episode(episode(success=i % 2 == 0))
        fixed = frozen_set(replay, CONFIG)
        self.assertEqual(len(fixed["episodes"]), 128)
        original = copy.deepcopy(fixed)
        for _ in range(150):
            replay.add_episode(episode(success=True))
        self.assertIs(fixed, frozen_set(replay, CONFIG))
        np.testing.assert_array_equal(original["sequences"]["observations"], fixed["sequences"]["observations"])
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory)/"replay.npy")
            replay.save(path); restored = OnlineSequenceReplay.load(path)
            self.assertEqual(original["indices"], restored.fixed_critic_diagnostic_set["indices"])
            np.testing.assert_array_equal(original["ood_noise"], restored.fixed_critic_diagnostic_set["ood_noise"])

    def test_readiness_itself_does_not_move_sampler_or_global_rng(self):
        replay = OnlineSequenceReplay(5000, 4)
        for i in range(150): replay.add_episode(episode(success=i % 2 == 0))
        class Observer:
            def q_values_for_episodes(self, episodes):
                random.random(); np.random.rand(); torch.rand(1)
                return [(np.ones(len(e["actions"])), np.ones(len(e["actions"]))) for e in episodes]
            def fixed_td_diagnostic(self, batch):
                replay.rng.random()
                return {"q1": np.ones(256), "q2": np.ones(256), "td_target": np.ones(256), "td_mae": .1, "td_mse": .01}
            def ood_action_stress(self, batch, noise):
                return {"mean":0., "p95":0., "max":99., "normalized_p95":0.}
        before = copy.deepcopy(replay.rng.bit_generator.state)
        torch_before = torch.get_rng_state().clone()
        metrics = replay_metrics(Observer(), replay, 150, 75, CONFIG, [], env_steps=100000)
        self.assertEqual(before, replay.rng.bit_generator.state)
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
        self.assertTrue(metrics["ood_stress_pass"])
        self.assertFalse(metrics["td_plateau"])


class TestRolloutSnapshot(unittest.TestCase):
    def test_weight_change_waits_for_hidden_reset(self):
        actor = ToyActor()
        executor = BoundarySnapshotExecutor(actor, torch.ones(14), torch.zeros(14), 2, horizon=3)
        obs = flat_to_obs(np.zeros(59, np.float32))
        first = executor.actions_for([0, 1], [obs, obs], train_policy_version=0)
        with torch.no_grad(): actor.weight.fill_(4)
        second = executor.actions_for([0], [obs], train_policy_version=1)
        executor.reset_indices([1])
        mixed = executor.actions_for([0, 1], [obs, obs], train_policy_version=1)
        fourth = executor.actions_for([0], [obs], train_policy_version=1)
        np.testing.assert_allclose(first[0], 1, atol=1e-5)
        np.testing.assert_allclose(second[0], 2, atol=1e-5)
        np.testing.assert_allclose(mixed[0], 3, atol=1e-5)
        np.testing.assert_allclose(mixed[1], 4, atol=1e-5)
        np.testing.assert_allclose(fourth[0], 4, atol=1e-5)
        self.assertEqual(executor.versions, [1, 1])
        self.assertEqual(len(executor.executors), 1)


class TestBoundedLag(unittest.TestCase):
    def test_process_simulation_overlaps_real_torch_backward(self):
        parent, child = mp.get_context("spawn").Pipe()
        process = mp.get_context("spawn").Process(target=simulation_worker, args=(child,))
        process.start(); child.close()
        try:
            self.assertTrue(parent.poll(30)); self.assertEqual(parent.recv(), "READY")
            model = torch.nn.Linear(64,64)
            optimizer = torch.optim.SGD(model.parameters(), lr=.001)
            data = torch.ones(16,64)
            # Warm optimizer/library initialization before the timed dispatch.
            model(data).square().mean().backward(); optimizer.step()
            class Vector:
                def any_ready(self): return parent.poll(0)
            def learn():
                optimizer.zero_grad(); model(data).square().mean().backward(); optimizer.step()
                return True
            credit = TransitionCredit(.25,256); credit.collect(16)
            parent.send("STEP")
            intervals = []
            used = overlap_burst(Vector(), credit, learn, 4, intervals)
            self.assertTrue(parent.poll(30))
            results = [(0,parent.recv())]
            self.assertGreater(used,0)
            self.assertGreater(interval_overlap(intervals,results),0)
            self.assertLessEqual(credit.max_collector_lag_seen,256)
        finally:
            parent.close(); process.join(5)
            if process.is_alive(): process.terminate(); process.join()

    def test_credit_equivalence_throttle_and_no_unused_credits(self):
        credit = TransitionCredit(.25, 256)
        credit.collect(256)
        self.assertEqual(credit.lag, 256)
        self.assertTrue(credit.must_throttle(16))
        for _ in range(64): credit.consume()
        self.assertEqual(credit.lag, 0)
        self.assertEqual(credit.collector_transition_head, credit.learner_consumed_transition_equivalent)
        self.assertFalse(credit.must_throttle(16))

    def test_poll_between_atomic_updates_and_actual_interval_intersection(self):
        credit = TransitionCredit(.25, 256); credit.collect(16)
        class Vector:
            def any_ready(self): return True
        intervals = []
        used = overlap_burst(Vector(), credit, lambda: True, 4, intervals)
        self.assertEqual(used, 1)
        self.assertEqual(credit.updates_due, 3)
        a, b = intervals[0]
        results = [(0, ("OK", None, 0, False, False,
                        {"_stage3_env_started": a-1, "_stage3_env_finished": b+1}))]
        self.assertGreater(interval_overlap(intervals, results), 0)


class TestDiagnosticMath(unittest.TestCase):
    def test_plateau_rejects_cumulative_worsening(self):
        history = [{"td_mae":v, "qmin_mean":1., "qmin_std":1.} for v in (1.,1.04,1.08)]
        plateau, worsening, stable = stability(history, CONFIG["critic_readiness"])
        self.assertFalse(plateau); self.assertTrue(worsening)
        history = [{"td_mae":v, "qmin_mean":1., "qmin_std":1.} for v in (1.,1.01,1.02)]
        self.assertTrue(stability(history, CONFIG["critic_readiness"])[0])

    def test_shared_target_termination_and_truncation_do_not_bootstrap(self):
        agent = RecurrentGMMTD3(ToyActor(), ConstantTwin(), CONFIG, torch.device("cpu"), torch.ones(14), torch.zeros(14))
        batch = {key:np.stack([episode()[key][:11]]*2) for key in
                 ("observations","next_observations","actions","rewards","terminals","episode_steps")}
        # Index 0 is a true termination and index 1 represents a truncation
        # already mapped by replay to the same Bellman terminal mask.
        batch["terminals"][:, -1] = 1
        final = {key:value[:,-1] for key,value in batch.items() if key in
                 ("observations","actions","rewards","terminals")}
        target = agent.bellman_target(agent._tensor_batch(final), batch)[0]
        np.testing.assert_allclose(target.numpy().reshape(-1), [1.,1.], atol=1e-6)
        np.testing.assert_allclose(agent.fixed_td_diagnostic(batch)["td_target"], target.numpy().reshape(-1))

    def test_actor_prefetch_exact_indices_and_critic_source_log_is_clean(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [str(Path(directory)/f"{i}.h5") for i in range(3)]
            for path in paths: make_file(path)
            offline = BalancedOfflineDemonstrations(paths, 4)
            online = OnlineSequenceReplay(1000, 4); online.add_episode(episode())
            critics, actors = prepare_round_batches(offline, online, 16, CONFIG, torch.device("cpu"),
                actor_ready=True, update_count=4, critic_updates=100, transfer=False)
            self.assertEqual(len(critics), 4); self.assertEqual(list(actors), [3])
            counters = source_sample_metrics(offline)
            self.assertEqual(sum(counters[f"critic_offline_{s}_samples"] for s in ("rnn","transformer","gmm")), 512)
            self.assertEqual(sum(counters[f"actor_offline_{s}_samples"] for s in ("rnn","transformer","gmm")), 32)
            self.assertLessEqual(max(counters[f"critic_offline_{s}_samples"] for s in ("rnn","transformer","gmm"))-
                                 min(counters[f"critic_offline_{s}_samples"] for s in ("rnn","transformer","gmm")), 1)


class TestTrainerPipeline(unittest.TestCase):
    def test_real_learner_through_sync_and_async_trainer_control_flow(self):
        """Synthetic env fixture; not a MuJoCo/NPU performance measurement."""
        import train_stage3_v5_vector as trainer
        import stage3_v5_actor as actor_module
        import stage3_v5_agent as agent_module
        import torch._dynamo  # Initialize once before patch.dict restores sys.modules.
        from stage3_v5_actor import module_hash
        obs = flat_to_obs(np.zeros(59,np.float32))
        class Vector:
            def __init__(self, dataset, num_envs, seed, **kwargs):
                self.initial_observations = [obs]*num_envs
                self.action_low, self.action_high = -np.ones(14), np.ones(14)
                self.processes, self.thread = [], None
                self.steps = [0]*num_envs
            def step_async(self, actions, ids):
                def simulate():
                    start=time.perf_counter(); time.sleep(.02 if min(self.steps)>=500 else .002)
                    self.results=[]
                    for i in ids:
                        self.steps[i]+=1
                        self.results.append((i,("OK",obs,1.,self.steps[i]%22==0,False,
                            {"_stage3_env_started":start,"_stage3_env_finished":time.perf_counter()})))
                self.thread=threading.Thread(target=simulate); self.thread.start()
            def any_ready(self): return not self.thread.is_alive()
            def step_wait(self, ids): self.thread.join(); return self.results
            def reset_many(self, seeds, rebuild=False): return {i:obs for i in seeds}
            def close(self):
                if self.thread is not None: self.thread.join()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); shared=root/"shared"; shared.mkdir()
            paths=[str(root/f"{i}.h5") for i in range(3)]
            for path in paths: make_file(path)
            config=copy.deepcopy(CONFIG)
            config.update({"bc_rnn_checkpoint":str(shared/"actor_init.pth"),
                           "expert_dataset":str(root/"unused.h5"),"bc_rnn_checkpoint_sha256":"synthetic"})
            config["offline_sources"]=dict(zip(("bc_rnn","bc_transformer","bc_gmm"),paths))
            actor=ToyActor()
            torch.save({"actor_state_dict":actor.state_dict()},shared/"actor_init.pth")
            (shared/"config_resolved.json").write_text(json.dumps(config))
            (shared/"pair_fairness.json").write_text(json.dumps({"actor_hashes_identical":True,"actor_hash":module_hash(actor)}))
            checkpoint=str(root/"critic.pth")
            (shared/"stage2_source_manifest.json").write_text(json.dumps({"multi_q":{"checkpoint":checkpoint}}))
            evaluation=types.ModuleType("stage3_v3_evaluation")
            evaluation.build_env=lambda _:None; evaluation.close_env=lambda _:None
            evaluation.evaluate_actor=lambda *args:None
            def load_actor(*args):
                rollout=types.SimpleNamespace(action_normalization_stats={"actions":{"scale":np.ones(14),"offset":np.zeros(14)}})
                return ToyActor(),rollout,{}
            reports=[]
            for mode in ("C","D"):
                argv=["train", "--group","multi_q","--device","cpu","--pair-run-dir",str(root),
                      "--critic-init-checkpoint",checkpoint,"--benchmark-mode",mode,
                      "--num-envs","2","--benchmark-warmup-steps","1024","--total-env-steps","1032"]
                with patch.object(sys,"argv",argv), patch.object(trainer,"StaggeredVectorEnv",Vector), \
                     patch.object(actor_module,"load_exact_actor",load_actor), \
                     patch.object(agent_module,"strict_stage2_load",lambda *args:(ConstantTwin(),{})), \
                     patch.dict(sys.modules,{"stage3_v3_evaluation":evaluation}),contextlib.redirect_stdout(io.StringIO()):
                    trainer.main()
                files=list((root/"multi_q"/"benchmarks").glob(f"{mode}_*/measurements.json"))
                report=json.loads(files[-1].read_text()); reports.append(report)
                self.assertEqual(report["measured_transitions"],8)
                self.assertLessEqual(report["max_collector_lag_seen"],256)
                self.assertAlmostEqual(report["effective_utd"],8/33)
                self.assertEqual(sum(report[f"critic_offline_{s}_samples"] for s in ("rnn","transformer","gmm")),1024)
                self.assertEqual(sum(report[f"actor_offline_{s}_samples"] for s in ("rnn","transformer","gmm")),0)
            self.assertGreater(reports[1]["overlap_critic_updates"],0)
            self.assertGreater(reports[1]["measured_overlap_ms"],0)


if __name__ == "__main__": unittest.main()
