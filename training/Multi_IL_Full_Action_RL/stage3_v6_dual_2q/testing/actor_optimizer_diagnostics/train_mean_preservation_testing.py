"""Testing-only real continuation for component-mean preservation necessity.

Derives the current production Stage3-v6 trainer at runtime and changes only:
  * RecurrentGMMTD3 -> MeanPreservationAgent,
  * run type -> testing-only,
  * checkpoint / diagnostics milestones -> 130/140/150/160K,
  * checkpoint metadata -> testing-only mean-preservation branch.

FULL_PRODUCTION dispatches to the inherited production Actor update unchanged.
FULL_MEAN_PRESERVATION adds only the component-mean output penalty.
"""
from __future__ import annotations

import hashlib
import inspect
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V6_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[4]
for folder in (REPO_ROOT, V6_DIR, HERE):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))


def main():
    import train_stage3_v6_vector as P
    import stage3_v6_agent as V6
    import mean_preservation_agent as A
    import module_online_diagnostics as D

    branch = os.environ["MEAN_PRESERVATION_BRANCH"]
    if branch not in A.BRANCHES:
        raise RuntimeError(branch)

    source_checkpoint = Path(
        os.environ["MEAN_PRESERVATION_SOURCE_CHECKPOINT"]
    ).resolve()

    if "--quad-run-dir" not in sys.argv:
        raise RuntimeError("--quad-run-dir required")
    pair = Path(sys.argv[sys.argv.index("--quad-run-dir") + 1]).resolve()
    if not pair.is_relative_to(HERE):
        raise RuntimeError(f"Testing pair escaped testing directory: {pair}")

    if (
        "--target-mode" not in sys.argv
        or sys.argv[sys.argv.index("--target-mode") + 1] != "random2q"
    ):
        raise RuntimeError("Mean-preservation continuation is random2q-only")
    if (
        "--group" not in sys.argv
        or sys.argv[sys.argv.index("--group") + 1] != "multi_q"
    ):
        raise RuntimeError("Mean-preservation continuation is multi_q-only")

    total = int(sys.argv[sys.argv.index("--total-env-steps") + 1])
    if total != 160000:
        raise RuntimeError("Mean-preservation continuation is fixed to 160000")

    if "--resume" not in sys.argv:
        raise RuntimeError("Mean-preservation continuation must resume")
    resume = Path(sys.argv[sys.argv.index("--resume") + 1]).resolve()
    if resume != source_checkpoint:
        raise RuntimeError(f"Unexpected resume checkpoint: {resume}")

    production_files = [
        Path(P.__file__).resolve(),
        (V6_DIR / "stage3_v6_agent.py").resolve(),
        (V6_DIR.parent / "stage3_v5_rgmm_td3" / "stage3_v5_agent.py").resolve(),
        (V6_DIR / "stage3_v6_config.json").resolve(),
    ]
    hashes_before = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in production_files
    }

    V6.RecurrentGMMTD3 = A.MeanPreservationAgent

    original_restore = P.restore_checkpoint
    def restore_checkpoint(path, agent, config, torch, OnlineSequenceReplay):
        payload, online = original_restore(
            path, agent, config, torch, OnlineSequenceReplay
        )
        agent.configure_after_resume(payload)
        return payload, online
    P.restore_checkpoint = restore_checkpoint

    original_payload = P.checkpoint_payload
    def checkpoint_payload(*args, **kwargs):
        payload = original_payload(*args, **kwargs)
        payload["testing_only"] = True
        payload["mean_preservation_branch"] = branch
        payload["mean_preservation_contract"] = {
            "mean_rms_safe": A.MEAN_RMS_SAFE,
            "mean_rms_hard": A.MEAN_RMS_HARD,
            "max_anchor_RL_grad_ratio": A.MAX_ANCHOR_RL_GRAD_RATIO,
            "max_lambda": A.MAX_LAMBDA,
            "anchor_terms": ["gmm_component_means"],
            "explicitly_unanchored": ["logits", "std", "hidden_state"],
        }
        payload["production_source_sha256"] = hashes_before
        return payload
    P.checkpoint_payload = checkpoint_payload

    original_save = P.save_checkpoint
    def guarded_save(path, *args, **kwargs):
        resolved = Path(path).resolve()
        if not resolved.is_relative_to(HERE):
            raise RuntimeError(f"Testing checkpoint escaped testing root: {resolved}")
        return original_save(path, *args, **kwargs)
    P.save_checkpoint = guarded_save

    source = inspect.getsource(P.main)

    run_type_old = (
        'config["run_type"] = "BENCHMARK" if args.benchmark_mode '
        'else ("SMOKE" if args.smoke else "FORMAL")'
    )
    if source.count(run_type_old) != 1:
        raise RuntimeError("Production run_type source changed")
    source = source.replace(
        run_type_old,
        'config["run_type"] = "TESTING_ONLY_MEAN_PRESERVATION_NECESSITY"',
        1,
    )

    checkpoint_old = (
        '        if args.benchmark_mode:\n'
        '            checkpoint_steps.add(int(args.benchmark_warmup_steps))'
    )
    if source.count(checkpoint_old) != 1:
        raise RuntimeError("Production checkpoint schedule source changed")
    source = source.replace(
        checkpoint_old,
        '        checkpoint_steps = {140000, 150000, 160000, total}\n'
        + checkpoint_old,
        1,
    )

    start_marker = "        metric_rows = []\n        active_cursor = 0"
    if source.count(start_marker) != 1:
        raise RuntimeError("Production metric-row marker changed")
    source = source.replace(
        start_marker,
        '        testing_milestone(agent, config, env_steps, group_dir, torch, '
        'source_checkpoint)\n'
        + start_marker,
        1,
    )

    checkpoint_tail = (
        '                                    episodes, successes, online, torch, '
        'handoff, credit.pending, offline)\n\n'
        '            # Reset all completed workers'
    )
    if source.count(checkpoint_tail) != 1:
        raise RuntimeError("Production in-loop checkpoint source changed")
    source = source.replace(
        checkpoint_tail,
        '                                    episodes, successes, online, torch, '
        'handoff, credit.pending, offline)\n'
        '                    testing_milestone(agent, config, env_steps, '
        'group_dir, torch, source_checkpoint)\n\n'
        '            # Reset all completed workers',
        1,
    )

    namespace = dict(vars(P))
    namespace["testing_milestone"] = D.testing_milestone
    namespace["source_checkpoint"] = source_checkpoint

    exec(
        compile(
            source,
            str(HERE / "derived_mean_preservation_production_main"),
            "exec",
        ),
        namespace,
    )

    if "--validate-only" in sys.argv:
        print("MEAN_PRESERVATION_DERIVED_MAIN_COMPILE_PASS", flush=True)
        return

    namespace["main"]()

    hashes_after = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in production_files
    }
    if hashes_after != hashes_before:
        raise RuntimeError("Production source changed during experiment")

    print("MEAN_PRESERVATION_TESTING_TRAINER_EXITED_ALL_VECTOR_ENVS_CLOSED", flush=True)
    print("FORMAL TRAINING REMAINS STOPPED", flush=True)


if __name__ == "__main__":
    main()
