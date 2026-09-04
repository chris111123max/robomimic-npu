#!/usr/bin/env python3
"""CPU-only synthetic HDF5 smoke test for Stage2-new.

This intentionally mirrors the audited Stage1 schema; it never touches the
server dataset configured for formal runs.
"""
from __future__ import annotations
import json, subprocess, sys, tempfile
from pathlib import Path
import h5py, numpy as np, torch
from critic_network import build_critic, load_stage2_critic_checkpoint
from stage2_new_dataset import CANONICAL_KEYS, POLICIES, load_split_datasets, monte_carlo_returns
from stage2_new_evaluation import _auc, action_gradient_diagnostics, evaluate_critic
from stage2_new_sampler import MultiPolicyBalancedSampler, RNNTransitionSampler

def write_fixture(root):
    shapes = {"robot0_eef_pos":3,"robot0_eef_quat":4,"robot0_gripper_qpos":2,"robot1_eef_pos":3,"robot1_eef_quat":4,"robot1_gripper_qpos":2,"object":41}
    for pidx, policy in enumerate(POLICIES):
        path = root / policy / "transitions.hdf5"; path.parent.mkdir(parents=True)
        with h5py.File(path,"w") as file:
            file.attrs["policy_id"] = policy; file.attrs["canonical_observation_keys"] = json.dumps(list(CANONICAL_KEYS)); episodes = file.create_group("episodes")
            for episode_id, seed in enumerate(range(10000,10100)):
                group=episodes.create_group(f"episode_{episode_id:06d}"); length=3; success=(seed+pidx)%2==0
                obs, nxt = group.create_group("obs"), group.create_group("next_obs")
                for key, width in shapes.items():
                    value=np.full((length,width), float(pidx+episode_id)/100, np.float32); obs.create_dataset(key,data=value); nxt.create_dataset(key,data=value+0.01)
                group.create_dataset("actions",data=np.full((length,14),pidx/10,np.float32)); group.create_dataset("rewards",data=np.array([0,0,1 if success else 0],np.float32)); group.create_dataset("dones",data=np.array([0,0,1],bool)); group.create_dataset("terminated",data=np.array([0,0,0],bool)); group.create_dataset("truncated",data=np.array([0,0,1],bool))
                for name,value,dtype in (("episode_id",episode_id,np.int64),("initial_seed",seed,np.int64),("episode_success",success,np.bool_),("episode_length",length,np.int64)):
                    group.create_dataset(name,data=np.full(length,value,dtype=dtype)); group.attrs[name]=value

def main():
    assert np.allclose(monte_carlo_returns([0,0,1],.99),[.99**2,.99,1])
    assert np.allclose(monte_carlo_returns([0,0,0],.99),[0,0,0])
    assert np.allclose(np.concatenate((monte_carlo_returns([0,0,1],.99),monte_carlo_returns([0,0,0],.99))),[.99**2,.99,1,0,0,0])
    with tempfile.TemporaryDirectory(prefix="stage2_new_smoke_") as temporary:
        root=Path(temporary)/"data"; out=Path(temporary)/"out"; write_fixture(root)
        train,val=load_split_datasets(root,range(10000,10080),range(10080,10100),.99)
        assert train["bc_rnn"].transition_count==240 and val["bc_gmm"].transition_count==60
        assert not (set(train["bc_rnn"].seeds)&set(val["bc_rnn"].seeds))
        assert RNNTransitionSampler(train["bc_rnn"],1).sample(17)["state"].shape==(17,59)
        sampler=MultiPolicyBalancedSampler(train,2)
        for _ in range(300): sampler.sample(257)
        proportions=sampler.proportions(); assert all(abs(value-1/3)<0.002 for value in proportions.values()), proportions
        critic=build_critic(device="cpu"); modules=list(critic.q1.network); assert [type(x).__name__ for x in modules]==["Linear","LayerNorm","ReLU","Linear","LayerNorm","ReLU","Linear"], modules
        state=torch.randn(8,59); action=torch.randn(8,14); q1,q2=critic(state,action); assert q1.shape==q2.shape==(8,1)
        optimizer=torch.optim.AdamW(critic.parameters(),lr=3e-4,weight_decay=1e-4); assert optimizer.param_groups[0]["weight_decay"]==1e-4
        loss=(q1.square()+q2.square()).mean(); optimizer.zero_grad(); loss.backward(); optimizer.step()
        check=Path(temporary)/"roundtrip.pth"; torch.save({"critic_state_dict":critic.state_dict(),"model_config":{"obs_dim":59,"action_dim":14,"hidden_dims":[256,256],"activation":"relu","layer_norm":True}},check)
        restored,_=load_stage2_critic_checkpoint(check,"cpu"); assert torch.equal(critic(state,action)[0],restored(state,action)[0])
        gradients=action_gradient_diagnostics(critic,state.numpy(),action.numpy(),"cpu"); assert gradients["q1_gradients"].shape==(8,14) and gradients["q2_norm"]["mean"] is not None
        metrics=evaluate_critic(critic,val,"cpu"); assert "balanced_aggregate" in metrics and _auc([.1,.2],[False,False])["value"] is None
        script=Path(__file__).with_name("train_stage2_new_critics.py")
        subprocess.run([sys.executable,str(script),"--dataset-root",str(root),"--output-root",str(out),"--device","cpu","--mode","both","--max-updates","2","--batch-size","8","--run-id","cli_override"],check=True)
        run=out/"cli_override"; assert (run/"rnn_q"/"checkpoints"/"best.pth").is_file() and (run/"multi_q"/"checkpoints"/"best.pth").is_file()
        print(json.dumps({"status":"PASS","multi_policy_proportions":proportions,"run":str(run)}))
if __name__=="__main__": main()
