"""BC-RNN-compatible recurrent actor using pomdp-baselines' SAC Gaussian head."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
VENDOR = HERE.parent / "third_party" / "pomdp_baselines"
if str(VENDOR) not in sys.path: sys.path.insert(0, str(VENDOR))
from policies.models.actor import TanhGaussianPolicy  # noqa: E402
import torchkit.pytorch_utils as ptu  # noqa: E402

CANONICAL_KEYS = ["robot0_eef_pos","robot0_eef_quat","robot0_gripper_qpos",
                  "robot1_eef_pos","robot1_eef_quat","robot1_gripper_qpos","object"]
BC_RNN_KEYS = ["object","robot0_eef_pos","robot0_eef_quat","robot0_gripper_qpos",
               "robot1_eef_pos","robot1_eef_quat","robot1_gripper_qpos"]
SHAPES = {"robot0_eef_pos":3,"robot0_eef_quat":4,"robot0_gripper_qpos":2,
          "robot1_eef_pos":3,"robot1_eef_quat":4,"robot1_gripper_qpos":2,"object":41}

def slices(order):
    out={}; cursor=0
    for key in order: out[key]=slice(cursor,cursor+SHAPES[key]); cursor+=SHAPES[key]
    return out

class ActorObservationAdapter(nn.Module):
    def __init__(self):
        super().__init__(); cs=slices(CANONICAL_KEYS)
        index=torch.tensor([i for key in BC_RNN_KEYS for i in range(cs[key].start,cs[key].stop)],dtype=torch.long)
        self.register_buffer("canonical_to_bc_rnn_index",index,persistent=True)
    def forward(self,state): return state.index_select(-1,self.canonical_to_bc_rnn_index)

class Stage3RActor(nn.Module):
    def __init__(self,initial_log_std=-3.0,horizon=10):
        super().__init__(); self.observation_adapter=ActorObservationAdapter()
        self.lstm=nn.LSTM(59,400,num_layers=2,batch_first=True,bidirectional=False,dropout=0.0)
        self.policy=TanhGaussianPolicy(obs_dim=400,action_dim=14,hidden_sizes=[])
        nn.init.constant_(self.policy.last_fc_log_std.weight,0.0); nn.init.constant_(self.policy.last_fc_log_std.bias,initial_log_std)
        self.horizon=int(horizon); self._state=None; self._counter=0
    def reset(self): self._state=None; self._counter=0
    def recurrent_features(self,canonical,initial_state=None):
        return self.lstm(self.observation_adapter(canonical),initial_state)
    def forward_sequence(self,canonical,reset_interval=True):
        if not reset_interval: return self.recurrent_features(canonical)[0]
        chunks=[]
        for start in range(0,canonical.shape[1],self.horizon): chunks.append(self.recurrent_features(canonical[:,start:start+self.horizon])[0])
        return torch.cat(chunks,dim=1)
    def actions_from_features(self,features,deterministic=True,return_log_prob=False):
        return self.policy(features,deterministic=deterministic,return_log_prob=return_log_prob)
    def deterministic_sequence(self,canonical): return self.actions_from_features(self.forward_sequence(canonical),True)[0]
    @torch.no_grad()
    def act(self,canonical_state,deterministic=True,return_log_prob=False):
        if self._state is None or self._counter % self.horizon == 0: self._state=None
        x=self.observation_adapter(canonical_state).unsqueeze(1); out,self._state=self.lstm(x,self._state); self._counter+=1
        return self.policy(out[:,0],deterministic=deterministic,return_log_prob=return_log_prob)

def transfer_lstm(teacher,student):
    source=teacher.nets["rnn"].nets
    names=[f"{kind}_{part}_l{layer}" for layer in range(2) for kind in ("weight","bias") for part in ("ih","hh")]
    mapped={}
    with torch.no_grad():
        for name in names:
            src=getattr(source,name); dst=getattr(student.lstm,name)
            if src.shape!=dst.shape: raise RuntimeError(f"LSTM shape mismatch {name}: {src.shape} != {dst.shape}")
            dst.copy_(src); mapped[name]=list(src.shape)
    return mapped

def set_vendor_device(device): ptu.set_device(device)

def checkpoint_payload(actor,epoch,config,validation,optimizer=None):
    return {"format_version":"multi_il_full_action_rl.stage3_r.actor.v1","stage":"Stage3-R","epoch":int(epoch),
            "actor_state_dict":actor.state_dict(),"optimizer_state_dict":None if optimizer is None else optimizer.state_dict(),
            "architecture":{"state_dim":59,"action_dim":14,"lstm_hidden_size":400,"lstm_num_layers":2,"batch_first":True,"horizon":10,"sac_head":"pomdp-baselines TanhGaussianPolicy"},
            "observation_keys":CANONICAL_KEYS,"actor_observation_order":BC_RNN_KEYS,"teacher_checkpoint":config["teacher_checkpoint"],
            "source_dataset":config["dataset"],"validation":validation,"log_std_frozen":True,"initial_log_std":config["initial_log_std"],
            "evaluation_horizon":config["evaluation_horizon"],"terminate_on_success":config["terminate_on_success"],
            "pomdp_baselines_commit":config["pomdp_baselines_commit"]}

def load_actor(path,device):
    payload=torch.load(path,map_location=device); actor=Stage3RActor(payload["initial_log_std"],payload["architecture"]["horizon"]).to(device)
    actor.load_state_dict(payload["actor_state_dict"],strict=True); actor.eval(); set_vendor_device(device); return actor,payload
