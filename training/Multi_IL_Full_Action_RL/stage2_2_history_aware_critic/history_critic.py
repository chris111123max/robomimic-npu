"""Independent history-aware Twin-Q networks for Stage2.2."""
from __future__ import annotations
import copy
import torch
from torch import nn


class HistoryQNetwork(nn.Module):
    def __init__(self, obs_dim=59, action_dim=14, token_dim=64, hidden_dim=96,
                 layers=1, head_hidden_dim=128):
        super().__init__()
        self.obs_dim, self.action_dim = int(obs_dim), int(action_dim)
        self.token_schema = ("observation_t", "previous_executed_action_t", "normalized_episode_progress_t")
        self.token_encoder = nn.Sequential(
            nn.Linear(self.obs_dim + self.action_dim + 1, int(token_dim)),
            nn.LayerNorm(int(token_dim)), nn.ReLU())
        self.lstm = nn.LSTM(int(token_dim), int(hidden_dim), int(layers), batch_first=True)
        self.q_head = nn.Sequential(nn.Linear(int(hidden_dim) + self.action_dim, int(head_hidden_dim)),
                                    nn.LayerNorm(int(head_hidden_dim)), nn.ReLU(),
                                    nn.Linear(int(head_hidden_dim), 1))

    def _tokens(self, observations, previous_actions, progress):
        if progress.ndim == observations.ndim - 1:
            progress = progress.unsqueeze(-1)
        return self.token_encoder(torch.cat((observations, previous_actions, progress), dim=-1))

    def encode_history(self, observations, previous_actions, progress, state=None):
        encoded, state = self.lstm(self._tokens(observations, previous_actions, progress), state)
        return encoded, state

    def diagnostic_forward(self, observations, previous_actions, progress, current_actions):
        token = self._tokens(observations, previous_actions, progress)
        context, state = self.lstm(token)
        q = self.q_from_context(context, current_actions)
        return q, {"token_embedding":token,"context":context,"final_hidden":state[0],"final_cell":state[1]}

    def q_from_context(self, context, action):
        if action.ndim == context.ndim:
            return self.q_head(torch.cat((context, action), dim=-1))
        if action.ndim == context.ndim + 1:
            expanded = context.unsqueeze(-2).expand(*action.shape[:-1], context.shape[-1])
            return self.q_head(torch.cat((expanded, action), dim=-1))
        raise ValueError(f"Unsupported context/action ranks: {context.shape}, {action.shape}")

    def forward_sequence(self, observations, previous_actions, progress, current_actions,
                         state=None, burn_in=0):
        burn_in = int(burn_in)
        if burn_in:
            with torch.no_grad():
                _, state = self.encode_history(observations[:, :burn_in], previous_actions[:, :burn_in],
                                               progress[:, :burn_in], state)
            observations, previous_actions = observations[:, burn_in:], previous_actions[:, burn_in:]
            progress, current_actions = progress[:, burn_in:], current_actions[:, burn_in:]
        context, state = self.encode_history(observations, previous_actions, progress, state)
        return self.q_from_context(context, current_actions), state

    def advance_context(self, next_observation, executed_action, next_progress, state):
        context, state = self.encode_history(next_observation.unsqueeze(1), executed_action.unsqueeze(1),
                                             next_progress.unsqueeze(1), state)
        return context[:, 0], state


class HistoryAwareTwinQ(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.q1 = HistoryQNetwork(**kwargs)
        self.q2 = HistoryQNetwork(**kwargs)

    def encode_history(self, observations, previous_actions, progress, states=(None, None)):
        z1, s1 = self.q1.encode_history(observations, previous_actions, progress, states[0])
        z2, s2 = self.q2.encode_history(observations, previous_actions, progress, states[1])
        return (z1, z2), (s1, s2)

    def q_from_context(self, contexts, actions):
        return self.q1.q_from_context(contexts[0], actions), self.q2.q_from_context(contexts[1], actions)

    def forward_sequence(self, observations, previous_actions, progress, actions, burn_in=0):
        q1, _ = self.q1.forward_sequence(observations, previous_actions, progress, actions, burn_in=burn_in)
        q2, _ = self.q2.forward_sequence(observations, previous_actions, progress, actions, burn_in=burn_in)
        return q1, q2

    def diagnostic_forward(self,observations,previous_actions,progress,actions):
        q1,d1=self.q1.diagnostic_forward(observations,previous_actions,progress,actions)
        q2,d2=self.q2.diagnostic_forward(observations,previous_actions,progress,actions)
        return (q1,q2),(d1,d2)


class MatchedMemorylessQ(nn.Module):
    """Matched control: current token plus candidate action, without history."""
    def __init__(self,obs_dim=59,action_dim=14,hidden_dims=(240,256)):
        super().__init__();self.obs_dim=int(obs_dim);self.action_dim=int(action_dim)
        first,second=map(int,hidden_dims)
        self.feature_encoder=nn.Sequential(nn.Linear(self.obs_dim+self.action_dim+1+self.action_dim,first),nn.LayerNorm(first),nn.ReLU(),nn.Linear(first,second),nn.LayerNorm(second),nn.ReLU())
        self.q_head=nn.Linear(second,1)

    def forward_sequence(self,observations,previous_actions,progress,actions):
        return self.q_head(self.feature_encoder(torch.cat((observations,previous_actions,progress,actions),-1)))

    def diagnostic_forward(self,observations,previous_actions,progress,actions):
        feature=self.feature_encoder(torch.cat((observations,previous_actions,progress,actions),-1));q=self.q_head(feature)
        return q,{"token_embedding":feature,"context":feature,"final_hidden":feature[:,-1:],"final_cell":feature[:,-1:]}


class MatchedMemorylessTwinQ(nn.Module):
    def __init__(self,**kwargs):super().__init__();self.q1=MatchedMemorylessQ(**kwargs);self.q2=MatchedMemorylessQ(**kwargs)
    def forward_sequence(self,observations,previous_actions,progress,actions,burn_in=0):
        del burn_in;return self.q1.forward_sequence(observations,previous_actions,progress,actions),self.q2.forward_sequence(observations,previous_actions,progress,actions)
    def diagnostic_forward(self,observations,previous_actions,progress,actions):
        q1,d1=self.q1.diagnostic_forward(observations,previous_actions,progress,actions);q2,d2=self.q2.diagnostic_forward(observations,previous_actions,progress,actions);return (q1,q2),(d1,d2)


def architecture_config(config):
    return {key: config[key] for key in ("obs_dim", "action_dim", "token_dim", "lstm_hidden_dim",
            "lstm_layers", "head_hidden_dim", "legacy_replay_burn_in_length", "learning_sequence_length","history_semantics","matched_hidden_dims")}


def build_critic(config, device=None, critic_type="history_aware_twin_q"):
    if critic_type=="matched_memoryless_twin_q":
        model=MatchedMemorylessTwinQ(obs_dim=config["obs_dim"],action_dim=config["action_dim"],hidden_dims=config["matched_hidden_dims"])
    elif critic_type=="history_aware_twin_q":
        model = HistoryAwareTwinQ(obs_dim=config["obs_dim"], action_dim=config["action_dim"],token_dim=config["token_dim"], hidden_dim=config["lstm_hidden_dim"],layers=config["lstm_layers"], head_hidden_dim=config["head_hidden_dim"])
    else:raise ValueError(f"unknown critic_type {critic_type}")
    return model if device is None else model.to(device)


def checkpoint_payload(model, optimizer, config, step, checkpoint_metric, best_metric=None,critic_type="history_aware_twin_q"):
    return {"stage_version":"2.2", "critic_type":critic_type,
            "architecture":architecture_config(config), "critic_state_dict":model.state_dict(),
            "optimizer_state_dict":optimizer.state_dict(), "step":int(step),"checkpoint_step":int(step),
            "checkpoint_validation_metric":None if checkpoint_metric is None else float(checkpoint_metric),
            "best_validation_metric":None if best_metric is None else float(best_metric),
            "normalization":copy.deepcopy(config["normalization"]), "token_schema":list(config["token_schema"]),
            "horizon":int(config["horizon"]), "training_target":config["training_target"],
            "loss":config["loss"],"gamma":float(config["gamma"]),"history_semantics":config["history_semantics"], "dataset_sources":{
                name:f'{config["dataset_root"]}/{name}/transitions.hdf5'
                for name in ("bc_rnn","bc_transformer","bc_gmm")},
            "sampling_ratios":{"rnn_q":[1,0,0],"multi_q":[1/3,1/3,1/3]},
            "random_seed":int(config["training_seed"]),
            "terminated_truncated_semantics":config["terminated_truncated_semantics"]}


def load_checkpoint(path, config, device="cpu",critic_type="history_aware_twin_q"):
    payload = torch.load(path, map_location=device)
    expected = {"stage_version":"2.2", "critic_type":critic_type,
                "architecture":architecture_config(config), "normalization":config["normalization"],
                "token_schema":config["token_schema"], "horizon":int(config["horizon"]),
                "training_target":config["training_target"],"loss":config["loss"],"gamma":float(config["gamma"]),
                "terminated_truncated_semantics":config["terminated_truncated_semantics"],"history_semantics":config["history_semantics"]}
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"Stage2.2 checkpoint contract mismatch for {key}: {payload.get(key)!r} != {value!r}")
    model = build_critic(config, device,critic_type)
    model.load_state_dict(payload["critic_state_dict"], strict=True)
    return model, payload
