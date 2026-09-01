"""Stage2-R-v2 clipped recurrent critic update and failure capture."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from stage2_r_critic import aligned


class NonFiniteGradientError(RuntimeError):
    pass


def cpu_tree(value):
    if torch.is_tensor(value):return value.detach().cpu()
    if isinstance(value,dict):return {key:cpu_tree(item) for key,item in value.items()}
    if isinstance(value,list):return [cpu_tree(item) for item in value]
    if isinstance(value,tuple):return tuple(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def finite_stats(value):
    finite=value.detach()[torch.isfinite(value)]
    if finite.numel()==0:return {"mean":None,"std":None,"min":None,"max":None}
    return {"mean":float(finite.mean().item()),"std":float(finite.std(unbiased=False).item()),
            "min":float(finite.min().item()),"max":float(finite.max().item())}


def recurrent_diagnostics(critic,previous_actions,previous_rewards,observs):
    action=critic.action_embedder(previous_actions);reward=critic.reward_embedder(previous_rewards);obs=critic._get_obs_embedding(observs)
    output,state=critic.rnn(torch.cat((action,reward,obs),dim=-1))
    if isinstance(state,tuple):hidden,cell=state
    else:hidden,cell=state,None
    return output,{"hidden_norm":float(hidden.detach().float().norm().item()),
                   "cell_norm":None if cell is None else float(cell.detach().float().norm().item())}


def forward_values(critic,target,batch,gamma):
    observs,previous_actions,previous_rewards,target_actions=aligned(batch)
    q1,q2=critic(previous_actions,previous_rewards,observs,batch["act"])
    with torch.no_grad():
        tq1,tq2=target(previous_actions,previous_rewards,observs,target_actions)
        target_q=torch.minimum(tq1[1:],tq2[1:]);bellman=batch["rew"]+float(gamma)*(1.0-batch["term"])*target_q
    output,state_norms=recurrent_diagnostics(critic,previous_actions,previous_rewards,observs)
    return q1,q2,target_q,bellman,output,state_norms


def save_failure(debug_root,index,group,batch,critic,target,optimizer,diagnostics):
    directory=Path(debug_root)/f"update_{int(index):08d}";directory.mkdir(parents=True,exist_ok=False)
    torch.save(cpu_tree(batch),directory/"batch.pt")
    torch.save(cpu_tree(critic.state_dict()),directory/"critic_before_failure.pth")
    torch.save(cpu_tree(target.state_dict()),directory/"target_critic_before_failure.pth")
    torch.save(cpu_tree(optimizer.state_dict()),directory/"optimizer_before_failure.pth")
    diagnostics={"update":int(index),"group":group,**diagnostics}
    (directory/"nan_diagnostics.json").write_text(json.dumps(diagnostics,indent=2,ensure_ascii=False),encoding="utf-8")
    return directory


def update_v2(critic,target,optimizer,batch,config,index,group,debug_root,soft_update):
    q1,q2,target_q,bellman,hidden,state_norms=forward_values(critic,target,batch,config["gamma"])
    mask=batch["mask"];valid=torch.clamp(mask.sum(),min=1.0)
    loss1=(((q1-bellman)**2)*mask).sum()/valid;loss2=(((q2-bellman)**2)*mask).sum()/valid;loss=loss1+loss2
    forward=(q1,q2,target_q,bellman,hidden,loss)
    if not all(torch.isfinite(value).all() for value in forward):
        diagnostics=diagnostic_payload(loss,q1,q2,target_q,bellman,batch,state_norms,0,[])
        directory=save_failure(debug_root,index,group,batch,critic,target,optimizer,diagnostics)
        raise NonFiniteGradientError(f"Non-finite forward value at update {index}; saved {directory}")
    optimizer.zero_grad(set_to_none=True);loss.backward()
    bad_names=[];bad_count=0
    for name,parameter in critic.named_parameters():
        if parameter.grad is None:continue
        count=int((~torch.isfinite(parameter.grad)).sum().item())
        if count:bad_names.append(name);bad_count+=count
    if bad_count:
        diagnostics=diagnostic_payload(loss,q1,q2,target_q,bellman,batch,state_norms,bad_count,bad_names)
        directory=save_failure(debug_root,index,group,batch,critic,target,optimizer,diagnostics)
        optimizer.zero_grad(set_to_none=True)
        raise NonFiniteGradientError(f"NaN/Inf gradients at update {index}: count={bad_count}; saved {directory}")
    try:
        grad_norm=torch.nn.utils.clip_grad_norm_(critic.parameters(),max_norm=float(config["max_gradient_norm"]),
                                                 norm_type=float(config["gradient_norm_type"]),error_if_nonfinite=True,foreach=False)
    except RuntimeError as error:
        diagnostics=diagnostic_payload(loss,q1,q2,target_q,bellman,batch,state_norms,0,["global_norm"]);diagnostics["clip_error"]=str(error)
        directory=save_failure(debug_root,index,group,batch,critic,target,optimizer,diagnostics)
        optimizer.zero_grad(set_to_none=True)
        raise NonFiniteGradientError(f"Non-finite global gradient norm at update {index}; saved {directory}") from error
    preclip=float(grad_norm.item());optimizer.step()
    if index%int(config["target_update_interval"])==0:soft_update(critic,target,float(config["tau"]))
    if any(not torch.isfinite(parameter).all() for parameter in critic.parameters()):
        raise RuntimeError(f"Optimizer produced non-finite critic parameters at update {index}")
    return {"critic_loss":float(loss.item()),"q1_loss":float(loss1.item()),"q2_loss":float(loss2.item()),
            "q_mean":float(torch.minimum(q1,q2)[mask.bool()].mean().item()),"target_q_mean":float(bellman[mask.bool()].mean().item()),
            "preclip_grad_norm":preclip,"gradient_clipped":float(preclip>float(config["max_gradient_norm"])),
            "hidden_norm":state_norms["hidden_norm"],"cell_norm":state_norms["cell_norm"],"nan_inf_count":0,
            "effective_timesteps":int(mask.sum().item())}


def diagnostic_payload(loss,q1,q2,target_q,bellman,batch,state_norms,bad_count,bad_names):
    q_pred=torch.cat((q1.detach().reshape(-1),q2.detach().reshape(-1)))
    return {"loss":float(loss.detach().item()) if torch.isfinite(loss) else str(loss.detach().item()),
            "q_pred":finite_stats(q_pred),"target_q":finite_stats(target_q),"td_target":finite_stats(bellman),
            "reward_min":float(batch["rew"].min().item()),"reward_max":float(batch["rew"].max().item()),
            "action_min":float(batch["act"].min().item()),"action_max":float(batch["act"].max().item()),
            "obs_min":float(batch["obs"][...,:59].min().item()),"obs_max":float(batch["obs"][...,:59].max().item()),
            "next_obs_min":float(batch["obs2"][...,:59].min().item()),"next_obs_max":float(batch["obs2"][...,:59].max().item()),
            **state_norms,"bad_gradient_count":int(bad_count),"bad_parameter_names":bad_names}
