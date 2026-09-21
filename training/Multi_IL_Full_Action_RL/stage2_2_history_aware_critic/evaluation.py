"""Shared held-out evaluation for every Stage2.2 variant."""
from __future__ import annotations
import numpy as np
import torch
from sequence_dataset import POLICIES, previous_actions

def ranks(x):
    """Zero-based average ranks with exact tie handling."""
    x=np.asarray(x);order=np.argsort(x,kind="mergesort");out=np.empty(len(x),float);start=0
    while start<len(x):
        end=start+1
        while end<len(x) and x[order[end]]==x[order[start]]:end+=1
        out[order[start:end]]=(start+end-1)/2.0;start=end
    return out
def corr(a,b):
    return None if len(a)<2 or np.std(a)==0 or np.std(b)==0 else float(np.corrcoef(a,b)[0,1])
def auc(scores,labels):
    labels=np.asarray(labels,bool); p=labels.sum(); n=(~labels).sum()
    return None if not p or not n else float((ranks(scores)[labels].sum()-p*(p-1)/2)/(p*n))
def metrics(q1,q2,target,success,progress):
    q=np.minimum(q1,q2); error=q-target
    result={"q1_mse":float(np.mean((q1-target)**2)),"q2_mse":float(np.mean((q2-target)**2)),
      "twin_mean_mse":float((np.mean((q1-target)**2)+np.mean((q2-target)**2))/2),"mae":float(np.mean(np.abs(error))),
      "spearman":corr(ranks(q),ranks(target)),"pearson":corr(q,target),"auc":auc(q,success),
      "q_mean":float(q.mean()),"q_std":float(q.std()),"q_min":float(q.min()),"q_max":float(q.max()),
      "success_q_mean":float(q[success].mean()) if success.any() else None,"failure_q_mean":float(q[~success].mean()) if (~success).any() else None}
    result["outcome_slices"]={}
    for name,mask in (("success",success),("failure",~success)):
        if mask.any(): result["outcome_slices"][name]={"count":int(mask.sum()),"q_mean":float(q[mask].mean()),"q_std":float(q[mask].std()),"target_mean":float(target[mask].mean()),"mae":float(np.mean(np.abs(error[mask]))),"mse":float(np.mean(error[mask]**2)),"spearman":corr(ranks(q[mask]),ranks(target[mask]))}
    slices={"early":progress<1/3,"middle":(progress>=1/3)&(progress<2/3),"late":progress>=2/3}
    result["progress_slices"]={name:{"count":int(mask.sum()),"mae":float(np.mean(np.abs(error[mask]))),"mse":float(np.mean(error[mask]**2)),"q_mean":float(q[mask].mean()),"target_mean":float(target[mask].mean())} for name,mask in slices.items() if mask.any()}
    return result

@torch.no_grad()
def evaluate(model,datasets,device,horizon=700):
    was_training=model.training;model.eval(); output={}
    for policy in POLICIES:
        q1s=[];q2s=[];targets=[];labels=[];progress=[]
        for e in datasets[policy].episodes:
            o=torch.as_tensor(e.observations[None],device=device); p=torch.as_tensor(previous_actions(e.actions)[None],device=device)
            t=torch.arange(e.length,device=device,dtype=torch.float32)[None,:,None]/float(horizon); a=torch.as_tensor(e.actions[None],device=device)
            q1,q2=model.forward_sequence(o,p,t,a);q1s.append(q1.cpu().numpy().ravel());q2s.append(q2.cpu().numpy().ravel())
            targets.append(e.returns);labels.append(np.full(e.length,e.success));progress.append(np.arange(e.length)/float(horizon))
        output[policy]=metrics(np.concatenate(q1s),np.concatenate(q2s),np.concatenate(targets),np.concatenate(labels).astype(bool),np.concatenate(progress))
    output["balanced_aggregate"]={key:float(np.mean([output[p][key] for p in POLICIES])) for key in ("q1_mse","q2_mse","twin_mean_mse","mae")}
    model.train(was_training)
    return output
