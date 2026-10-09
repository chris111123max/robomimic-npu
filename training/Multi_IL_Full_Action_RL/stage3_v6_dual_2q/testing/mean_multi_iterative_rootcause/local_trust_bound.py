"""Last round: local per-update policy-output trust bound; not a BC anchor."""
from run_blocks import *
from core import schedule,qprobe,actor_outputs,policy_drift,production_actor_loss,build_batch_bank,module_hash,evaluate_four
from stage3_v5_actor import flat_to_obs
import copy
def outputs(actor,observations):
 actor.eval()
 with torch.no_grad():
  d=actor.forward_train(flat_to_obs(observations),rnn_init_state=None,return_state=False)
  m=d.component_distribution.base_dist.loc.detach().clone()
  p=d.mixture_distribution.probs.detach().clone()
 return m,p,(m*p[...,None]).sum(-2)
def distance(a,b):
 m=torch.linalg.vector_norm(a[0]-b[0],dim=-1).max()
 p=torch.linalg.vector_norm(a[1]-b[1],dim=-1).max()
 w=torch.linalg.vector_norm(a[2]-b[2],dim=-1).max()
 return float(torch.maximum(torch.maximum(m,p),w))
def main():
 prev_report=(HERE/'round4/ROUND_REPORT.md').read_text()
 prev=read(HERE/'round4/result.json')
 assert prev['behavior']['FAILED_STATE_BATCH']['success_count']==0
 out=HERE/'round5';out.mkdir(exist_ok=True)
 device,ready,ref,bad,critic,scale,offset,_=prior.prepare()
 cfg=ready['config'];ch=module_hash(critic)
 bank,probe,replay=build_batch_bank(cfg,ready,64)
 old=read(TEST/'mean_multi_collapse_diagnosis/round1/result.json')
 epsilon=old['metrics']['625']['drift']['weighted_action_l2_mean']/625.
 dump(out/'design.json',{'previous_round_conclusion':'state-batch correction does notrescue, despite81.6% Qgain; no further state sampling tuning','why_next':'distinguish pure state-batch explanation from harmful cumulative output movement','single_change':'scale actual optimizer proposal only when max per-token normalized component/weighted action or probability-vector change from PREVIOUS actor exceeds epsilon','epsilon':epsilon,'calibration':'old known625-update3/4 actor meanweighteddrift/625; heuristic measured bound, not proven universal safe radius','reference_for_constraint':'immediately preceding actor, NOT fixed BC actor; no anchor loss','updates':1250,'objective_schedule_batches':'unchanged production loss/schedule/original64bank','baseline':'Round4 freshly evaluated original1250 actor0/4; reused valid four-env control; no redundant baseline rollout','one_strength_only':True,'not_online':'frozen Critic/replay Actor-only mechanism test, not online continuation'})
 actor=copy.deepcopy(ref);optim=torch.optim.Adam(actor.parameters(),lr=0.)
 optim.load_state_dict(copy.deepcopy(ready['actor_optimizer']))
 envs,lrs=schedule(cfg,1250)
 contexts=dict(np.load(TEST/'actor_collapse_diagnosis/results/fixed_contexts.npz'))
 probeobs=torch.as_tensor(probe['observations'],device=device,dtype=torch.float32)
 baseout=actor_outputs(ref,probeobs,device);metrics={};scales=[];clipped=0
 parameters=list(actor.parameters())
 with (out/'updates.jsonl').open('x') as f:
  for u in range(1,1251):
   b=bank[(u-1)%64]
   obs=torch.as_tensor(b['observations'],device=device,dtype=torch.float32)
   oldoutputs=outputs(actor,obs)
   before=[p.detach().clone() for p in parameters]
   actor.train()
   for pg in optim.param_groups:pg['lr']=lrs[u-1]
   optim.zero_grad(set_to_none=True)
   loss,ex=production_actor_loss(actor,critic,b,scale,offset,cfg,device);loss.backward()
   assert bool(torch.isfinite(loss)) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters)
   torch.nn.utils.clip_grad_norm_(parameters,cfg['actor_max_grad_norm']);optim.step()
   proposal=[p.detach().clone() for p in parameters]
   proposal_distance=distance(outputs(actor,obs),oldoutputs)
   factor=min(1.,epsilon/max(proposal_distance,1e-12))
   accepted=proposal_distance
   if factor<1:
    clipped+=1
    for trial in range(12):
     with torch.no_grad():
      for p,x,y in zip(parameters,before,proposal):p.copy_(x+factor*(y-x))
     accepted=distance(outputs(actor,obs),oldoutputs)
     if accepted<=epsilon*1.002:break
     factor*=.5
    assert accepted<=epsilon*1.002,(u,accepted,epsilon)
   scales.append(factor)
   f.write(json.dumps({'update':u,'loss':float(loss),'lr':lrs[u-1],'proposal_output_change_max':proposal_distance,'accepted_output_change_max':accepted,'parameter_step_fraction':factor},allow_nan=False)+'\n');f.flush()
   if u%100==0:print(json.dumps({'event':'update','u':u,'loss':float(loss),'factor':factor,'max_change':accepted}),flush=True)
   if u in (625,1250):
    q,raw=qprobe(actor,ref,critic,contexts,device,scale,offset)
    rp={n:p.detach() for n,p in ref.named_parameters()}
    drift=float(torch.sqrt(sum((p-rp[n]).square().sum() for n,p in actor.named_parameters())))
    metrics[str(u)]={'q':q,'policy_drift':policy_drift(actor_outputs(actor,probeobs,device),baseout),'parameter_drift_unique_l2':drift}
 assert module_hash(critic)==ch
 offline={'metrics':metrics,'actor_updates':1250,'critic_updates':0,'clipped_updates':clipped,'mean_parameter_step_fraction':float(np.mean(scales)),'epsilon':epsilon,'critic_hash_unchanged':True}
 dump(out/'offline.json',offline)
 os.chdir(out)
 behavior,contract=evaluate_four([('LOCAL_TRUST_REGION',[actor],1)],scale,offset,out)
 baseline=read(TEST/'mean_multi_collapse_diagnosis/round1/result.json')['metrics']['1250']
 retention=metrics['1250']['q']['all']['qmean_gain']/baseline['q']['all']['qmean_gain']
 dump(out/'result.json',{'offline':offline,'behavior':behavior,'env_contract':contract,'baseline_reused':prev['behavior']['BASELINE_REUSED'],'baseline_contract_reused':prev['env_contract'],'qgain_retention':retention,'source_start':'critic_ready140Kactor0','baseline_updates':1250,'new_actor_updates':1250,'critic_updates':0,'formal_training_stopped':True,'decision':'STOP','caveat':'one finite-update trust bound can delay optimization; no claim permanent prevention or formal3M fix'})
 print('ROUND5_DONE',flush=True)
if __name__=='__main__':main()
