"""Round4: paired state-distribution intervention, production objective unchanged."""
from run_blocks import *
from core import schedule,qprobe,actor_outputs,policy_drift,production_actor_loss,build_batch_bank,module_hash,evaluate_four
import copy
def main():
 prev_report=(HERE/'round3/ROUND_REPORT.md').read_text()
 prev=read(HERE/'round3/result.json');assert prev['decision']=='CONTINUE_TO_ROUND_4'
 out=HERE/'round4';out.mkdir(exist_ok=True)
 device,ready,ref,bad,critic,scale,offset,_=prior.prepare()
 cfg=ready['config'];ch=module_hash(critic)
 bank,probe,replay=build_batch_bank(cfg,ready,64)
 # Exact offline half stays fixed; only replace online half with existing failed-state windows.
 rec=[json.loads(l) for l in (TEST/'mean_multi_collapse_diagnosis/round1/FROZEN_1250_trajectories.jsonl').read_text().splitlines()]
 windows=[]
 for seed in SEEDS:
  rs=[r for r in rec if r['seed']==seed]
  assert not rs[-1]['success']
  for s in range(0,len(rs)-9,10):windows.append(rs[s:s+10])
 rng=np.random.default_rng(20261004)
 order=rng.integers(0,len(windows),size=(64,32))
 for j,b in enumerate(bank):
  assert np.all(b['is_offline'][:32]==1) and np.all(b['is_offline'][32:]==0)
  w=[windows[ix] for ix in order[j]]
  b['observations'][32:]=np.asarray([[r['observation_flat'] for r in h] for h in w])
  b['actions'][32:]=np.asarray([[r['action'] for r in h] for h in w])
  b['episode_steps'][32:]=np.asarray([[r['timestep'] for r in h] for h in w])
 np.save(out/'failed_window_order.npy',order)
 dump(out/'design.json',{'previous_round_conclusion':'Critic recognizes actual failed states as lowQ, despite favoring failed actor on oldBC states','hypothesis':'old-state surrogate optimization mismatches induced execution state distribution','single_change':'replace online32/64 Actor rows with existing failed-policy visited aligned windows; unchangedoffline32/64','failed_window_count':len(windows),'start_checkpoint':'mean2q/multi_q/critic_ready.pth','updates':1250,'critic_updates':0,'baseline':'reuse exact previous64-bank baseline actor1250 weights; same source,schedule,objective and untouchedoffline rows, re-evaluate baseline first','not_online':'frozen networks/replay diagnostic; failed-state bank from older frozen test, not fresh on-policy sampler'})
 actor=copy.deepcopy(ref)
 optim=torch.optim.Adam(actor.parameters(),lr=0.);optim.load_state_dict(copy.deepcopy(ready['actor_optimizer']))
 envs,lrs=schedule(cfg,1250)
 contexts=dict(np.load(TEST/'actor_collapse_diagnosis/results/fixed_contexts.npz'))
 probeobs=torch.as_tensor(probe['observations'],device=device,dtype=torch.float32)
 baseout=actor_outputs(ref,probeobs,device)
 metrics={}
 with (out/'updates.jsonl').open('x') as f:
  for u in range(1,1251):
   actor.train()
   for pg in optim.param_groups:pg['lr']=lrs[u-1]
   optim.zero_grad(set_to_none=True)
   loss,ex=production_actor_loss(actor,critic,bank[(u-1)%64],scale,offset,cfg,device)
   loss.backward()
   assert bool(torch.isfinite(loss)) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in actor.parameters())
   torch.nn.utils.clip_grad_norm_(actor.parameters(),cfg['actor_max_grad_norm']);optim.step()
   f.write(json.dumps({'update':u,'lr':lrs[u-1],'loss':float(loss)},allow_nan=False)+'\n');f.flush()
   if u%100==0:print(json.dumps({'event':'update','u':u,'loss':float(loss)}),flush=True)
   if u in (625,1250):
    actor.eval();q,raw=qprobe(actor,ref,critic,contexts,device,scale,offset)
    ps={n:p.detach() for n,p in ref.named_parameters()}
    pd=float(torch.sqrt(sum((p-ps[n]).square().sum() for n,p in actor.named_parameters())))
    metrics[str(u)]={'q':q,'policy_drift':policy_drift(actor_outputs(actor,probeobs,device),baseout),'parameter_drift_unique_l2':pd}
 assert module_hash(critic)==ch
 dump(out/'offline.json',{'metrics':metrics,'actor_updates':1250,'critic_updates':0,'critic_hash_unchanged':True,'new_replay_transitions':0})
 # Save only testing actor for possible final ambiguity test; never formal checkpoints.
 torch.save({'testing_only':True,'actor':actor.state_dict(),'updates':1250},out/'state_aware_actor.pth')
 os.chdir(out)
 behavior,contract=evaluate_four([('BASELINE_REUSED',[bad],1),('FAILED_STATE_BATCH',[actor],1)],scale,offset,out)
 dump(out/'result.json',{'metrics':metrics,'behavior':behavior,'env_contract':contract,'new_actor_updates':1250,'critic_updates':0,'critic_hash_unchanged':True,'formal_training_stopped':True,'cost':'offline Actor mechanism intervention +4-env rollouts; NOT online continuation'})
 print('ROUND4_DONE',flush=True)
if __name__=='__main__':main()
