from core import *
def main():
 out=HERE/'round1';out.mkdir(exist_ok=True)
 device,ready,reference,critic,scale,offset=setup()
 hash_before=module_hash(critic);cfg=ready['config']
 bank,probe,replaypath=build_batch_bank(cfg,ready,64)
 contexts=dict(np.load(TEST/'actor_collapse_diagnosis/results/fixed_contexts.npz'))
 actor=copy.deepcopy(reference);actor.train();reference.eval()
 optim=torch.optim.Adam(actor.parameters(),lr=0.,weight_decay=0.);optim.load_state_dict(ready['actor_optimizer']);assert len(optim.state)==0
 envs,lrs=schedule(cfg,1250)
 dump(out/'contract.json',{'critic_ready_env_steps':140000,'critic_updates':ready['updates'],'actor_updates':0,'critic_target_mode':'mean2q unchanged, frozen (no new TD updates)','objective':'unchanged production Q1 component-mean expectation','replay':str(replaypath),'batch_bank_size':64,'sequence_batch_size':64,'offline_fraction':.5,'device':'npu:0','total_actor_updates':1250,'implied_env_step_last':int(envs[-1]),'lr_mapping':'interpolation of formal logged Actor-update/env-step counts, not actual environment steps','lrs':lrs,'envs':envs.tolist()})
 probe_obs=torch.as_tensor(probe['observations'],device=device,dtype=torch.float32)
 base_outputs=actor_outputs(reference,probe_obs,device)
 snapshots={};metrics={}
 with (out/'updates.jsonl').open('x') as f:
  for u in range(1,1251):
   actor.train()
   for pg in optim.param_groups:pg['lr']=lrs[u-1]
   optim.zero_grad(set_to_none=True)
   loss,extra=production_actor_loss(actor,critic,bank[(u-1)%64],scale.reshape(1,1,1,14),offset.reshape(1,1,1,14),cfg,device)
   loss.backward()
   assert torch.isfinite(loss) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in actor.parameters())
   gn=torch.nn.utils.clip_grad_norm_(actor.parameters(),cfg['actor_max_grad_norm']);optim.step()
   if u%100==0:print(json.dumps({'event':'update','actor_update':u,'loss':float(loss),'lr':lrs[u-1]}),flush=True)
   f.write(json.dumps({'actor_update':u,'implied_env_step':int(envs[u-1]),'lr':lrs[u-1],'loss':float(loss),'grad_norm':float(gn)})+'\n');f.flush()
   if u in (1,250,625,1250):
    actor.eval();q,raw=qprobe(actor,reference,critic,contexts,device,scale,offset)
    metrics[str(u)]={'actor_updates':u,'implied_env_step':int(envs[u-1]),'q':q,'drift':policy_drift(actor_outputs(actor,probe_obs,device),base_outputs)}
    np.savez_compressed(out/f'paired_q_{u}.npz',**raw)
    if u in (625,1250):
     snapshots[str(u)]=copy.deepcopy(actor)
     torch.save({'testing_only':True,'source_env_steps':140000,'actor_virtual_updates':u,'actor':actor.state_dict()},out/f'actor_{u}.pth')
 assert module_hash(critic)==hash_before
 dump(out/'offline.json',{'metrics':metrics,'critic_hash_unchanged':True,'critic_optimizer_updates':0,'new_replay_transitions':0})
 results,contract=evaluate_four([('READY',[reference],1),('FROZEN_625',[snapshots['625']],1),('FROZEN_1250',[snapshots['1250']],1)],scale,offset,out)
 verdict='PASS' if any(results[k]['success_count']<results['READY']['success_count'] and metrics[str(u)]['q']['all']['q1_gain']>0 for k,u in [('FROZEN_625',625),('FROZEN_1250',1250)]) else 'INCONCLUSIVE'
 dump(out/'result.json',{'verdict':verdict,'metrics':metrics,'behavior':results,'contract':contract,'supports':'frozen-ready learned-Q gradient can harm behavior without online Critic/replay feedback' if verdict=='PASS' else 'no demonstrated behavior harm in this bounded frozen experiment','formal_training_stopped':True})
if __name__=='__main__':main()
