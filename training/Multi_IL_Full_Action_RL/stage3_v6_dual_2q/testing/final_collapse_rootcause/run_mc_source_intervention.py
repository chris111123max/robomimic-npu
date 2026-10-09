"""Causal value-source substitution; production-strength Actor updates, testing only."""
import os,sys,copy,time,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from helpers import HERE,read,dump,load_tools
import numpy as np

def run():
 import torch
 bt,core=load_tools();out=HERE/'round5';os.chdir(out);started=time.time();device,ready,reference,readycritic,scale,offset=core.setup();cfg=ready['config']
 source=read(core.RUN/'shared/stage2_source_manifest.json')['multi_q']['checkpoint'];mccritic,payload=core.strict_stage2_load(source,device);mccritic.eval().requires_grad_(False);readycritic.eval().requires_grad_(False)
 saved=torch.load(HERE.parent/'closed_loop_acceptance_test/fixed_batch_bank.pth',map_location='cpu',weights_only=False);design=read(HERE.parent/'closed_loop_acceptance_test/design.json')
 assert bt.acc.qg.arrays_hash(saved['bank'])==design['bank_hash'];assert bt.acc.qg.arrays_hash([saved['contexts']])==design['q_probe_hash']
 bank=saved['bank'];contexts=saved['contexts'];actor=copy.deepcopy(reference);actor.requires_grad_(True);reference.eval().requires_grad_(False)
 optim=torch.optim.Adam(actor.parameters(),lr=0.,weight_decay=0.);optim.load_state_dict(copy.deepcopy(ready['actor_optimizer']));assert len(optim.state)==0
 envs,lrs=core.schedule(cfg,1250);ch=core.module_hash(mccritic);rch=core.module_hash(readycritic);rh=core.module_hash(reference);metrics={}
 with (out/'updates.jsonl').open('x') as f:
  for u in range(1,1251):
   actor.train()
   for pg in optim.param_groups:pg['lr']=lrs[u-1]
   optim.zero_grad(set_to_none=True);loss,_=core.production_actor_loss(actor,mccritic,bank[(u-1)%64],scale.reshape(1,1,1,14),offset.reshape(1,1,1,14),cfg,device);loss.backward()
   assert bool(torch.isfinite(loss)) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in actor.parameters())
   gn=torch.nn.utils.clip_grad_norm_(actor.parameters(),cfg['actor_max_grad_norm']);optim.step();f.write(json.dumps({'actor_update':u,'implied_env_step':int(envs[u-1]),'lr':lrs[u-1],'loss':float(loss),'grad_norm':float(gn)},allow_nan=False)+'\n');f.flush()
   if u%100==0:print('MC_SOURCE_ACTOR_UPDATE',u,float(loss),flush=True)
   if u in [1,250,625,1250]:
    actor.eval()
    with torch.no_grad():
     qmc,_=core.qprobe(actor,reference,mccritic,contexts,device,scale,offset);qr,_=core.qprobe(actor,reference,readycritic,contexts,device,scale,offset)
     drift=core.policy_drift(core.actor_outputs(actor,torch.as_tensor(saved['probe']['observations'],device=device),device),core.actor_outputs(reference,torch.as_tensor(saved['probe']['observations'],device=device),device))
    metrics[str(u)]={'own_MC_critic_q':qmc,'ready_critic_q':qr,'drift':drift}
 actor.eval();state={'testing_only':True,'actor':actor.state_dict(),'source_env_steps':140000,'actor_virtual_updates':1250,'critic_source':source,'not_formal_training':True}
 torch.save(state,out/'mc_source_actor_1250.pth');offline_seconds=time.time()-started
 assert core.module_hash(mccritic)==ch and core.module_hash(readycritic)==rch and core.module_hash(reference)==rh
 behavior,contract=core.evaluate_four([('MC_SOURCE_1250',[actor],1)],scale,offset,out)
 assert contract['test_valid'] and all(contract[k]==4 for k in ['parallel_envs_initialized','parallel_envs_used','parallel_envs_closed'])
 dump(out/'result.json',{'intervention':'replace frozen ready Critic with original Stage2.2 MC Critic; all Actor mechanics unchanged','critic_source':source,'source_checkpoint_step':payload.get('step'),'metrics':metrics,'behavior':behavior,'env_contract':contract,'model_hashes_unchanged':True,'batch_bank_hash':design['bank_hash'],'q_probe_hash':design['q_probe_hash'],'lrs':lrs,'implied_env_step':int(envs[-1]),'cost':{'offline_probe':True,'simulator_calls':1,'episodes':4,'env_steps':sum(e['length'] for e in behavior['MC_SOURCE_1250']['episodes']),'actor_updates':1250,'critic_updates':0,'offline_seconds':offline_seconds,'wall_seconds':time.time()-started},'formal_training_stopped':True})
 print('MC_SOURCE_INTERVENTION_DONE',flush=True)
if __name__=='__main__':run()
