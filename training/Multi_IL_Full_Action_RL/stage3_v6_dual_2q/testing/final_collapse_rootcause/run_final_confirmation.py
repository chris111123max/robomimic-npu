"""Final unscreened disjoint-seed confirmation; no optimization."""
import os,sys,copy,time,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from helpers import HERE,read,dump,load_tools

def run():
 import torch
 bt,core=load_tools();out=HERE/'round6';reg=read(out/'preregistration.json');os.chdir(out);started=time.time();device,ready,ref,critic,scale,offset=core.setup();core.SEEDS=tuple(reg['seeds'])
 source=read(core.RUN/'shared/stage2_source_manifest.json')['multi_q']['checkpoint'];mc,_=core.strict_stage2_load(source,device);mc.eval().requires_grad_(False);critic.eval().requires_grad_(False)
 branches=[('READY',[ref],1)];actors={'READY':ref}
 for name,path in [('READY_Q_1250',reg['ready_actor_1250']),('MC_Q_1250',reg['mc_actor_1250'])]:
  actor=copy.deepcopy(ref);state=torch.load(path,map_location='cpu',weights_only=False);actor.load_state_dict(state['actor']);actor.eval().requires_grad_(False);actors[name]=actor;branches.append((name,[actor],1))
 hashes={n:core.module_hash(a) for n,a in actors.items()};ch=core.module_hash(critic);mh=core.module_hash(mc)
 saved=torch.load(HERE.parent/'closed_loop_acceptance_test/fixed_batch_bank.pth',map_location='cpu',weights_only=False);qmetrics={}
 with torch.no_grad():
  for name,actor in actors.items():
   qr,_=core.qprobe(actor,ref,critic,saved['contexts'],device,scale,offset);qm,_=core.qprobe(actor,ref,mc,saved['contexts'],device,scale,offset);qmetrics[name]={'ready_critic':qr,'MC_critic':qm}
 context_values={}
 from stage3_v5_history_critic import encode_replay_contexts
 with torch.no_grad():
  for prefix,label in [('OLD','OLD_REF_FIXED'),('BAD','BAD_REF_FIXED')]:
   b=read(HERE/'round4'/(''+label+'_result.json'));p=b['probe'];o=torch.as_tensor(p['observations'],device=device,dtype=torch.float32);a=torch.as_tensor(p['past_actions_production_raw'],device=device,dtype=torch.float32);steps=torch.as_tensor(p['episode_steps'],device=device,dtype=torch.long);act=torch.as_tensor(p['actions']['FIXED']['executed'],device=device,dtype=torch.float32)
   context_values[prefix]={}
   for name,c in [('READY',critic),('MC',mc)]:
    ctx=encode_replay_contexts(c,o,a,steps,700);q1,q2=c.q_from_context(tuple(v[:,-1] for v in ctx),act);context_values[prefix][name]={'q1':q1.cpu().numpy().reshape(-1).tolist(),'q2':q2.cpu().numpy().reshape(-1).tolist()}
 behavior,contract=core.evaluate_four(branches,scale,offset,out)
 assert contract['test_valid'] and all(contract[k]==4 for k in ['parallel_envs_initialized','parallel_envs_used','parallel_envs_closed'])
 assert core.module_hash(critic)==ch and core.module_hash(mc)==mh and all(core.module_hash(actors[n])==h for n,h in hashes.items())
 dump(out/'result.json',{'seeds':reg['seeds'],'unscreened_and_disjoint':True,'behavior':behavior,'qmetrics':qmetrics,'source_context_values':context_values,'actor_hashes':hashes,'env_contract':contract,'model_hashes_unchanged':True,'cost':{'existing_data_only':False,'simulator_calls':1,'episodes':12,'env_steps':sum(e['length'] for b in behavior.values() for e in b['episodes']),'actor_updates':0,'critic_updates':0,'wall_seconds':time.time()-started},'formal_training_stopped':True})
 print('FINAL_DISJOINT_CONFIRMATION_DONE',flush=True)
if __name__=='__main__':run()
