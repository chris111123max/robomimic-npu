"""Read-only production Bellman and actual rollout-action semantics audit."""
import os,sys,copy,time,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from helpers import HERE,OLD,read,dump,load_tools
import numpy as np

def setup_actors():
 import torch
 bt,core=load_tools();device,ready,bc,critic,scale,offset=core.setup()
 spec=next(s for s in read(OLD/'round1/preregistration.json')['states'] if s['block']==17)
 state=torch.load(spec['old_checkpoint'],map_location='cpu',weights_only=False)['actor'];full=torch.load(spec['candidate_checkpoint'],map_location='cpu',weights_only=False)['actor']
 old=copy.deepcopy(bc);old.load_state_dict(state);bad=copy.deepcopy(old)
 with torch.no_grad():
  for n,p in bad.named_parameters():p.copy_(state[n].to(device)+spec['fraction']*(full[n].to(device)-state[n].to(device)))
 actors={'REF':bc,'OLD':old,'BAD':bad}
 for a in actors.values():a.eval().requires_grad_(False)
 return core,device,ready,actors,critic,scale,offset

def run():
 import torch
 from stage3_v5_history_critic import encode_replay_contexts
 from stage3_v5_agent import target_final_distribution_vectorized,_last_reset_starts_from_numpy
 out=HERE/(sys.argv[1] if len(sys.argv)>1 else 'round3');os.chdir(out);start=time.time()
 core,device,ready,actors,critic,scale,offset=setup_actors()
 original={n:core.module_hash(a) for n,a in actors.items()};ch=core.module_hash(critic)
 target=copy.deepcopy(critic);target.load_state_dict(ready['target_q1_q2']);target.eval().requires_grad_(False)
 assert all(torch.equal(v,ready['target_actor'][k]) for k,v in ready['actor'].items())
 def tensor(x,dtype=torch.float32):return torch.as_tensor(x,device=device,dtype=dtype)
 def qeval(c,contexts,dist,noise=False):
  base=dist.component_distribution.base_dist;means=base.loc*scale+offset;probs=dist.mixture_distribution.probs
  raw1,raw2=c.q_from_context(contexts,means);a=means.clamp(-1,1);q1,q2=c.q_from_context(contexts,a)
  result={'component_means':means.cpu().numpy().tolist(),'mode_probabilities':probs.cpu().numpy().tolist(),'expected_q1_raw':(probs*raw1.squeeze(-1)).sum(-1).cpu().numpy().tolist(),'expected_q2_raw':(probs*raw2.squeeze(-1)).sum(-1).cpu().numpy().tolist(),'expected_q1_feasible':(probs*q1.squeeze(-1)).sum(-1).cpu().numpy().tolist(),'expected_q2_feasible':(probs*q2.squeeze(-1)).sum(-1).cpu().numpy().tolist(),'std':base.scale.cpu().numpy().tolist()}
  if noise:
   rng=np.random.default_rng(62731);epsilon=rng.standard_normal((*base.loc.shape[:-1],32,14)).astype(np.float32);epsilon=np.concatenate((epsilon,-epsilon),axis=-2)
   draws=((base.loc.unsqueeze(-2)+base.scale.unsqueeze(-2)*tensor(epsilon))*scale+offset).clamp(-1,1)
   sq1,sq2=c.q_from_context(contexts,draws.reshape(means.shape[0],-1,14))
   for k,q in [('q1',sq1),('q2',sq2)]:result['expected_'+k+'_actual_eval_std_antithetic64']=(probs*q.squeeze(-1).reshape(means.shape[0],5,64).mean(-1)).sum(-1).cpu().numpy().tolist()
  return result
 regions={}
 with torch.no_grad():
  for region in ['round1','round2']:
   source=read(HERE/region/'result.json');probe=source['branches']['REF_REF']['probe'];o=tensor(probe['observations']);a=tensor(probe['past_actions_production_raw']);steps=tensor(probe['episode_steps'],torch.long);cs=encode_replay_contexts(critic,o,a,steps,700);contexts=tuple(c[:,-1] for c in cs)
   current={}
   for name,actor in actors.items():
    dist,_=target_final_distribution_vectorized(actor,o,episode_steps=steps,horizon=10)
    current[name]=qeval(critic,contexts,dist,True)
   decomposition={}
   for name,b in source['branches'].items():
    if b['continuation']!='REF':continue
    rows=[json.loads(l) for l in Path(b['trace']).read_text().splitlines()];fork=int(probe['episode_steps'][0][-1]);selected=[next(r for r in rows if r['seed']==e['seed'] and r['timestep']==fork) for e in b['episodes']]
    act=a.clone();act[:,-1]=tensor([x['action_executed_feasible'] for x in selected]);no=torch.cat((o[:,1:],tensor([x['next_observation_flat'] for x in selected]).unsqueeze(1)),1)
    contextn=encode_replay_contexts(target,o,act,steps,700,next_observations=no);cn=tuple(c[:,-1] for c in contextn)
    dist,_=target_final_distribution_vectorized(actors['REF'],no,horizon=10,starts=_last_reset_starts_from_numpy(np.asarray(probe['episode_steps']),10))
    q=qeval(target,cn,dist);expected=.5*(np.asarray(q['expected_q1_raw'])+np.asarray(q['expected_q2_raw']));reward=np.asarray([x['reward'] for x in selected]);done=np.asarray([x['terminated'] or x['truncated'] for x in selected]);td=reward+.99*(1-done)*expected
    decomposition[name]={'immediate_reward':reward.tolist(),'terminated':done.tolist(),'target_expected_next_qmean':expected.tolist(),'production_td_target':td.tolist(),'finite_mc_reference_continuation':[e['mc_from_fork'] for e in b['episodes']],'current_q1_feasible':probe['actions'][b['action']]['q1'],'current_q2_feasible':probe['actions'][b['action']]['q2'],'target_actor':'readiness Actor, byte-identical to actor at140K','next_action_source':'categorical probability expectation of five physical component means; not Q of weighted average action','success':[e['success'] for e in b['episodes']]}
   regions[region]={'mixture_semantics':current,'bellman_decomposition':decomposition}

 assert core.module_hash(critic)==ch and all(core.module_hash(actors[n])==h for n,h in original.items())
 result={'regions':regions,'target_actor_equals_readiness_actor':True,'ready_env_steps':ready['env_steps'],'ready_actor_updates':ready['actor_updates'],'ready_critic_updates':ready['updates'],'production_contract':{'actor_objective':'sum p_k Q1(h,mu_k), no actor gradient through std','rollout':'categorical mode plus eval Gaussian std then executable clipping','target':'r+.99*(1-terminal)*mean(sum p Q1_target(mu),sum p Q2_target(mu))','current_history':'10 sliding observations; first prior action zero; other prior actions shifted','successor_history':'next observations, first prior zero; subsequent prior actions a_(s+1)..a_t','progress':'episode steps+1 /700 for successor','actor_RNN':'zero reset every10; suffix after last reset','done_mask':'terminal_for_td = terminated OR truncated; finite episode closure preserved'},'limitations':['Single-RNG finite MC is a paired realized outcome, not a many-rollout conditional expectation.','Bellman residual on a novel context is not evidence that this context was supervised in replay.','Mean-vs-sampled-Q agreement alone cannot prove closed-loop correctness.'],'cost':{'existing_data_only':False,'offline_probe':True,'simulator_calls':0,'episodes':0,'env_steps':0,'actor_updates':0,'critic_updates':0,'wall_seconds':time.time()-start},'model_hashes_unchanged':True}
 dump(out/'result.json',result);print('VALUE_AUDIT_DONE',flush=True)
if __name__=='__main__':run()
