"""Bounded testing-only offline diagnosis. Reuse completed rollouts; never spawn envs."""
import sys,os,json,hashlib,shutil
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
TEST=HERE.parent
RL=TEST.parents[1]
ROOT=RL.parents[1]
OLD=TEST/'critic_counterfactual_identifiability'
def dump(name,data):
 p=HERE/name
 if p.exists(): raise FileExistsError(p)
 p.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
sources={
 'stage2_2_history_aware_critic/sequence_dataset.py':[(1,61)],
 'stage2_2_history_aware_critic/sequence_sampler.py':[(30,85)],
 'stage2_2_history_aware_critic/train_stage2_2.py':[(55,112)],
 'stage2_2_history_aware_critic/history_critic.py':[(1,60)],
 'stage3_v5_rgmm_td3/stage3_v5_history_critic.py':[(27,84)],
 'stage3_v5_rgmm_td3/stage3_v5_agent.py':[(25,103),(304,374)],
 'stage3_v5_rgmm_td3/stage3_v5_actor.py':[(67,112),(278,335)],
 'stage3_v5_rgmm_td3/stage3_v5_replay.py':[(25,105),(165,190),(230,290)],
 'stage3_v6_dual_2q/stage3_v6_agent.py':[(75,160)],
 'stage3_v6_dual_2q/stage3_v6_readiness.py':[(55,150)]}
audit=[]
for rel,ranges in sources.items():
 p=RL/rel; lines=p.read_text().splitlines()
 audit.append({'path':str(p),'sha256':digest(p),'excerpts':[{ 'start':a,'end':b,'text':'\n'.join(f'{i+1}: {s}' for i,s in enumerate(lines) if a<=i+1<=b)} for a,b in ranges]})
dump('source_audit.json',audit)
reused=[('discovery','round2','BC_strict','round3','BC_strict'),('independent','round4','BC_heldout','round4','CURRENT_heldout')]
manifest=[]; pairs=[]; probe_contexts=[]
for group,ra,fa,rb,fb in reused:
 records=[]; traces=[]
 for r,f in [(ra,fa),(rb,fb)]:
  for suffix in ['_result.json','_trace.jsonl']:
   p=OLD/r/(f+suffix); dest=HERE/'raw_evidence'/r/p.name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(p,dest)
   manifest.append({'source':str(p),'copy':str(dest),'sha256':digest(p),'bytes':p.stat().st_size})
  records.append(read(OLD/r/(f+'_result.json'))['episodes'])
  ts={}
  for line in (OLD/r/(f+'_trace.jsonl')).read_text().splitlines():
   d=json.loads(line); ts.setdefault(d['seed'],[]).append(d)
  for x in ts.values(): x.sort(key=lambda e:e['timestep'])
  traces.append(ts)
 assert len(records[0])==len(records[1])==4
 for a,b in zip(*records):
  seed=a['seed']; fork=a['fork']; assert (seed,fork)==(b['seed'],b['fork'])
  for k in ['history_observations','history_actions','episode_steps','q','pair_check']:
   assert a[k]==b[k],(seed,k)
  candidate=a.get('executed_candidate',a.get('candidate')); assert candidate==b.get('executed_candidate',b.get('candidate'))
  assert max(map(abs,candidate))<=1.0000001
  ta,tb=traces[0][seed],traces[1][seed]
  assert [x['timestep'] for x in ta]==list(range(a['length']))
  assert [x['timestep'] for x in tb]==list(range(b['length']))
  for t in range(fork+1):
   for k in ['observation_flat','next_observation_flat','reward','action_executed' if 'action_executed' in ta[t] else 'action']:
    assert ta[t][k]==tb[t][k],(seed,t,k)
  mc=[]
  for e,tr in [(a,ta),(b,tb)]:
   g=sum(.99**(t-fork)*tr[t]['reward'] for t in range(fork,len(tr)))
   assert abs(g-e['mc_from_fork'])<1e-12
   assert abs(g-tr[fork]['finite_mc_return'])<1e-12
   mc.append(g)
  pairs.append({'group':group,'seed':seed,'fork':fork,'BC_success':a['success'],'CURRENT_success':b['success'],'BC_length':a['length'],'CURRENT_length':b['length'],'BC_mc':mc[0],'CURRENT_mc':mc[1],'delta_mc':mc[1]-mc[0],'q_same':a['q'],'strict_record_and_prefix_match':True,'pair_check':a['pair_check']})
  probe_contexts.append(a)
dump('raw_evidence_manifest.json',manifest)
dump('round1_reused_pairs.json',pairs)
print('ROUND1 COMPLETE eight strict pairs; no simulator rerun',flush=True)
# Round 2: one frozen-network pass, exactly the same eight histories.
os.chdir(HERE)
sys.path.insert(0,str(TEST/'mean_multi_collapse_diagnosis'))
from core import setup,load_exact_actor,strict_stage2_load,module_hash,RUN
import torch,numpy as np
from stage3_v5_actor import flat_to_obs
from stage3_v5_agent import target_final_distribution_vectorized,_last_reset_starts_from_numpy
from stage3_v5_history_critic import encode_replay_contexts
with torch.no_grad():
 device,ready,bc,critic,scale,offset=setup()
 assert str(device)=='npu:0'
 bc.eval().requires_grad_(False)
 current,_,_=load_exact_actor(RUN/'shared/bc_rnn_gmm_source.pth',device)
 current.load_state_dict(torch.load(TEST/'mean_multi_collapse_diagnosis/round1/actor_1250.pth',map_location='cpu',weights_only=False)['actor'],strict=True)
 current.eval().requires_grad_(False)
 source,_,_=load_exact_actor(RUN/'shared/bc_rnn_gmm_source.pth',device)
 source.eval().requires_grad_(False)
 eq_actor=all(torch.equal(ready['actor'][k],source.state_dict()[k].cpu()) for k in ready['actor'])
 eq_target=all(torch.equal(ready['target_actor'][k],ready['actor'][k]) for k in ready['actor'])
 assert eq_actor and eq_target
 manifest_source=read(RUN/'shared/stage2_source_manifest.json')
 mc,payload=strict_stage2_load(manifest_source['multi_q']['checkpoint'],device)
 mc.eval().requires_grad_(False)
 before={'BC':module_hash(bc),'CURRENT':module_hash(current),'ready_critic':module_hash(critic),'MC_critic':module_hash(mc)}
 probe=[]
 for e in probe_contexts:
  obs=torch.tensor([e['history_observations']],device=device,dtype=torch.float32)
  acts=torch.tensor([e['history_actions']],device=device,dtype=torch.float32)
  steps=torch.tensor([e['episode_steps']],device=device,dtype=torch.long)
  row={'seed':e['seed'],'fork':e['fork'],'distributions':{},'Q_expectations':{}}
  distributions={}
  for name,actor in [('BC',bc),('CURRENT',current)]:
   dist,_=target_final_distribution_vectorized(actor,obs,steps,horizon=10)
   start=max(i for i,s in enumerate(e['episode_steps']) if s%10==0)
   state=None
   for i in range(start,10):
    native,state=actor.forward_train_step(flat_to_obs(obs[:,i]),rnn_state=state)
   err=float((dist.component_distribution.base_dist.loc-native.component_distribution.base_dist.loc).abs().max().cpu())
   perr=float((dist.mixture_distribution.probs-native.mixture_distribution.probs).abs().max().cpu())
   assert max(err,perr)<5e-5,(name,e['seed'],err,perr)
   target_starts=_last_reset_starts_from_numpy(np.array([e['episode_steps']])-1,10)
   same,_=target_final_distribution_vectorized(actor,obs,horizon=10,starts=target_starts)
   assert torch.equal(same.component_distribution.base_dist.loc,dist.component_distribution.base_dist.loc)
   row['distributions'][name]={'native_vs_vectorized_mean_max_abs':err,'native_vs_vectorized_probability_max_abs':perr,'next_step_reset_metadata_consistent':True,'std_max':float(dist.component_distribution.base_dist.scale.max().cpu()),'probs':dist.mixture_distribution.probs.cpu().tolist(),'means_env':(dist.component_distribution.base_dist.loc*scale+offset).cpu().tolist()}
   distributions[name]=dist
  for name,c in [('ready',critic),('MC',mc)]:
   encoded=encode_replay_contexts(c,obs,acts,steps,700); contexts=tuple(x[:,-1] for x in encoded)
   values={}
   for actor_name,d in distributions.items():
    raw=d.component_distribution.base_dist.loc*scale+offset
    q1,q2=c.q_from_context(contexts,raw)
    clipped1,clipped2=c.q_from_context(contexts,raw.clamp(-1,1))
    p=d.mixture_distribution.probs
    values[actor_name]={'Q1':float((p*q1.squeeze(-1)).sum().cpu()),'Q2':float((p*q2.squeeze(-1)).sum().cpu()),'clip_Q1_delta':float((p*(clipped1-q1).squeeze(-1)).sum().cpu()),'clip_Q2_delta':float((p*(clipped2-q2).squeeze(-1)).sum().cpu()),'max_raw_bound_excess':float((raw.abs()-1).clamp_min(0).max().cpu())}
   values['CURRENT_minus_BC']={k:values['CURRENT'][k]-values['BC'][k] for k in ['Q1','Q2']}
   row['Q_expectations'][name]=values
  probe.append(row)
 after={'BC':module_hash(bc),'CURRENT':module_hash(current),'ready_critic':module_hash(critic),'MC_critic':module_hash(mc)}
 assert before==after
 dump('round2_frozen_semantics_probe.json',{'device':str(device),'contexts':8,'ready_actor_equals_BC_source':eq_actor,'ready_target_actor_equals_ready_actor':eq_target,'ready_env_steps':ready['env_steps'],'ready_actor_updates':ready['actor_updates'],'MC_gamma':payload['gamma'],'model_hashes_before':before,'model_hashes_after':after,'probe':probe,'limitation':'Current-step replay-action surrogate; not Q under CURRENT future policy, not new simulator samples. Shifted reset metadata algebra checked; successor reward/value not re-simulated.'})
print('ROUND2 COMPLETE frozen npu:0 pass; no optimizer/simulator',flush=True)
