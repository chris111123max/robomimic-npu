"""Result-driven critic gradient diagnosis; all writes testing-only."""
import os,sys,json,time,copy,hashlib,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
TEST=HERE.parent
SUB=TEST/'subfraction_local_gradient_test'
BACK=TEST/'competence_backtracking_test'
EXCLUDED={20003,20004,20010,20011,20008,20002,20005,20007}
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def prep():
 evidence={}
 for name in ('competence_backtracking_test','subfraction_local_gradient_test','closed_loop_acceptance_test','q_gain_matched_path_test','mean_multi_iterative_rootcause','mean_multi_upstream_rootcause'):
  s=read(TEST/name/'final_summary.json');md=(TEST/name/'FINAL_REPORT.md').read_text()
  evidence[name]={'summary':s,'report':md,'summary_sha256':sha(TEST/name/'final_summary.json'),'report_sha256':sha(TEST/name/'FINAL_REPORT.md')}
 for n in ('proposal_logs.jsonl','actor_updates.jsonl'):
  rows=[json.loads(l) for l in (BACK/n).read_text().splitlines()]
  evidence[n]={'sha256':sha(BACK/n),'fully_read_rows':len(rows)}
 dump(HERE/'round0/existing_evidence.json',evidence)
 (HERE/'round0/ROUND_REPORT.md').write_text('# Round0\n\nPrevious: frozen learned-Q ascent can collapse competence; local bounds merely delay collapse at matched Q gain.\nCurrent question: cross-seed generalization before upstream attribution.\nExisting data sufficient? NO. Historical readiness successes found only on excluded acceptance/reporting seeds; formal20000/20001/20006/20009 failed.\nAnalysis: fully read six requested summaries/reports, prior raw40proposal and10000update logs, and baseline seed evaluations.\nEvidence: prior CAUSAL / MECHANISTIC / NEGATIVE.\nRuled out: slower learning as root-cause fix; online replay/later Critic necessary trigger; blanket immediate single-action harm; global NN-distance proxy as sufficient cause.\nTop hypothesis: wrong or non-policy-improving approximate Q optimization signal, upstream provenance unknown.\nDecision: CONTINUE_TO_ROUND_1.\n')
 oldreg=read(SUB/'preregistration.json');prior=read(SUB/'round1_result.json')
 controls={};states=[]
 for r in prior['results']:
  b=r['block'];f=.015625 if b==6 else .001953125
  a=next(a for a in r['attempts'] if a['fraction']==f)
  st={'block':b,'fraction':f,'old_checkpoint':r['source']['old_checkpoint'],'candidate_checkpoint':r['source']['candidate_checkpoint'],'historical_evaluation':a['evaluation'],'historical_block_q':a['block_delta_q'],'old_hash':r['old_state_hash'],'candidate_hash':r['candidate_state_hash']}
  states.append(st);controls[st['old_checkpoint']]=b
 baseline={'production_hashes':{p:sha(p) for p in oldreg['production_hashes']},'formal_checkpoint_hashes':{p:sha(p) for p in oldreg['formal_checkpoint_hashes']},'input_hashes':{p:sha(p) for p in oldreg['input_hashes']}}
 dump(HERE/'safety_before.json',baseline)
 reg={'scope':'mean2q/multi_q','device':'npu:0','rounds_max':4,'states':states,'controls':controls,'seed_selection_rule':'no disjoint historical4successful seeds found; readiness ONLY on disjoint ascending consecutive4seed groups starting20012; first new4/4 success and zero sim errors; lock before any candidate','seed_groups':[list(range(20012+4*i,20016+4*i)) for i in range(16)],'seed_group_cap':16,'cap_failure':'INCONCLUSIVE_STOP, no candidate without4/4 ready baseline','excluded_seeds':sorted(EXCLUDED),'candidate_seed_selection_forbidden':True,'old_controls_reason':'two distinct theta_old controls distinguish incremental direction damage from pre-existing policy generalization loss','fraction_search_forbidden':True,'q_identity_tolerance':1e-7,'history':10,'common_execution_rng_seed':20007,'simulator_contract':'exactly4parallel workers initialized/used/closed, candidates serial','round2_not_preregistered_yet':'decide only after Round1 analysis','no_optimizer_steps':True}
 dump(HERE/'round1/preregistration.json',reg);print('ROUND0_PREREG_DONE',flush=True)
def load_tools():
 spec=importlib.util.spec_from_file_location('previous_backtracking',BACK/'run_backtracking.py')
 bt=importlib.util.module_from_spec(spec);spec.loader.exec_module(bt)
 bt.HERE=HERE/'round1';bt.acc.qg.HERE=HERE/'round1'
 return bt,bt.core

def run():
 import torch,numpy as np
 out=HERE/'round1';reg=read(out/'preregistration.json');safety=read(HERE/'safety_before.json')
 assert all(sha(p)==h for p,h in safety['input_hashes'].items())
 bt,core=load_tools();device,ready,ref,critic,scale,offset=core.setup();ch=core.module_hash(critic)
 saved=torch.load(read(SUB/'preregistration.json')['manifest']['bank_file'],map_location='cpu',weights_only=False)
 design=read(SUB/'preregistration.json')['manifest']
 for key,hashkey in [('bank','bank_hash'),('probe','drift_probe_hash'),('contexts','q_probe_hash')]:
  assert bt.acc.qg.arrays_hash(saved[key] if key=='bank' else [saved[key]])==design[hashkey]
 probeobs=torch.as_tensor(saved['probe']['observations'],device=device,dtype=torch.float32);base=core.actor_outputs(ref,probeobs,device)
 def metric(actor,name):return bt.metric(actor,ref,critic,saved['contexts'],device,scale,offset,probeobs,base,name,0)
 def eval_branches(branches,seeds,label):
  dest=out/'evaluations'/label;dest.mkdir(parents=True,exist_ok=False);oldcwd=Path.cwd();oldseeds=core.SEEDS;oldrng=bt.acc.rng();t=time.time()
  try:
   core.SEEDS=tuple(seeds);os.chdir(dest);behavior,contract=core.evaluate_four(branches,scale,offset,dest)
  finally:core.SEEDS=oldseeds;os.chdir(oldcwd);bt.acc.restore_rng(oldrng)
  assert contract['test_valid'] and all(contract[k]==4 for k in ('parallel_envs_initialized','parallel_envs_used','parallel_envs_closed'))
  result={'device':'npu:0','seeds':seeds,'behavior':behavior,'contract':contract,'env_steps':sum(e['length'] for b in behavior.values() for e in b['episodes']),'wall_seconds':time.time()-t,'actor_updates':0,'critic_updates':0}
  dump(dest/'invocation.json',result);print('EVAL_DONE',label,{k:v['success_count'] for k,v in behavior.items()},flush=True)
  return result
 selection=[];chosen=None
 for i,seeds in enumerate(reg['seed_groups']):
  assert not set(seeds)&EXCLUDED
  e=eval_branches([('READINESS',[ref],1)],seeds,'SELECT_'+str(i).zfill(2));selection.append(e)
  if e['behavior']['READINESS']['success_count']==4 and e['behavior']['READINESS']['sim_errors']==0:
   chosen=seeds;dump(out/'independent_seeds.json',{'seeds':seeds,'selection':selection,'locked_before_candidate':True});break
 if chosen is None:
  dump(out/'selection_failed.json',{'selection':selection,'classification':'INCONCLUSIVE','reason':'No readiness4/4 group at preregistered16group cap, candidates not run'});return
 controls={};branches=[];identities=[]
 for path in reg['controls']:
  actor=copy.deepcopy(ref);state=torch.load(path,map_location='cpu',weights_only=False)['actor'];actor.load_state_dict(state,strict=True)
  label='OLD_'+Path(path).stem.upper();controls[path]=(actor,label,metric(actor,label));branches.append((label,[actor],1))
 for st in reg['states']:
  b=st['block'];old=torch.load(st['old_checkpoint'],map_location='cpu',weights_only=False)['actor'];full=torch.load(st['candidate_checkpoint'],map_location='cpu',weights_only=False)['actor'];actor=copy.deepcopy(ref)
  actor.load_state_dict(full);assert core.module_hash(actor)==st['candidate_hash']
  actor.load_state_dict(old);assert core.module_hash(actor)==st['old_hash']
  with torch.no_grad():
   for n,p in actor.named_parameters():p.copy_(old[n].to(device)+st['fraction']*(full[n].to(device)-old[n].to(device)))
  label='BLOCK_'+str(b).zfill(2);m=metric(actor,label);om=controls[st['old_checkpoint']][2]
  inc={k:m['q']['all'][k]-om['q']['all'][k] for k in ('q1_gain','q2_gain','qmean_gain')}
  assert all(abs(inc[k]-st['historical_block_q'][k])<=reg['q_identity_tolerance'] for k in inc)
  identities.append({'block':b,'fraction':st['fraction'],'actor_hash':core.module_hash(actor),'block_delta_q':inc,'same_Q_verified':True,'old_control_label':controls[st['old_checkpoint']][1]})
  branches.append((label,[actor],1))
 dump(out/'state_identity.json',identities)
 result=eval_branches(branches,chosen,'FIXED_STATES')
 assert core.module_hash(critic)==ch
 dump(out/'result.json',{'selection':selection,'seeds':chosen,'states':reg['states'],'identities':identities,'evaluation':result,'critic_hash_unchanged':True,'actor_updates':0,'critic_updates':0,'formal_training_stopped':True})
 print('ROUND1_DONE',flush=True)
if __name__=='__main__':
 if sys.argv[1]=='prepare':prep()
 elif sys.argv[1]=='run':run()
