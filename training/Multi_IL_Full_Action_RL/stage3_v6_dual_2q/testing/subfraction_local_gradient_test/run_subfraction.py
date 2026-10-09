"""Testing-only fixed-proposal line search. Zero optimizer steps."""
import os,sys,json,time,hashlib,copy,subprocess,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'competence_backtracking_test'
ACCEPTANCE=(20003,20004,20010,20011)
FRACTIONS=(.03125,.015625,.0078125,.00390625)
KEYS=('q1_gain','q2_gain','qmean_gain')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def prep():
 evidence={}
 for n in ('FINAL_REPORT.md','final_summary.json','round2_analysis.json','round1_result.json','proposal_logs.jsonl','actor_updates.jsonl','fixed_inputs_manifest.json'):
  p=OLD/n;t=p.read_text()
  obj=[json.loads(l) for l in t.splitlines()] if p.suffix=='.jsonl' else json.loads(t) if p.suffix=='.json' else t
  evidence[n]={'path':str(p),'sha256':sha(p),'fully_read':True,'rows':len(obj) if isinstance(obj,list) else None}
 rows=[json.loads(l) for l in (OLD/'proposal_logs.jsonl').read_text().splitlines()]
 rej=[r for r in rows if r['decision']=='FULL_REJECT'];assert len(rej)==34
 lo=rej[0]['block'];hi=rej[-1]['block'];targets=[lo+i*(hi-lo)/3 for i in range(4)]
 chosen=[min(rej,key=lambda r:(abs(r['block']-t),r['block'])) for t in targets]
 assert [r['block'] for r in chosen]==[6,17,29,40]
 severity=[r['backtracking_attempts'][-1]['evaluation']['behavior']['success_count'] for r in chosen]
 assert 1 in severity and 2 in severity
 manifest=read(OLD/'fixed_inputs_manifest.json');inputs={manifest['bank_file']:sha(manifest['bank_file'])};selected=[]
 for r in chosen:
  a=r['backtracking_attempts'][-1];assert a['fraction']==.0625
  op=r['accepted_metrics']['checkpoint'];cp=r['full_candidate']['metrics']['checkpoint']
  inputs[op]=sha(op);inputs[cp]=sha(cp)
  selected.append({'block':r['block'],'old_checkpoint':op,'candidate_checkpoint':cp,'old_file_sha256':inputs[op],'candidate_file_sha256':inputs[cp],'historical_success_at_0p0625':a['evaluation']['behavior']['success_count'],'historical_full_block_q':r['full_candidate']['block_q_increment'],'historical_old_q':r['accepted_q'],'historical_full_q':r['full_candidate']['metrics']['q']['all'],'historical_old_drift':r['accepted_metrics']['drift']})
 root=HERE.parents[4]
 tracked=subprocess.check_output(['git','ls-files','training/Multi_IL_Full_Action_RL'],cwd=root,text=True).splitlines()
 production={str(root/f):sha(root/f) for f in tracked if '/testing/' not in f and (root/f).is_file() and Path(f).suffix in ('.py','.json','.yaml','.yml','.sh')}
 run=Path('/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/stage3v6_readiness_v2_multi_mean_random_formal_20260929_130945')
 formal={str(f):sha(f) for f in run.rglob('*.pth') if 'actor_collapse_diagnosis' not in f.parts}
 reg={'registered_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'branch':'mean2q/multi_q ONLY','device':'npu:0','selected':selected,'selection_rule':'closest actual rejected block to four equal-spaced positions over min/max rejected block; tie earliest; severity covered before new outcomes','block_targets':targets,'fractions':list(FRACTIONS),'optional_fraction':.001953125,'optional_rule':'all four mandatory fractions failed; final Q1 > noise floor AND real success <3/4','noise_floor_rule':'max(1e-7,10*max absolute Q1/Q2/Qmean gain difference between repeated identical theta_old offline probes); Q1 <=floor means Q_SIGNAL_VANISHED, no simulator','historical_q_identity_tolerance':1e-7,'safe_rule':'success>=3/4 AND sim_errors==0 AND block delta Q1>0 relative theta_old','material_retention_threshold':.05,'seeds':list(ACCEPTANCE),'parallel_envs':4,'order':'serial proposals; descending fractions; stop first safe','interpolation':'named parameters old+f*(candidate-old); nonparameter buffers old','no_optimizer_constructed':True,'round2':'read-only','classification_rule':'safe majority+material majority: small trust region; safe majority+negligible majority: negligible progress; no-safe majority+positive twin final gains: local misalignment; 2safe2unsafe: heterogeneous; otherwise inconclusive','trend_limit':'blocks11-40 share accepted_block10; chronology not increasing accepted Actor drift; two old states only','evidence':evidence,'input_hashes':inputs,'manifest':manifest,'production_hashes':production,'formal_checkpoint_hashes':formal}
 dump(HERE/'preregistration.json',reg)
 fusion=root/'fusion_result.json'
 dump(HERE/'fusion_before.json',{'path':str(fusion),'exists':fusion.exists(),'sha256':sha(fusion) if fusion.exists() else None,'git_status':subprocess.check_output(['git','status','--short','--','fusion_result.json'],cwd=root,text=True).strip()})
 print('PREREGISTERED',[(r['block'],r['historical_success_at_0p0625']) for r in selected],len(production),len(formal),flush=True)

def run():
 import torch,numpy as np
 reg=read(HERE/'preregistration.json')
 assert all(sha(p)==h for p,h in reg['input_hashes'].items())
 spec=importlib.util.spec_from_file_location('backtracking_reuse',OLD/'run_backtracking.py')
 bt=importlib.util.module_from_spec(spec);spec.loader.exec_module(bt)
 bt.HERE=HERE;bt.acc.qg.HERE=HERE;core=bt.core
 device,ready,ref,critic,scale,offset=core.setup();assert str(device)=='npu:0'
 critic_hash=core.module_hash(critic)
 saved=torch.load(reg['manifest']['bank_file'],map_location='cpu',weights_only=False)
 assert bt.acc.qg.arrays_hash(saved['bank'])==reg['manifest']['bank_hash']
 assert bt.acc.qg.arrays_hash([saved['probe']])==reg['manifest']['drift_probe_hash']
 assert bt.acc.qg.arrays_hash([saved['contexts']])==reg['manifest']['q_probe_hash']
 contexts=saved['contexts'];probeobs=torch.as_tensor(saved['probe']['observations'],device=device,dtype=torch.float32)
 base=core.actor_outputs(ref,probeobs,device)
 def metric(actor,name):return bt.metric(actor,ref,critic,contexts,device,scale,offset,probeobs,base,name,0)
 def evaluate(actor,name):
  out=HERE/'evaluations'/name;out.mkdir(parents=True,exist_ok=False)
  oldrng=bt.acc.rng();oldcwd=Path.cwd();oldseeds=core.SEEDS;t=time.time()
  try:
   core.SEEDS=ACCEPTANCE;os.chdir(out)
   behavior,contract=core.evaluate_four([(name,[actor],1)],scale,offset,out)
  finally:
   core.SEEDS=oldseeds;os.chdir(oldcwd);bt.acc.restore_rng(oldrng)
  assert contract['test_valid'] and all(contract[k]==4 for k in ('parallel_envs_initialized','parallel_envs_used','parallel_envs_closed'))
  r={'label':name,'device':'npu:0','seeds':list(ACCEPTANCE),'behavior':behavior[name],'contract':contract,'env_steps':sum(e['length'] for e in behavior[name]['episodes']),'wall_time_seconds':time.time()-t,'actor_updates':0,'critic_updates':0,'command':'python -u '+str(Path(__file__).resolve())+' run'}
  dump(out/'invocation.json',r);print('SIM_DONE',name,r['behavior']['success_count'],r['env_steps'],flush=True)
  return r
 results=[];started=time.time()
 with (HERE/'fraction_logs.jsonl').open('x') as log:
  for selected in reg['selected']:
   block=selected['block'];name='B'+str(block).zfill(2)
   old=torch.load(selected['old_checkpoint'],map_location='cpu',weights_only=False)['actor']
   candidate=torch.load(selected['candidate_checkpoint'],map_location='cpu',weights_only=False)['actor']
   actor=copy.deepcopy(ref);actor.load_state_dict(old,strict=True)
   oh=core.module_hash(actor);mold=metric(actor,name+'_old');mrepeat=metric(actor,name+'_old_repeat')
   repeatnoise=max(abs(mold['q']['all'][k]-mrepeat['q']['all'][k]) for k in KEYS);floor=max(1e-7,10*repeatnoise)
   actor.load_state_dict(candidate,strict=True);ch=core.module_hash(actor);mfull=metric(actor,name+'_full_identity')
   for k in KEYS:
    assert abs(mold['q']['all'][k]-selected['historical_old_q'][k])<=reg['historical_q_identity_tolerance'],('OLD_IDENTITY_INVALID',block,k)
    assert abs(mfull['q']['all'][k]-selected['historical_full_q'][k])<=reg['historical_q_identity_tolerance'],('FULL_IDENTITY_INVALID',block,k)
   full={k:mfull['q']['all'][k]-mold['q']['all'][k] for k in KEYS};assert all(full[k]>0 for k in KEYS)
   result={'block':block,'old_state_hash':oh,'candidate_state_hash':ch,'source':selected,'identity_verified':True,'replayed_optimizer_steps':0,'noise_floor':floor,'repeat_probe_max_difference':repeatnoise,'old_metrics':mold,'full_block_delta_q':full,'attempts':[],'max_safe_fraction':None,'safe_retention':None}
   dump(HERE/(name+'_identity.json'),{k:v for k,v in result.items() if k!='attempts'})
   fractions=list(FRACTIONS)
   for i in range(5):
    if i==4:
     last=result['attempts'][-1]
     if not(last['block_delta_q']['q1_gain']>floor and last.get('evaluation') and last['evaluation']['behavior']['success_count']<3):break
     fractions.append(.001953125)
    f=fractions[i];label=name+'_F'+str(f).replace('.','p')
    actor.load_state_dict(old,strict=True)
    with torch.no_grad():
     for n,p in actor.named_parameters():p.copy_(old[n].to(device)+f*(candidate[n].to(device)-old[n].to(device)))
    m=metric(actor,label);inc={k:m['q']['all'][k]-mold['q']['all'][k] for k in KEYS}
    at={'block':block,'fraction':f,'metrics':m,'block_delta_q':inc,'noise_floor':floor,'evaluation':None,'safe':False}
    if inc['q1_gain']<=0:at['status']='NOT_Q_IMPROVING'
    elif inc['q1_gain']<=floor:at['status']='Q_SIGNAL_VANISHED'
    else:
     at['evaluation']=evaluate(actor,label);e=at['evaluation']['behavior'];at['safe']=e['success_count']>=3 and e['sim_errors']==0
     at['status']='SAFE_Q_IMPROVING_FRACTION' if at['safe'] else 'COMPETENCE_FAILED'
    result['attempts'].append(at);log.write(json.dumps(at,allow_nan=False)+'\n');log.flush()
    assert core.module_hash(critic)==critic_hash
    if at['safe']:
     result['max_safe_fraction']=f;result['safe_retention']={k:inc[k]/full[k] for k in KEYS}
     result['classification']='MATERIAL_SAFE_REGION' if result['safe_retention']['q1_gain']>=.05 else 'MATHEMATICALLY_SAFE_BUT_PRACTICALLY_NEGLIGIBLE'
     break
   if result['max_safe_fraction'] is None:
    result['classification']='NO_SAFE_FRACTION_DOWN_TO_1_OVER_512' if len(result['attempts'])==5 else 'NO_SAFE_FRACTION_DOWN_TO_1_OVER_256'
    if all(a['status'] in ('NOT_Q_IMPROVING','Q_SIGNAL_VANISHED') for a in result['attempts']):result['classification']='Q_SIGNAL_VANISHED'
   dump(HERE/(name+'_result.json'),result);results.append(result)
   print('PROPOSAL_DONE',block,result['max_safe_fraction'],result['safe_retention'],flush=True)
 dump(HERE/'round1_result.json',{'results':results,'runtime_seconds':time.time()-started,'critic_hash_before':critic_hash,'critic_hash_after':core.module_hash(critic),'actor_optimizer_steps':0,'critic_optimizer_steps':0,'target_updates':0,'device':str(device),'source_env_steps':ready['env_steps']})
 print('ROUND1_DONE',flush=True)
if __name__=='__main__':
 if sys.argv[1]=='prepare':prep()
 elif sys.argv[1]=='run':run()
 else:raise ValueError(sys.argv[1])
