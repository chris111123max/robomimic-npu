"""Q-gain matched path test; exact prior local intervention, testing only."""
import sys,copy,json,time,hashlib,os
from pathlib import Path
HERE=Path(__file__).resolve().parent
TEST=HERE.parent
OLD=TEST/'mean_multi_iterative_rootcause'
sys.path.insert(0,str(OLD))
import local_trust_bound as local
from core import *
HERE=Path(__file__).resolve().parent
TEST=HERE.parent
from scipy.stats import spearmanr,pearsonr
MAX_UPDATES=40000
TOLERANCE=.05
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def hashes():return {p:sha(p) for p in read(OLD/'safety_before.json')}
def arrays_hash(items):
 h=hashlib.sha256()
 for x in items:
  for k in sorted(x):
   a=np.asarray(x[k]);h.update(k.encode());h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes())
 return h.hexdigest()
def prepare():
 device,ready,ref,critic,scale,offset=setup()
 bank,probe,replay=build_batch_bank(ready['config'],ready,64)
 contexts=dict(np.load(TEST/'actor_collapse_diagnosis/results/fixed_contexts.npz'))
 probeobs=torch.as_tensor(probe['observations'],device=device,dtype=torch.float32)
 base=actor_outputs(ref,probeobs,device)
 return device,ready,ref,critic,scale,offset,bank,probe,contexts,probeobs,base,replay
def measure(actor,ref,critic,contexts,device,scale,offset,probeobs,base,name,updates):
 with torch.no_grad():
  q,raw=qprobe(actor,ref,critic,contexts,device,scale,offset)
  drift=policy_drift(actor_outputs(actor,probeobs,device),base)
  rp=dict(ref.named_parameters())
  drift['parameter_l2']=float(torch.sqrt(sum((p-rp[n]).square().sum() for n,p in actor.named_parameters())))
 delta={}
 for twin in ('q1','q2'):
  v=np.asarray(raw[twin+'_expected_current']-raw[twin+'_expected_init'],dtype=float)
  delta[twin]={'mean':float(v.mean()),'median':float(np.median(v)),'p10':float(np.percentile(v,10)),'p90':float(np.percentile(v,90)),'positive_fraction':float(np.mean(v>0))}
 np.savez_compressed(HERE/(name+'_probe.npz'),**raw)
 r={'name':name,'actor_updates':updates,'q':q,'delta_distribution':delta,'drift':drift}
 dump(HERE/(name+'_metrics.json'),r)
 return r
def save_actor(actor,name,updates,optim=None):
 p=HERE/(name+'.pth');assert not p.exists()
 x={'testing_only':True,'actor':actor.state_dict(),'source_env_steps':140000,'actor_virtual_updates':updates}
 if optim is not None:x['actor_optimizer']=optim.state_dict()
 torch.save(x,p);return str(p)
def stage_a():
 assert not (HERE/'design.json').exists()
 evidence={}
 for p in [OLD/'FINAL_REPORT.md',OLD/'final_summary.json']+sorted(OLD.glob('round[1-5]/*.json'))+sorted(OLD.glob('round[1-5]/*.md')):
  evidence[str(p)]={'sha256':sha(p),'content':read(p) if p.suffix=='.json' else p.read_text()}
 if not (HERE/'prior_evidence_read.json').exists():dump(HERE/'prior_evidence_read.json',evidence)
 dump(HERE/'safety_before.json',hashes())
 device,ready,ref,critic,scale,offset,bank,probe,contexts,probeobs,base,replay=prepare()
 cfg=ready['config'];old=read(TEST/'mean_multi_collapse_diagnosis/round1/result.json')
 epsilon=old['metrics']['625']['drift']['weighted_action_l2_mean']/625.
 design={'scope':'mean2q/multi_q ONLY','device':'npu:0','relative_tolerance':TOLERANCE,'max_actor_updates':MAX_UPDATES,'cap_reason':'prior 1250 gain ratio implies about22.3x updates under linear progress; 40000 about32x baseline allows headroom, no tuning','epsilon':epsilon,'constraint':'EXACT imported previous-policy output bound: max component mean L2, weighted action L2, probabilities L2; scale parameter proposal, keep Adam moments, up to12 backtracking trials, acceptance epsilon*1.002','objective':'unchanged production expected Q1','schedule':'EXACT prior core.schedule interpolation; beyond final real metadata interpolation endpoint saturates at315904, LR saturates accordingly, not actual env steps','q_probe':'exact prior fixed_contexts.npz 256 contexts, aligned history10, component probability-weighted Q1/Q2 evaluated with frozen readiness Critic; normalized means rescaled to env actions','drift_probe':'EXACT previous build_batch_bank 65th heldout batch','replay':str(replay),'bank_hash':arrays_hash(bank),'drift_probe_hash':arrays_hash([probe]),'q_probe_hash':arrays_hash([contexts]),'reference_behavior_source':str(TEST/'mean_multi_collapse_diagnosis/round1/result.json'),'seeds':list(SEEDS),'parallel_envs':4,'checkpoint_timing':'synchronous post Actor optimizer acceptance; no learner catch-up or env steps'}
 dump(HERE/'design.json',design)
 refs={}
 # START ready actor; existing exact evaluation is reused.
 r=measure(ref,ref,critic,contexts,device,scale,offset,probeobs,base,'unrestricted_start',0)
 r['checkpoint']=save_actor(ref,'unrestricted_start',0)
 r['behavior']=old['behavior']['READY'];refs['START']=r
 # LOW: absent250 weight reconstructed, original production update only.
 actor=copy.deepcopy(ref);optim=torch.optim.Adam(actor.parameters(),lr=0.)
 optim.load_state_dict(copy.deepcopy(ready['actor_optimizer']));assert not optim.state
 envs,lrs=schedule(cfg,250)
 for u in range(1,251):
  actor.train()
  for pg in optim.param_groups:pg['lr']=lrs[u-1]
  optim.zero_grad(set_to_none=True)
  loss,_=production_actor_loss(actor,critic,bank[(u-1)%64],scale,offset,cfg,device)
  loss.backward();assert bool(torch.isfinite(loss))
  assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in actor.parameters())
  torch.nn.utils.clip_grad_norm_(actor.parameters(),cfg['actor_max_grad_norm']);optim.step()
 r=measure(actor,ref,critic,contexts,device,scale,offset,probeobs,base,'unrestricted_low',250)
 r['checkpoint']=save_actor(actor,'unrestricted_low',250)
 assert abs(r['q']['all']['q1_gain']-old['metrics']['250']['q']['all']['q1_gain'])<1e-7
 refs['LOW']=r
 for label,u in [('MID',625),('COLLAPSE',1250)]:
  src=TEST/f'mean_multi_collapse_diagnosis/round1/actor_{u}.pth'
  actor.load_state_dict(torch.load(src,map_location='cpu',weights_only=False)['actor'])
  r=measure(actor,ref,critic,contexts,device,scale,offset,probeobs,base,'unrestricted_'+label.lower(),u)
  r['checkpoint']=save_actor(actor,'unrestricted_'+label.lower(),u)
  r['source_checkpoint']=str(src);r['source_sha256']=sha(src)
  r['behavior']=old['behavior']['FROZEN_'+str(u)]
  assert abs(r['q']['all']['q1_gain']-old['metrics'][str(u)]['q']['all']['q1_gain'])<1e-7
  refs[label]=r
 dump(HERE/'stage_a.json',{'references':refs,'targets':{k:refs[k]['q']['all']['q1_gain'] for k in ['LOW','MID','COLLAPSE']},'new_actor_updates':250,'new_env_steps':0,'existing_evaluation_contract':old['contract'],'exact_reference_reproduction_verified':True})
 print('STAGE_A_DONE',json.dumps({k:r['q']['all'] for k,r in refs.items()}),flush=True)
def stage_b():
 design=read(HERE/'design.json');a=read(HERE/'stage_a.json')
 device,ready,ref,critic,scale,offset,bank,probe,contexts,probeobs,base,replay=prepare()
 assert arrays_hash(bank)==design['bank_hash'] and arrays_hash([contexts])==design['q_probe_hash'] and arrays_hash([probe])==design['drift_probe_hash']
 actor=copy.deepcopy(ref);optim=torch.optim.Adam(actor.parameters(),lr=0.)
 optim.load_state_dict(copy.deepcopy(ready['actor_optimizer']));assert not optim.state
 cfg=ready['config'];envs,lrs=schedule(cfg,MAX_UPDATES);epsilon=design['epsilon'];ch=module_hash(critic)
 params=list(actor.parameters());matches={};labels=['LOW','MID','COLLAPSE'];started=time.time();clipped=0;scales=[]
 nextlabel=0;lastq=None
 with (HERE/'updates.jsonl').open('x') as f:
  for u in range(1,MAX_UPDATES+1):
   b=bank[(u-1)%64];obs=torch.as_tensor(b['observations'],device=device,dtype=torch.float32)
   oldout=local.outputs(actor,obs);before=[p.detach().clone() for p in params]
   actor.train()
   for pg in optim.param_groups:pg['lr']=lrs[u-1]
   optim.zero_grad(set_to_none=True)
   loss,_=production_actor_loss(actor,critic,b,scale,offset,cfg,device);loss.backward()
   assert bool(torch.isfinite(loss)) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params)
   torch.nn.utils.clip_grad_norm_(params,cfg['actor_max_grad_norm']);optim.step()
   proposal=[p.detach().clone() for p in params]
   dist=local.distance(local.outputs(actor,obs),oldout);factor=min(1.,epsilon/max(dist,1e-12));accepted=dist
   if factor<1:
    clipped+=1
    for trial in range(12):
     with torch.no_grad():
      for p,x,y in zip(params,before,proposal):p.copy_(x+factor*(y-x))
     accepted=local.distance(local.outputs(actor,obs),oldout)
     if accepted<=epsilon*1.002:break
     factor*=.5
    assert accepted<=epsilon*1.002,(u,accepted)
   scales.append(factor)
   f.write(json.dumps({'update':u,'loss':float(loss),'lr':lrs[u-1],'implied_env_step':int(envs[u-1]),'proposal_output_change_max':dist,'accepted_output_change_max':accepted,'parameter_step_fraction':factor},allow_nan=False)+'\n')
   if u%100==0:f.flush()
   if u%50==0 or u==MAX_UPDATES:
    with torch.no_grad():q,_=qprobe(actor,ref,critic,contexts,device,scale,offset)
    lastq=q['all'];gain=lastq['q1_gain'];target=a['targets'][labels[nextlabel]]
    progress={'update':u,'delta_q1':gain,'delta_q2':lastq['q2_gain'],'delta_qmean':lastq['qmean_gain'],'next_target':labels[nextlabel],'target_q1':target,'elapsed_seconds':time.time()-started}
    with (HERE/'probe_progress.jsonl').open('a') as pf:pf.write(json.dumps(progress,allow_nan=False)+'\n')
    if u%500==0:print(json.dumps(progress),flush=True)
    if gain>=target*(1-TOLERANCE):
     err=abs(gain-target)/target
     if err<=TOLERANCE:
      label=labels[nextlabel];r=measure(actor,ref,critic,contexts,device,scale,offset,probeobs,base,'constrained_'+label.lower(),u)
      r.update(checkpoint=save_actor(actor,'constrained_'+label.lower(),u,optim),target_q1=target,relative_matching_error=err,absolute_matching_error=abs(gain-target))
      matches[label]=r
      dump(HERE/('match_'+label.lower()+'.json'),r)
      print('Q_MATCH',json.dumps(r),flush=True)
      nextlabel+=1
      if nextlabel==len(labels):break
     else:
      raise RuntimeError(('probe interval overshot predeclared tolerance, do not silently retune',progress))
 assert module_hash(critic)==ch
 end={'matches':matches,'actor_updates':u,'critic_updates':0,'clipped_updates':clipped,'mean_parameter_step_fraction':float(np.mean(scales)),'critic_hash_unchanged':True,'elapsed_seconds':time.time()-started,'last_q':lastq,'reached_collapse':nextlabel==3}
 if nextlabel<3:
  end['unmatched_endpoint']=measure(actor,ref,critic,contexts,device,scale,offset,probeobs,base,'constrained_unmatched_endpoint',u)
  end['unmatched_endpoint']['checkpoint']=save_actor(actor,'constrained_unmatched_endpoint',u,optim)
 dump(HERE/'stage_b.json',end);print('STAGE_B_DONE',json.dumps(end),flush=True)
def stage_c():
 a=read(HERE/'stage_a.json');b=read(HERE/'stage_b.json')
 device,ready,ref,critic,scale,offset=setup()
 actors=[];actor=copy.deepcopy(ref)
 # Existing READY/MID/COLLAPSE four-env results are reused; LOW had no evaluation.
 entries=[('UNRESTRICTED_LOW',a['references']['LOW'])]
 entries += [('CONSTRAINED_'+k,v) for k,v in b['matches'].items()]
 if not b['reached_collapse']:entries.append(('CONSTRAINED_UNMATCHED_ENDPOINT',b['unmatched_endpoint']))
 for name,r in entries:
  x=copy.deepcopy(ref);x.load_state_dict(torch.load(r['checkpoint'],map_location='cpu',weights_only=False)['actor']);actors.append((name,[x],1))
 out=HERE/'evaluation';out.mkdir();os.chdir(out);started=time.time()
 behavior,contract=evaluate_four(actors,scale,offset,out)
 dump(HERE/'stage_c.json',{'behavior':behavior,'contract':contract,'runtime_seconds':time.time()-started,'total_env_steps':sum(e['length'] for r in behavior.values() for e in r['episodes']),'actor_updates_during_rollout':0,'critic_updates':0,'command':'python '+str(HERE/'run_test.py')+' stage_c'})
 print('STAGE_C_DONE',json.dumps(behavior),flush=True)
if __name__=='__main__':{'stage_a':stage_a,'stage_b':stage_b,'stage_c':stage_c}[sys.argv[1]]()
