"""Sequential single-NPU staged experiment; never launch a formal run."""
import json,os,sys,subprocess,shutil,hashlib,time,math
from pathlib import Path
H=Path(__file__).resolve().parent;O=H/'results_20261003';R=json.loads((H/'preregistration.json').read_text())

def status(d):
 p=O/'experiment_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)

def main():
 import torch
 O.mkdir(exist_ok=True)
 # Existing successful stages are reused; no duplicate trainer is launched.
 p=torch.load(R['source_checkpoint'],map_location='cpu',weights_only=False)
 init=torch.load(Path(R['target_run'])/'shared/actor_init.pth',map_location='cpu',weights_only=False)['actor_state_dict']
 assert p['env_steps']==140000 and p['actor_updates']==0 and not p['actor_optimizer']['state']
 assert set(p['actor'])==set(init) and all(torch.equal(p['actor'][k],init[k]) for k in init)
 source=Path(R['source_checkpoint']);assert Path(p['online_sequence_replay']).exists()
 integrity={'env_steps':p['env_steps'],'actor_updates':p['actor_updates'],'critic_updates':p['updates'],'actor_equals_init':True,'Adam_state_empty':True,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'replay':p['online_sequence_replay'],'preregistration_sha256':hashlib.sha256((H/'preregistration.json').read_bytes()).hexdigest()}
 if (O/'source_integrity.json').exists():assert json.loads((O/'source_integrity.json').read_text())==integrity
 else:(O/'source_integrity.json').write_text(json.dumps(integrity,indent=2)+'\n')
 del p,init
 states=[];overall={'status':'RUNNING','testing_only':True,'completed_stages':states,'formal_training_resumed':False};status(overall)
 src_shared=Path(R['target_run'])/'shared'
 for branch in ('BASELINE','ADAPTIVE'):
  shared=O/branch/'shared';shared.mkdir(parents=True,exist_ok=True)
  for name in ('config_resolved.json','quad_fairness.json','stage2_source_manifest.json'):
   destination=shared/name
   if destination.exists():assert destination.read_bytes()==(src_shared/name).read_bytes()
   else:shutil.copyfile(src_shared/name,destination)
  destination=shared/'actor_init.pth'
  if destination.exists():assert destination.resolve()==(src_shared/'actor_init.pth').resolve()
  else:destination.symlink_to(src_shared/'actor_init.pth')
 for target in R['staged_schedule']:
  for branch in ('BASELINE','ADAPTIVE'):
   for f,h in R['production_sha256'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
   pair=O/branch;group=pair/'mean2q/multi_q'
   resume=Path(R['source_checkpoint']) if target==160000 else group/'checkpoints'/f'step_{target-20000:07d}.pth'
   assert resume.exists();manifest=json.loads((pair/'shared/stage2_source_manifest.json').read_text())
   env=dict(os.environ,PRESERVATION_BRANCH=branch,PRESERVATION_GROUP_OUT=str(group),PYTHONUNBUFFERED='1')
   cmd=[sys.executable,'-u',str(H/'train_testing.py'),'--group','multi_q','--target-mode','mean2q','--device','npu:0','--quad-run-dir',str(pair),'--critic-init-checkpoint',manifest['multi_q']['checkpoint'],'--num-envs','16','--total-env-steps',str(target),'--resume',str(resume),'--startup-ready-file',str(pair/f'ready_{target}.json')]
   logfile=pair/f'stage_{target}.log'
   overall.update(active_branch=branch,active_target=target,active_log=str(logfile));status(overall)
   print('STAGE_START '+json.dumps({'branch':branch,'target':target,'resume':str(resume),'log':str(logfile)}),flush=True)
   started=time.time()
   if logfile.exists():
    assert 'TESTING_TRAINER_EXITED_ALL_VECTOR_ENVS_CLOSED' in logfile.read_text(), 'Existing incomplete/failed stage must be diagnosed, not overwritten'
    rc=0
    print('REUSE_COMPLETED_STAGE '+str(logfile),flush=True)
   else:
    with logfile.open('w') as log:
     process=subprocess.Popen(cmd,cwd=pair,env=env,stdout=log,stderr=subprocess.STDOUT)
     overall['active_pid']=process.pid;status(overall)
     rc=process.wait()
   text=logfile.read_text()
   if rc or 'TESTING_TRAINER_EXITED_ALL_VECTOR_ENVS_CLOSED' not in text:
    overall.update(status='FAILED',failure={'branch':branch,'target':target,'returncode':rc,'log':str(logfile)},active_pid=None);status(overall);raise RuntimeError(overall['failure'])
   ck=group/'checkpoints'/f'step_{target:07d}.pth';saved=torch.load(ck,map_location='cpu',weights_only=False)
   assert saved['env_steps']==target and saved['actor_updates']>0 and saved['testing_only'] and saved['actor_gate_open'] and saved['training_state']['critic_ready_step']==140000 and 140000 <= saved['gate_open_step'] <= 140016
   assert saved['preservation_preregistration_sha256']==integrity['preregistration_sha256']
   for key in ('actor','target_actor','q1_q2','target_q1_q2'):
    assert all(bool(torch.isfinite(v).all()) for v in saved[key].values() if torch.is_tensor(v))
   diagnostics=json.loads((group/'preservation_diagnostics'/f'step_{target:07d}.json').read_text())
   assert diagnostics['closed_loop']['sim_error_count']==0 and diagnostics['reference_hash_unchanged']
   for line in (group/'actor_update_diagnostics.jsonl').read_text().splitlines():
    row=json.loads(line)
    for value in row.values():
     if isinstance(value,float):assert math.isfinite(value)
   record={'branch':branch,'target':target,'critic_updates':saved['updates'],'actor_updates':saved['actor_updates'],'success':diagnostics['closed_loop']['success_count'],'execution_drift':diagnostics['fixed_140K_critic_probe']['execution_drift'],'duration_seconds':time.time()-started,'software_and_numeric_safety_pass':True,'checkpoint':str(ck)}
   states.append(record);overall.update(active_pid=None);status(overall);print('STAGE_PASS '+json.dumps(record),flush=True);del saved
  print('BOTH_BRANCHES_STAGE_SAFETY_PASS '+str(target),flush=True)
 overall.update(status='COMPLETE',active_branch=None,active_pid=None);status(overall)
 print('ALL_TESTING_TRAINERS_EXITED; FORMAL TRAINING REMAINS STOPPED',flush=True)
if __name__=='__main__':main()
