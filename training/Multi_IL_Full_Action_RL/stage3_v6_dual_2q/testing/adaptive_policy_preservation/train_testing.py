"""Production trainer with testing-only agent, observational hooks and safe output roots."""
import os,sys,inspect,json,hashlib
from pathlib import Path
H=Path(__file__).resolve().parent

def main():
 sys.path.insert(0,str(H.parent.parent))
 import train_stage3_v6_vector as P
 import preservation_agent as A
 import milestone_diagnostics as D
 P.RecurrentGMMTD3=None # Main imports the module alias below, not this attribute.
 A.V6.RecurrentGMMTD3=A.PreservationAgent
 pair=Path(sys.argv[sys.argv.index('--quad-run-dir')+1]).resolve()
 assert pair.is_relative_to(H) and 'testing' in str(pair)
 args_step=int(sys.argv[sys.argv.index('--total-env-steps')+1]);assert args_step in (160000,180000,200000)
 reg=A.REG
 for f,h in reg['production_sha256'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
 original_payload=P.checkpoint_payload
 def checkpoint_payload(*args,**kwargs):
  payload=original_payload(*args,**kwargs);payload['testing_only']=True;payload['policy_preservation_branch']=os.environ['PRESERVATION_BRANCH'];payload['preservation_preregistration_sha256']=hashlib.sha256((H/'preregistration.json').read_bytes()).hexdigest();return payload
 P.checkpoint_payload=checkpoint_payload
 original_save=P.save_checkpoint
 def guarded_save(path,*args,**kwargs):
  assert Path(path).resolve().is_relative_to(H),path
  return original_save(path,*args,**kwargs)
 P.save_checkpoint=guarded_save
 source=inspect.getsource(P.main)
 old='config["run_type"] = "BENCHMARK" if args.benchmark_mode else ("SMOKE" if args.smoke else "FORMAL")'
 assert source.count(old)==1;source=source.replace(old,'config["run_type"] = "TESTING_ONLY_REAL_SHORT_RUN"',1)
 source=source.replace('evaluation_steps = set()','evaluation_steps = set(range(150000, total + 1, 10000))',1)
 start='        if args.benchmark_mode:\n            checkpoint_steps.add(int(args.benchmark_warmup_steps))'
 assert source.count(start)==1;source=source.replace(start,'        checkpoint_steps = set(range(150000, total + 1, 10000)) | {total}\n'+start,1)
 marker='        metric_rows = []\n        active_cursor = 0'
 assert source.count(marker)==1
 init='        if env_steps == 140000:\n            save_checkpoint(group_dir / "checkpoints" / "140K_start.pth", agent, config, args.group, env_steps, generations, episodes, successes, online, torch, handoff, credit.pending, offline)\n        testing_milestone(agent, config, env_steps, group_dir, torch)\n'
 source=source.replace(marker,init+marker,1)
 marker='                                    episodes, successes, online, torch, handoff, credit.pending, offline)\n\n            # Reset all completed workers'
 assert source.count(marker)==1
 source=source.replace(marker,'                                    episodes, successes, online, torch, handoff, credit.pending, offline)\n                    testing_milestone(agent, config, env_steps, group_dir, torch)\n\n            # Reset all completed workers',1)
 ns=dict(vars(P));ns['testing_milestone']=D.testing_milestone
 exec(compile(source,str(H/'derived_production_main'),'exec'),ns)
 if '--validate-only' in sys.argv:
  print('DERIVED_PRODUCTION_MAIN_AND_ACTOR_COMPILE_PASS');return
 ns['main']()
 print('TESTING_TRAINER_EXITED_ALL_VECTOR_ENVS_CLOSED',flush=True)
if __name__=='__main__':main()
