"""Round3 follows Round2: actual execution distribution value probe, no simulator."""
from run_blocks import *
from core import qprobe
from scipy.stats import spearmanr
def main():
 previous_report=(HERE/'round2/ROUND_REPORT.md').read_text()
 previous=read(HERE/'round2/result.json')
 assert previous['decision']=='CONTINUE_TO_ROUND_3'
 out=HERE/'round3';out.mkdir(exist_ok=True)
 device,ready,ref,bad,critic,scale,offset,_=prior.prepare()
 sources={'BC':PREV/'round4/FORK_0p0_trace.jsonl','BLOCK8':HERE/'round1/BLOCK_8_trace.jsonl','BLOCK32':HERE/'round1/BLOCK_32_trace.jsonl','FAILED_FULL':TEST/'mean_multi_collapse_diagnosis/round1/FROZEN_1250_trajectories.jsonl'}
 contexts={'a3_observations':[],'a3_actions':[],'a3_episode_steps':[],'a3_success':[]}
 tags=[];mc=[];times=[];seeds=[]
 for name,p in sources.items():
  records=[json.loads(l) for l in p.read_text().splitlines()]
  for seed in SEEDS:
   rows=[r for r in records if r['seed']==seed]
   iswin=bool(rows[-1]['success'])
   for t in range(9,len(rows),10):
    h=rows[t-9:t+1]
    contexts['a3_observations'].append([r.get('obs',r.get('observation_flat')) for r in h])
    contexts['a3_actions'].append([r['action'] for r in h])
    contexts['a3_episode_steps'].append([r.get('t',r.get('timestep')) for r in h])
    contexts['a3_success'].append(iswin)
    mc.append(rows[t].get('mc',rows[t].get('finite_mc_return')))
    tags.append(name);times.append(t);seeds.append(seed)
 data={k:np.asarray(v) for k,v in contexts.items()}
 q,raw=qprobe(bad,ref,critic,data,device,scale,offset)
 tags=np.asarray(tags);mc=np.asarray(mc);times=np.asarray(times)
 raw.update(actual_mc=mc,tags=tags,times=times,seeds=np.asarray(seeds),**data)
 np.savez_compressed(out/'actual_execution_values.npz',**raw)
 summaries={}
 actual_qmean=(raw['q1_replay_action']+raw['q2_replay_action'])/2
 current=(raw['q1_expected_current']+raw['q2_expected_current'])/2
 reference=(raw['q1_expected_init']+raw['q2_expected_init'])/2
 for name in sources:
  masks={'all':tags==name,'early_before199':(tags==name)&(times<199),'after_block':(tags==name)&(times>=239)}
  summaries[name]={}
  for part,mask in masks.items():
   if not mask.any():continue
   summaries[name][part]={'count':int(mask.sum()),'actual_mc_mean':float(mc[mask].mean()),'q_executed_mean':float(actual_qmean[mask].mean()),'current_actor_expected_qmean':float(current[mask].mean()),'reference_actor_expected_qmean':float(reference[mask].mean()),'expected_qgain':float((current-reference)[mask].mean()),'both_expected_q_increase_fraction':float(np.mean((raw['q1_expected_current'][mask]>raw['q1_expected_init'][mask])&(raw['q2_expected_current'][mask]>raw['q2_expected_init'][mask]))),'q_vs_mc_spearman':float(spearmanr(actual_qmean[mask],mc[mask]).statistic) if np.std(mc[mask])>0 else None}
 src=Path('training/Multi_IL_Full_Action_RL/stage3_v5_rgmm_td3/stage3_v5_agent.py')
 lines=src.read_text().splitlines()
 source='\n'.join(lines[303:380])
 result={'previous_round_conclusion':'Q1/Q2 favor all injected block actions, but8 failsone while32 succeedsall; local scores not guarantees for future policy path','why_next':'compare replay/reference Q improvement with values on actually visited failed states; inspect whether source constrains policy behavior change','cost':'frozen forward offline probe','summaries':summaries,'production_actor_update_source':source,'source_observation':'inherited V5 actor loss = -expected Q1 on replay contexts; clips parameter gradient norm only, then unconditional optimizer step; no policy-output trust region or true-return acceptance test in this update','limitations':['frozen ready Q evaluates actual observed histories; post-update-policy MC is not necessarily its learned continuation value','same-time trajectories are not same states after divergence','only four known-good seeds; no calibrated global state-support classifier'],'decision':'CONTINUE_TO_ROUND_4'}
 dump(out/'result.json',result)
 print(json.dumps(summaries,indent=2),flush=True)
if __name__=='__main__':main()
