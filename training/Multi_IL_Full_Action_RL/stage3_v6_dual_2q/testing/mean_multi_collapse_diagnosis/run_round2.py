from core import *
from stage3_v5_history_critic import encode_replay_contexts
import os
ALPHAS=(0.,.25,.5,.75,1.,1.25)
def main():
 first=json.loads((HERE/'round1/result.json').read_text())
 assert first['verdict']=='PASS' and first['behavior']['FROZEN_1250']['success_count']==0
 out=HERE/'round2';out.mkdir(exist_ok=True)
 device,ready,reference,critic,scale,offset=setup()
 snapshot=torch.load(HERE/'round1/actor_1250.pth',map_location='cpu',weights_only=False)
 assert snapshot['testing_only'] and snapshot['actor_virtual_updates']==1250
 actor=copy.deepcopy(reference);actor.load_state_dict(snapshot['actor'],strict=True);actor.eval();reference.eval()
 contexts=dict(np.load(TEST/'actor_collapse_diagnosis/results/fixed_contexts.npz'))
 raw=dict(np.load(HERE/'round1/paired_q_1250.npz'))
 line={};rawline={}
 for ck in ('critic_ready.pth','step_0200000.pth'):
  state=torch.load(RUN/'mean2q/multi_q/checkpoints'/ck,map_location='cpu',weights_only=False)
  critic.load_state_dict(state['q1_q2'],strict=True)
  scores={a:{'q1':[],'q2':[]} for a in ALPHAS}
  with torch.no_grad():
   for lo in range(0,256,32):
    hi=lo+32
    obs=torch.as_tensor(contexts['a3_observations'][lo:hi],device=device,dtype=torch.float32)
    acts=torch.as_tensor(contexts['a3_actions'][lo:hi],device=device,dtype=torch.float32)
    steps=torch.as_tensor(contexts['a3_episode_steps'][lo:hi],device=device,dtype=torch.long)
    enc=encode_replay_contexts(critic,obs,acts,steps,700);ctx=(enc[0][:,-1],enc[1][:,-1])
    ai=torch.as_tensor(raw['init_action_env'][lo:hi],device=device);ac=torch.as_tensor(raw['current_action_env'][lo:hi],device=device)
    for a in ALPHAS:
     q1,q2=critic.q_from_context(ctx,ai+a*(ac-ai))
     scores[a]['q1'].extend(q1.cpu().numpy().reshape(-1).tolist());scores[a]['q2'].extend(q2.cpu().numpy().reshape(-1).tolist())
  rows=[]
  for a,v in scores.items():
   x=np.asarray(v['q1']);y=np.asarray(v['q2']);b1=np.asarray(scores[0.]['q1']);b2=np.asarray(scores[0.]['q2'])
   d1=x-b1;d2=y-b2;sg=contexts['a3_success'].astype(bool)
   rows.append({'alpha':a,'q1':float(x.mean()),'q2':float(y.mean()),'qmean':float(((x+y)/2).mean()),'q1_gain':float(d1.mean()),'q2_gain':float(d2.mean()),'qmean_gain':float(((d1+d2)/2).mean()),'both_positive_fraction':float(np.mean((d1>0)&(d2>0))),'success_context_both_positive_fraction':float(np.mean((d1[sg]>0)&(d2[sg]>0))),'twin_median':float(np.median(np.abs(x-y))),'twin_p95':float(np.percentile(np.abs(x-y),95)),'distance_from_reference':float(a*np.linalg.norm(raw['current_action_env']-raw['init_action_env'],axis=-1).mean())})
  line[ck]=rows;rawline[ck]=scores
 dump(out/'action_line.json',{'alphas':ALPHAS,'contexts':256,'line_definition':'a=argmax-mean reference action + alpha*(argmax-mean failed action - reference); fixed replay states and executed-action history','limitation':'point-action surface, not production mixture expectation; behavioral test below blends paired sampled actions on its own visited states','critics':line,'no_optimizer_steps':True})
 dump(out/'raw_action_line.json',{ck:{str(a):v for a,v in scores.items()} for ck,scores in rawline.items()})
 # Current probabilities/RNN untouched. Only executed action deviation is restricted.
 # Common-RNG samples of both actors; both recurrent streams see the same visited observation.
 os.chdir(out)
 results,contract=evaluate_four([(f'ALPHA_{str(a).replace(".","p")}',[reference,actor],a) for a in (.25,.5,.75)],scale,offset,out)
 rescue=max(v['success_count'] for v in results.values())
 verdict='PASS' if rescue>=3 else 'INCONCLUSIVE'
 dump(out/'result.json',{'verdict':verdict,'line':line,'behavior':results,'contract':contract,'intervention':'execute reference_sample + alpha*(failed_sample-reference_sample) at each own visited state; no retraining/anchor/critic/objective change','reference_actor_updates':0,'failed_actor_updates':1250,'additional_actor_updates':0,'critic_updates':0,'best_rescued_success_count':rescue,'formal_training_stopped':True})
if __name__=='__main__':main()
