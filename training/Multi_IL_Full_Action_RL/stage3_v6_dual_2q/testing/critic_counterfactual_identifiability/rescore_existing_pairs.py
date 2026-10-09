"""Re-score exact existing paired actions under original MC and ready critics."""
from run_round1 import *
def main():
 out=HERE/'round2';out.mkdir(exist_ok=False);os.chdir(out);device,ready,actor,td,scale,offset=setup();manifest=json.loads((RUN/'shared/stage2_source_manifest.json').read_text());mc,_=strict_stage2_load(manifest['multi_q']['checkpoint'],device);mc.eval().requires_grad_(False)
 result={'new_simulator_calls':0,'reuse':{},'pairs':[],'continuation_pairs':[],'limits':['Historical one-RNG realized returns, not conditional expectation.','BC success prefix fork verifies observed history only; marked approximate pairing, not strict physics snapshot.']}
 for rnd in ['round1','round2']:
  path=TEST/'final_collapse_rootcause'/rnd/'result.json';j=json.loads(path.read_text());assert j['strict_state_hash_all_branches_match'];p=j['branches']['REF_REF']['probe'];assert p['same_physical_state'];o=torch.as_tensor(p['observations'],device=device,dtype=torch.float32);a=torch.as_tensor(p['past_actions_production_raw'],device=device,dtype=torch.float32);st=torch.as_tensor(p['episode_steps'],device=device)
  result['reuse'][rnd]={'path':str(path),'env_contract':j['env_contract'],'state_hashes':p['state_hashes'],'strict_physics_history_rng':True,'branches':list(j['branches'])}
  q={}
  with torch.no_grad():
   for name,c in [('ready',td),('MC',mc)]:
    z=encode_replay_contexts(c,o,a,st,700);ctx=tuple(x[:,-1] for x in z);q[name]={}
    for ac,values in p['actions'].items():
     aa=torch.as_tensor(values['executed'],device=device,dtype=torch.float32);u,v=c.q_from_context(ctx,aa);q[name][ac]=torch.cat([u,v],-1).cpu().numpy()
  branches=j['branches'];reference=branches['REF_REF']
  for branch,b in branches.items():
   if b['continuation']!='REF' or b['action']=='REF':continue
   for i,(e,base) in enumerate(zip(b['episodes'],reference['episodes'])):
    assert e['seed']==base['seed'];row={'prefix':rnd,'seed':e['seed'],'action':b['action'],'continuation':'REF','mc':e['mc_from_fork'],'reference_mc':base['mc_from_fork'],'delta_mc':e['mc_from_fork']-base['mc_from_fork'],'success':e['success'],'reference_success':base['success'],'length':e['length'],'reference_length':base['length'],'q':{name:{'candidate':scores[b['action']][i].tolist(),'reference':scores['REF'][i].tolist(),'delta':(scores[b['action']][i]-scores['REF'][i]).tolist()} for name,scores in q.items()},'strict_pair':True};result['pairs'].append(row)
  for ac in ['REF','BAD']:
   if ac+'_BAD' in branches and ac+'_REF' in branches:
    for i,(e,base) in enumerate(zip(branches[ac+'_BAD']['episodes'],branches[ac+'_REF']['episodes'])):result['continuation_pairs'].append({'prefix':rnd,'action':ac,'seed':e['seed'],'BC_mc':base['mc_from_fork'],'BAD_mc':e['mc_from_fork'],'delta_mc':e['mc_from_fork']-base['mc_from_fork'],'BC_success':base['success'],'BAD_success':e['success']})
 # Existing successful BC contexts, explicitly approximate state pairing.
 j=json.loads((TEST/'mean_multi_upstream_rootcause/round4/result.json').read_text());result['reuse']['BC_success']={'env_contract':j['env_contract'],'strict_physics_history_rng':False,'pairing':'same reset/deterministic prefix and observed history only'}
 for ac,b in j['branches'].items():
  if ac=='0.0':continue
  for i,(e,base) in enumerate(zip(b['episodes'],j['branches']['0.0']['episodes'])):
   dq=b['paired_delta_q1_q2'][i];result['pairs'].append({'prefix':'BC_success','seed':e['seed'],'action':ac,'continuation':'BC','delta_mc':e['mc_from_fork']-base['mc_from_fork'],'mc':e['mc_from_fork'],'reference_mc':base['mc_from_fork'],'q':{'ready':{'delta':dq}},'strict_pair':False,'success':e['success'],'reference_success':base['success']})
 result['contradictions']=[x for x in result['pairs'] if any(np.mean(z['delta'])*x['delta_mc']<0 for z in x['q'].values())];dump(out/'existing_pairs.json',result)
 print(json.dumps({'pairs':len(result['pairs']),'contradictions':result['contradictions'],'continuation_pairs':result['continuation_pairs']}),flush=True)
if __name__=='__main__':main()
