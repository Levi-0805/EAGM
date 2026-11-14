"""Evaluate frozen legacy policies on a new exploratory holdout; never retrain.

This is a distinct evaluation protocol, not a repair of historical results.
"""
import argparse,hashlib,json,time
import numpy as np
from eagm2.runtime import ROOT,new_run,seed_all,stream_seed,Budget,dump,environment
from eagm2.data import make_env,collect
from eagm2.rollout import physical
from training.stage5_common import CFG,H_EVAL

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',required=True,type=int)
    p.add_argument('--source-attempt',default='legacy_reproduction_v1')
    p.add_argument('--attempt',default='legacy_fresh_holdout_v1');a=p.parse_args()
    source=ROOT/'runs/eagm2_development/tracking'/f'seed{a.seed}'/a.source_attempt/'results.json'
    raw=source.read_bytes();prior=json.loads(raw)
    # Freeze policies before creating or querying any holdout episodes.
    policies={'theta0':np.zeros(28),**{k:np.asarray(v['theta_final']) for k,v in prior['results'].items()}}
    if any(v.shape!=(28,) or not np.isfinite(v).all() for v in policies.values()):raise ValueError('Invalid saved policy')
    out=new_run('tracking',a.seed,a.attempt);seed_all(a.seed);budget=Budget();begin=time.perf_counter()
    derived=stream_seed(a.seed,'legacy_fresh_holdout_v1')
    dump(out/'config.json',dict(phase='exploratory_frozen_policy_holdout',root_seed=a.seed,stream_seed=derived,source=source,source_sha256=hashlib.sha256(raw).hexdigest(),policies=policies,episodes=30,delay_counts={'0':10,'1':10,'2':10},horizon=H_EVAL,training_updates=0,note='Distinct new evaluation distribution, not original Stage5 reproduction; no new research roots. No tuning or policy optimization.'))
    dump(out/'environment.json',environment())
    env=make_env('tracking',CFG)
    data=collect(env,'tracking',derived,budget,train_eps=0,val_eps=0,test_eps=30)
    starts=data['starts']['development_test'];rows=[]
    for s in starts:
        returns={k:physical(env,v,s,H_EVAL,budget,'frozen_policy_holdout') for k,v in policies.items()}
        rows.append(dict(episode_id=s['episode_id'],delay=s['delay'],frame=s['frame'],returns=returns))
    summary={}
    for stratum,subset in [('all',rows),('delay2',[r for r in rows if r['delay']==2])]:
        summary[stratum]={k:dict(n=len(subset),mean=float(np.mean([r['returns'][k] for r in subset])),paired_delta_vs_theta0=float(np.mean([r['returns'][k]-r['returns']['theta0'] for r in subset]))) for k in policies}
    assert source.read_bytes()==raw,'Source result changed during evaluation'
    dump(out/'results.json',dict(status='completed',phase='exploratory_frozen_policy_holdout',per_start=rows,summary=summary,budget=budget.as_dict(),wall_seconds=time.perf_counter()-begin,source_unchanged=True))
    dump(out/'episode_manifest.json',[dict(id=e['id'],episode_seed=e['episode_seed'],delay=e['delay']) for e in data['episodes']])
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
