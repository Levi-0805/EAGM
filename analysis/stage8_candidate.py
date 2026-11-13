"""Frozen single-seed engineering screen on joint tracking validation only."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vendor/legacy_sim')]
import numpy as np
import torch
from scripts.run_stage6c2_pathdir import collect_randomized,TRAIN_RANGES,pack_rows,eval_model
from controllers.partial_unknown import PartialUnknownArmEnv,ControllerBConfig
from training.dataset import make_split,flatten_split,fit_stats,pick_starts
from training.stage6c_common import random_probe_thetas,build_labels_at_thetas
from training.paired_train import clone_model,train_pair,train_mc
from training.seeds import seed_everything
from training.stage8_reliability import labels_with_reliability
from models.eagm import EAGMBase,McWorld
SEED=int(sys.argv[1]) if len(sys.argv)>1 else 40
OUT=ROOT/'runs'/f'stage8_c1_seed{SEED}'
def main():
 OUT.mkdir(exist_ok=False);t=time.time();seed=SEED;seed_everything(seed)
 env=PartialUnknownArmEnv(ControllerBConfig(episode_randomize=True,**TRAIN_RANGES,gain=1.,extra_lag=1,rate_limit=.05,smooth_beta=.2))
 episodes=collect_randomized(env,36,200,seed);split=make_split(36,seed=seed,n_train=24,n_val=6,n_test=6)
 trr=flatten_split(episodes,split.train_eps);vr=flatten_split(episodes,split.val_eps);stats=fit_stats(trr,env_lower=env.lower,env_upper=env.upper);tr=pack_rows(trr)
 starts=pick_starts(trr,delay=None,max_per_episode=1,max_total=12);vs=pick_starts(vr,delay=None,max_per_episode=1,max_total=6)
 labs,wlabs,n=labels_with_reliability(env,starts,random_probe_thetas(12,seed+101));vl,vn=build_labels_at_thetas(env,vs,random_probe_thetas(6,seed+201),feasible=True)
 torch.save(dict(episodes=episodes,split=split,labels=labs,weighted=wlabs,validation_labels=vl),OUT/'data.pt')
 seed_everything(seed+11);init=EAGMBase(stats);base=clone_model(init);dire=clone_model(init);weighted=clone_model(init);twin=clone_model(init)
 train_pair(base,dire,tr,updates=3000,use_history=False,dir_labels=labs,batch_seed=seed+30)
 train_pair(twin,weighted,tr,updates=3000,use_history=False,dir_labels=wlabs,batch_seed=seed+30)
 assert all(torch.equal(a,b) for a,b in zip(base.parameters(),twin.parameters()))
 seed_everything(seed+7);mc=McWorld(stats);train_mc(mc,tr,3000,batch_seed=seed+30)
 results={}
 for name,m,is_mc in [('mc',mc,True),('base',base,False),('dir',dire,False),('reliability',weighted,False)]:
  results[name]=eval_model(m,vr,vl,starts,env,vs,seed,False,is_mc)
  torch.save(dict(state=m.state_dict(),stats=stats),OUT/(name+'.pt'))
 a,b=results['reliability'],results['dir'];wins=sum(x['return_']>y['return_'] for x,y in zip(a['per_start'],b['per_start']))
 gate=a['policy_return_mean']>b['policy_return_mean'] and a['cosine_full_mean']>b['cosine_full_mean'] and wins>=4
 payload=dict(seed=seed,split='validation_only',results=results,gate=bool(gate),wins=wins,wall_seconds=time.time()-t,budget=dict(collection_steps=7200,train_dynamics_steps=4800,train_fd_steps=n,train_fd_queries=n//8,validation_fd_steps=vn,policy_eval_steps=4*2*6*16,wm_updates_per_model=3000,policy_updates=50,device='cpu',extra_disposable_base_updates=3000),weight_mean=float(np.mean([l['weights'].mean() for l in labs])))
 (OUT/'results.json').write_text(json.dumps(payload,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))
 print('DONE gate',gate,'wins',wins,flush=True)
if __name__=='__main__':main()
