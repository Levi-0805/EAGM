"""C2 equal-dimensional feature ablation; C3 leave-episode-out calibration screen."""
from pathlib import Path
import json,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'runs/stage7c2_true_ee_current_sites_seed20'
def main():
 t=time.time();d=torch.load(P/'data.pt',weights_only=False)
 rows=lambda ids:[r for e in d['episodes'] if e['episode_id'] in ids for r in e['rows']]
 tr,va=rows(d['split'].train_eps),rows(d['split'].val_eps)
 # Both have 43 inputs including intercept; replace current action with last observed action.
 def features(rr,history):
  return np.array([np.r_[r['c_obs'],r['a_hist'][-1] if history else r['a'],1.] for r in rr])
 out={}
 for history in [False,True]:
  x=features(tr,history);z=features(va,history);y=np.array([r['u_exec'] for r in tr]);target=np.array([r['u_exec'] for r in va]);mean=x.mean(0);std=x.std(0);std[-1]=1;std=np.maximum(std,1e-6);mean[-1]=0;x=(x-mean)/std;z=(z-mean)/std
  beta=np.linalg.solve(x.T@x+1e-3*np.eye(x.shape[1]),x.T@y);pred=z@beta
  out['lagged_action' if history else 'current_action']=dict(validation_command_mae=float(abs(pred-target).mean()),parameters=int(beta.size),per_episode={str(e):float(abs(pred[np.array([r['episode_id']==e for r in va])]-target[np.array([r['episode_id']==e for r in va])]).mean()) for e in d['split'].val_eps})
 # C3 numerical reliability calibration: held-out episode prediction of normalized gradient error
 labs=torch.load(P/'true_labels.pt',weights_only=False);x=[];y=[]
 # Labels alone cannot provide model error targets without evaluating a fitted model; quantify independent sample ceiling.
 out['C3']=dict(independent_training_labels=len(labs),target_dimension=28,decision='DEFER: 12 episode labels cannot support a flexible calibrated reliability model plus an independent calibration split; no test-derived fit performed',minimum_next_design='train-only leave-episode-out scalar calibration, validated before any update gating')
 out['wall_seconds']=time.time()-t;out['simulator_steps']=0;out['note']='C2 linear feature replacement is a feasibility ablation, not an EAGM architecture win; same feature count/ridge fixed before validation; no hyperparameter search.'
 (ROOT/'analysis/stage8_candidate_screen.json').write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()
