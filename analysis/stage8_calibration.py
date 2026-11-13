"""C3 train-label-only LOEO feasibility; not independent model calibration."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vendor/legacy_sim')]
import torch,numpy as np
from tasks.reach import FKSurrogate
from models.eagm import EAGMBase
from training.rollout import set_reach_fk_surrogate
from scripts.run_stage7c1_diagnosis import model_grad
from training.eval_metrics import cosine
p=ROOT/'runs/stage7c2_true_ee_current_sites_seed20';fk=FKSurrogate();fk.load_state_dict(torch.load(p/'fk.pt',weights_only=True));set_reach_fk_surrogate(fk)
b=torch.load(p/'dir_random.pt',weights_only=False);m=EAGMBase(b['stats']);m.load_state_dict(b['state']);labs=torch.load(p/'true_labels.pt',weights_only=False)
x=[];y=[]
for l in labs:
 g,_=model_grad(m,l['start'],l['theta']);x.append([1.,np.log(np.linalg.norm(g)+1e-12)]);y.append(cosine(g,l['g']))
x=np.array(x);y=np.array(y);pred=[];constant=[]
for i in range(len(y)):
 mask=np.arange(len(y))!=i;beta=np.linalg.solve(x[mask].T@x[mask]+1e-3*np.eye(2),x[mask].T@y[mask]);pred.append(float(x[i]@beta));constant.append(float(y[mask].mean()))
result=dict(n=len(y),predictions=pred,targets=y.tolist(),loo_mse=float(np.mean((pred-y)**2)),constant_loo_mse=float(np.mean((constant-y)**2)),sim_steps=0,limitation='World model already trained on all these labels. Only scalar regressor is cross-validated. This cannot certify independent calibration; no policy gating applied.')
(ROOT/'analysis/stage8_calibration_screen.json').write_text(json.dumps(result,indent=2));print(result['loo_mse'],result['constant_loo_mse'])
