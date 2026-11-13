"""Use 192 of reserved control steps for actual theta0/final state geometry."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vendor/legacy_sim')]
import numpy as np
import torch
from scripts.run_stage7c1_diagnosis import make_env
from training.stage8_1_geometry import Geometry
from training.rollout import actor_raw_numpy,ACTION_LIMIT
from training.eval_metrics import cosine
from tasks.reach import FKSurrogate
P=ROOT/'runs/stage7c2_true_ee_current_sites_seed20';O=ROOT/'runs/stage8_1_geometry'
def main():
 assert not (O/'physical_state_geometry.json').exists()
 mainbudget=json.loads((O/'budget.json').read_text());assert mainbudget['physical_steps']+192<=20000
 t=time.time();env=make_env();geo=Geometry(env.model);d=torch.load(P/'data.pt',weights_only=False);pol=json.loads((P/'policies.json').read_text());fk=FKSurrogate();fk.load_state_dict(torch.load(P/'fk.pt',weights_only=True));records=[];steps=0
 for tag,th in [('theta0_physical',np.zeros(28)),('old_final_physical',np.array(pol['dir_random']['theta']))]:
  for st in d['test_starts']:
   env.restore(st['snapshot'])
   for k in range(16):
    a=np.clip(actor_raw_numpy(th,env.observe(),env.control_state()[:14],st['refs'][k]),-ACTION_LIMIT,ACTION_LIMIT)
    assert steps<192;env.step(a);steps+=1;q=env.observe()[:14];goal=np.asarray(st['ee_goals']);pt,jt=geo.evaluate(q);qt=torch.tensor(q,dtype=torch.float32,requires_grad=True);ps=fk(qt).detach().numpy();js=torch.autograd.functional.jacobian(fk,qt).detach().numpy();gt=-jt.T@(pt-goal)/3;gs=-js.T@(ps-goal)/3
    records.append(dict(group=tag,episode=st['episode_id'],frame=k,q=q,goals=goal,p_true=pt,p_fk=ps,J_true=jt,J_fk=js,g_true=gt,g_fk=gs,g_trueJ_fkPosition=-jt.T@(ps-goal)/3,g_fkJ_truePosition=-js.T@(pt-goal)/3,position_rmse=float(np.sqrt(np.mean((pt-ps)**2))),position_mae=float(np.mean(abs(pt-ps))),jacobian_rmse=float(np.sqrt(np.mean((jt-js)**2))),jacobian_relative=float(np.linalg.norm(jt-js)/(np.linalg.norm(jt)+1e-12)),jacobian_cosine=cosine(jt,js),reward_gradient_cosine=cosine(gt,gs),reward_gradient_relative=float(np.linalg.norm(gt-gs)/(np.linalg.norm(gt)+1e-12)),position_value_true=float(-np.mean((pt-goal)**2)),position_value_fk=float(-np.mean((ps-goal)**2))))
 (O/'physical_state_geometry.json').write_text(json.dumps(records,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))
 (O/'supplemental_budget.json').write_text(json.dumps(dict(control_steps=steps,total_with_main=mainbudget['physical_steps']+steps,wall_seconds=time.time()-t,geometry=geo.metadata()),indent=2));print('total',mainbudget['physical_steps']+steps)
if __name__=='__main__':main()
