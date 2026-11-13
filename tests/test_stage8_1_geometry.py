import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vendor/legacy_sim')]
import torch,numpy as np
from scripts.run_stage7c1_diagnosis import make_env
from training.stage8_1_geometry import Geometry,TrueFK,reward,rollout
from tasks.reach import FKSurrogate
from models.eagm import EAGMBase
from training.rollout import set_reach_fk_surrogate,imagined_eagm
P=ROOT/'runs/stage7c2_true_ee_current_sites_seed20'
def run():
 env=make_env();g=Geometry(env.model);q=np.zeros(14);before=env.data.qpos.copy();warm=env.data.qacc_warmstart.copy()
 p,j=g.evaluate(q);fd=np.column_stack([(g.evaluate(q+np.eye(14)[i]*1e-5)[0]-g.evaluate(q-np.eye(14)[i]*1e-5)[0])/2e-5 for i in range(14)])
 assert np.max(abs(j-fd))<1e-7
 assert np.array_equal(before,env.data.qpos) and np.array_equal(warm,env.data.qacc_warmstart)
 fk=FKSurrogate();fk.load_state_dict(torch.load(P/'fk.pt',weights_only=True));fk.eval()
 for v in fk.parameters():v.requires_grad_(False)
 x=torch.zeros(28,requires_grad=True);a=torch.zeros(14,requires_grad=True);prev=torch.ones(14)*.01;goals=np.ones(6)*.1
 vals={};gs={}
 for c in 'ABCD':
  r=reward(c,x,a,prev,goals,fk,g);vals[c]=r.detach();gs[c]=torch.autograd.grad(r,(x,a))
 assert torch.equal(vals['A'],vals['C']) and torch.equal(vals['B'],vals['D'])
 for i in [0,1]:assert torch.equal(gs['A'][i],gs['D'][i]) and torch.equal(gs['B'][i],gs['C'][i])
 assert torch.autograd.gradcheck(lambda z:TrueFK.apply(z,g),(torch.zeros(14,dtype=torch.float64,requires_grad=True),),eps=1e-6,atol=1e-5)
 d=torch.load(P/'data.pt',weights_only=False);b=torch.load(P/'dir_random.pt',weights_only=False);m=EAGMBase(b['stats']);m.load_state_dict(b['state']);st=d['test_starts'][0];th=torch.zeros(28,requires_grad=True);set_reach_fk_surrogate(fk)
 old=imagined_eagm(m,th,st);new=rollout(m,th,st,'A',fk,g)
 assert torch.allclose(old,new,atol=1e-8)
 assert torch.allclose(torch.autograd.grad(old,th)[0],torch.autograd.grad(new,th)[0],atol=1e-8)
 print('PASS analytic geometry vs FD, autograd, no live-state mutation, A/C B/D values, A/D B/C gradients, native rollout parity; zero control steps')
if __name__=='__main__':run()
