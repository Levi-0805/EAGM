import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vendor/legacy_sim')]
import torch,numpy as np
from scripts.run_stage7c1_diagnosis import make_env
from training.stage8_2_dynamics import Kernel,Budget,ReplaceJacobian,N

def run():
 d=torch.load(ROOT/'runs/stage7c2_true_ee_current_sites_seed20/data.pt',weights_only=False);live=make_env();scratch=make_env();budget=Budget(100);errs=[]
 for st in d['test_starts']:
  live.restore(st['snapshot']);k=Kernel(scratch,st['snapshot'],budget);z=k.initial
  for t in range(8):
   a=np.ones(14)*.002;live.step(a);z=k.forward(z,a,t);errs.append(np.max(abs(z[:28]-live.observe())))
 assert max(errs)<1e-11,max(errs)
 z=torch.zeros(N,requires_grad=True);a=torch.zeros(14,requires_grad=True);j=torch.randn(N,N+14);v=torch.randn(N);y=ReplaceJacobian.apply(z,a,v,j);gz,ga=torch.autograd.grad(y.sum(),(z,a));assert torch.allclose(torch.cat([gz,ga]),j.sum(0),atol=1e-5)
 print('PASS full snapshot execution/warm-start replay and custom VJP; 96 control steps',max(errs))
if __name__=='__main__':run()
