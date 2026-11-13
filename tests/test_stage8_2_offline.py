"""No new physical steps: actor FD and saved intervention invariants."""
import sys,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vendor/legacy_sim')]
import numpy as np,torch
from training.rollout import actor
from training.stage8_2_dynamics import ReplaceJacobian,N
class AuditTests(unittest.TestCase):
 def test_actor_fd(self):
  rng=np.random.default_rng(82)
  for _ in range(6):
   x=torch.tensor(rng.normal(0,.1,28),dtype=torch.double);tgt=torch.tensor(rng.normal(0,.1,14),dtype=torch.double);ref=tgt+.02
   v=torch.zeros(28,dtype=torch.double);jac=torch.autograd.functional.jacobian(lambda th:actor(th,x,tgt,ref),v).numpy();fd=np.zeros((14,28))
   for i in range(28):
    d=torch.zeros_like(v);d[i]=1e-5;fd[:,i]=((actor(v+d,x,tgt,ref)-actor(v-d,x,tgt,ref))/(2e-5)).numpy()
   self.assertLess(np.max(abs(jac-fd)),1e-10)
 def test_custom_vjp_numerical_linearization(self):
  torch.manual_seed(82);z=torch.zeros(N,dtype=torch.double,requires_grad=True);a=torch.zeros(14,dtype=torch.double,requires_grad=True);j=torch.randn(N,N+14,dtype=torch.double);value=torch.randn(N,dtype=torch.double);weight=torch.randn(N,dtype=torch.double)
  out=ReplaceJacobian.apply(z,a,value,j);self.assertTrue(torch.equal(out,value));g=torch.cat(torch.autograd.grad(out@weight,(z,a))).numpy();v=torch.randn(N+14,dtype=torch.double);h=1e-5;fd=(((value+j@(h*v))@weight-(value-j@(h*v))@weight)/(2*h)).item();self.assertAlmostEqual(fd,float(g@v.numpy()),places=7)
 def test_saved_forward_invariance_and_budget(self):
  m=json.loads((ROOT/'runs/stage8_2a_single_step/multistep.json').read_text())
  for i in range(6):
   for h in [1,2,4,8]:
    rr=[r for r in m if r['start']==i and r['horizon']==h];self.assertEqual(len(rr),4)
    for r in rr:self.assertTrue(np.array_equal(r['state'],rr[0]['state']));self.assertEqual(r['forward_value'],rr[0]['forward_value'])
  b=json.loads((ROOT/'runs/stage8_2_summary/total_budget.json').read_text());self.assertLessEqual(b['D_control_steps'],20000);self.assertLessEqual(b['ABC_control_steps'],40000)
if __name__=='__main__':unittest.main()
