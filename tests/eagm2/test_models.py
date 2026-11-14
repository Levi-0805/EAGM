import copy
import io
import unittest
import torch
from eagm2.models import ExecutionWorldModel, PrivilegedExecutionWorldModel, capacity_matched_width
from eagm2.prior import observable_step

class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(611);torch.set_num_threads(1)
        self.x=torch.randn(2,28)*.01
        self.c=torch.zeros(2,29);self.c[:,28]=torch.tensor([1.,2.])
        self.a=torch.randn(2,14)*.005;self.prev=torch.zeros(2,14)
        self.xh=torch.randn(2,8,28)*.01;self.ch=self.c[:,None].repeat(1,8,1);self.ah=torch.randn(2,8,14)*.003
        self.stats={'xstd':torch.ones(28),'ystd':torch.ones(28)*.01,'ustd':torch.ones(14)*.1}

    def test_causal_prefix_future_perturbation(self):
        for mem in ('recurrent','window'):
            m=ExecutionWorldModel(self.stats,memory=mem,prior='delay',width=32)
            h=m.burnin(self.xh[:,:4],self.ch[:,:4],self.ah[:,:4])
            changed=self.xh.clone();changed[:,4:]+=1000
            h2=m.burnin(changed[:,:4],self.ch[:,:4],self.ah[:,:4])
            torch.testing.assert_close(h,h2)
            # No simulator object or hidden field is accepted by deployable API.
            with self.assertRaises(TypeError):m.step(self.x,self.c,self.a,self.prev,h,privileged_state=torch.randn(2,74))
            with self.assertRaises(TypeError):m.step(self.x,self.c,self.a,self.prev,h,controller_b={'u_prev':1})

    def test_second_derivative_and_chain_fd(self):
        for mem in ('none','recurrent','window'):
            m=ExecutionWorldModel(self.stats,memory=mem,prior='delay',width=24).double()
            x,c,p=self.x.double(),self.c.double(),self.prev.double()
            h=m.burnin(self.xh.double(),self.ch.double(),self.ah.double())
            theta=torch.zeros(14,dtype=torch.float64,requires_grad=True)
            def objective(theta):
                xx,cc,pp,hh=x,c,p,h
                total=theta.sum()*0
                for k in range(4):
                    a=.01*torch.tanh(theta).expand(2,-1)
                    xx,u,hh=m.step(xx,cc,a,pp,hh)
                    cc,pp=observable_step(cc,a,m.lower,m.upper,m.alpha)
                    total=total+xx.square().sum()
                return total
            J=objective(theta);g=torch.autograd.grad(J,theta,create_graph=True)[0]
            second=torch.autograd.grad(g.square().sum(),tuple(m.parameters()),allow_unused=True)
            self.assertTrue(any(v is not None and torch.isfinite(v).all() for v in second))
            fd=[];eps=1e-5
            for j in range(14):
                e=torch.eye(14,dtype=torch.float64)[j]*eps
                fd.append((objective(theta.detach()+e)-objective(theta.detach()-e))/(2*eps))
            torch.testing.assert_close(g,torch.stack(fd),atol=1e-7,rtol=1e-4)

    def test_alignment_observe_window_and_checkpoint(self):
        m=ExecutionWorldModel(self.stats,memory='window',prior='delay',width=32)
        h=m.burnin(self.xh,self.ch,self.ah)
        y,u,hn=m.step(self.x,self.c,self.a,self.prev,h)
        cn,_=observable_step(self.c,self.a,m.lower,m.upper,m.alpha)
        torch.testing.assert_close(hn[:,-1],m.record(y,cn,self.a))
        buf=io.BytesIO();torch.save(m.state_dict(),buf);buf.seek(0)
        restored=ExecutionWorldModel(self.stats,memory='window',prior='delay',width=32)
        restored.load_state_dict(torch.load(buf,weights_only=True))
        torch.testing.assert_close(y,restored.step(self.x,self.c,self.a,self.prev,h)[0])

    def test_visual_latent_dimension_and_capacity(self):
        m=ExecutionWorldModel({},obs_dim=32,memory='recurrent',prior='delay',width=40)
        x=torch.zeros(2,32);h=m.initial_state(x)
        self.assertEqual(m.step(x,self.c,self.a,self.prev,h)[0].shape,(2,32))
        target=ExecutionWorldModel(self.stats,memory='recurrent',width=64).parameter_counts()['trainable']
        before=torch.get_rng_state().clone()
        width,count=capacity_matched_width(self.stats,target,memory='none')
        self.assertLessEqual(abs(count-target)/target,.05)
        torch.testing.assert_close(torch.get_rng_state(),before)
        print('capacity match',target,width,count)

    def test_privileged_explicit_separation(self):
        m=PrivilegedExecutionWorldModel(self.stats,width=32)
        self.assertTrue(m.is_privileged)
        with self.assertRaises(TypeError):m.step(self.x,self.c,self.a,self.prev)
        y,u=m.step_privileged(self.x,self.c,self.a,self.prev,privileged_state=torch.zeros(2,74))
        self.assertEqual(y.shape,self.x.shape)
        self.assertEqual(u.shape,self.a.shape)

if __name__=='__main__':unittest.main(verbosity=2)
