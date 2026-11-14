import unittest
import torch
from torch import nn
from eagm2.baselines import paml_loss, vagram_loss, QCritic, ValueCritic

class BilinearCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.tensor(1.,dtype=torch.float64))
    def forward(self,state,action):
        return self.weight*(state*action).sum(-1)

class BaselineTests(unittest.TestCase):
    def test_paml_matches_analytic_gradient_difference(self):
        critic=BilinearCritic()
        theta=torch.tensor([.3],dtype=torch.float64,requires_grad=True)
        real=torch.tensor([[1.],[2.]],dtype=torch.float64)
        pred=torch.tensor([[1.5],[2.5]],dtype=torch.float64,requires_grad=True)
        actor=lambda x:theta*x
        loss=paml_loss(critic,actor,real,pred,[theta])
        expected=(pred.square().mean()-real.square().mean()).square()
        torch.testing.assert_close(loss,expected)
        actual=torch.autograd.grad(loss,pred,retain_graph=True)[0]
        analytic=4*(pred.square().mean()-real.square().mean())*pred/pred.numel()
        torch.testing.assert_close(actual,analytic)
        self.assertIsNone(torch.autograd.grad(loss,critic.weight,allow_unused=True)[0])
        self.assertTrue(critic.weight.requires_grad)

    def test_paml_zero_for_equal_states(self):
        theta=torch.tensor([.3],dtype=torch.float64,requires_grad=True)
        real=torch.tensor([[1.],[2.]],dtype=torch.float64)
        pred=real.clone().requires_grad_(True)
        loss=paml_loss(BilinearCritic(),lambda x:x*theta,real,pred,[theta])
        self.assertAlmostEqual(float(loss.detach()),0)
        torch.testing.assert_close(torch.autograd.grad(loss,pred)[0],torch.zeros_like(pred))

    def test_vagram_coordinatewise_not_squared_dot_product(self):
        real=torch.zeros((1,2),dtype=torch.float64)
        pred=torch.tensor([[1.,-1.]],dtype=torch.float64,requires_grad=True)
        value=lambda x:x.sum(-1)
        loss=vagram_loss(value,real,pred)
        self.assertAlmostEqual(float(loss.detach()),1.)
        self.assertNotEqual(float(loss.detach()),0.) # squared gradient dot error would cancel
        torch.testing.assert_close(torch.autograd.grad(loss,pred)[0],pred)

    def test_vagram_units_and_detached_weights(self):
        real=torch.tensor([[1.,2.]],dtype=torch.float64)
        pred=torch.tensor([[1.4,1.7]],dtype=torch.float64,requires_grad=True)
        value=nn.Linear(2,1,bias=False).double()
        with torch.no_grad():value.weight.copy_(torch.tensor([[2.,-3.]],dtype=torch.float64))
        raw=vagram_loss(value,real,pred)
        scaled=vagram_loss(value,real,pred,state_scale=torch.tensor([.1,10.],dtype=torch.float64))
        torch.testing.assert_close(raw,scaled)
        self.assertIsNone(torch.autograd.grad(scaled,value.weight,allow_unused=True)[0])

    def test_neural_critic_and_actor_model_second_derivative(self):
        torch.manual_seed(3)
        critic=QCritic(3,2,width=8).double()
        actor=nn.Linear(3,2).double()
        model=nn.Linear(3,3).double()
        x=torch.randn(5,3,dtype=torch.float64)
        pred=model(x)
        loss=paml_loss(critic,actor,x,pred,actor.parameters())
        grads=torch.autograd.grad(loss,tuple(model.parameters()))
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(float(g.norm()) for g in grads),0)
        value=ValueCritic(3,width=8).double()
        vg=vagram_loss(value,x,model(x))
        self.assertTrue(torch.isfinite(vg))

if __name__=='__main__':unittest.main()
