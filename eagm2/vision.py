"""Actual RGB policy with a frozen 32-dimensional policy adapter.

No physical/simulator state is accepted by deployment APIs. BC targets may be
provided to fit_base, but never become forward inputs. Adaptation optimizes
external theta only after freezing encoder/base/adapter basis.
"""
from __future__ import annotations
import torch
from torch import nn

class DeterministicPool2d(nn.Module):
    """Adaptive 2x2 averaging using deterministic reduction backward."""
    def forward(self,x):
        h,w=x.shape[-2:]
        return torch.stack([
            torch.stack([x[...,i*h//2:((i+1)*h+1)//2,
                              j*w//2:((j+1)*w+1)//2].mean((-2,-1))
                         for j in range(2)],dim=-1)
            for i in range(2)],dim=-2)

class VisionPolicy(nn.Module):
    def __init__(self,obs_dim=32,record_dim=29,adapter_dim=32):
        super().__init__();self.obs_dim=obs_dim;self.record_dim=record_dim;self.adapter_dim=adapter_dim
        self.encoder=nn.Sequential(nn.Conv2d(3,16,5,stride=2,padding=2),nn.SiLU(),nn.Conv2d(16,24,3,stride=2,padding=1),nn.SiLU(),nn.Conv2d(24,32,3,stride=2,padding=1),nn.SiLU(),DeterministicPool2d(),nn.Flatten(),nn.Linear(128,obs_dim),nn.Tanh())
        self.base=nn.Sequential(nn.Linear(obs_dim+record_dim,64),nn.SiLU(),nn.Linear(64,14))
        self.basis=nn.Linear(obs_dim+record_dim,14*adapter_dim,bias=False)
        nn.init.normal_(self.basis.weight,std=.03)
    def encode(self,rgb):
        if rgb.ndim==3:rgb=rgb.unsqueeze(0)
        if rgb.shape[-1]==3:rgb=rgb.movedim(-1,-3)
        if rgb.ndim!=4 or rgb.shape[1]!=3:raise ValueError('RGB must be NHWC or NCHW with 3 channels')
        if rgb.dtype==torch.uint8:rgb=rgb.float()/255.
        return self.encoder(rgb)
    def action_from_latent(self,z,c_obs=None,theta=None):
        if self.record_dim:
            if c_obs is None or c_obs.shape[-1]!=self.record_dim:raise ValueError('Observable execution record required')
            features=torch.cat((z,c_obs),-1)
        else:features=z
        raw=self.base(features)
        if theta is not None:
            if theta.shape[-1]!=self.adapter_dim:raise ValueError('Wrong adapter dimension')
            B=self.basis(features).reshape(*features.shape[:-1],14,self.adapter_dim)
            raw=raw+.1*(B@theta.unsqueeze(-1)).squeeze(-1)
        return .03*torch.tanh(raw)
    def forward(self,rgb,c_obs=None,theta=None):
        return self.action_from_latent(self.encode(rgb),c_obs,theta)
    def freeze_base(self):
        self.eval()
        for p in self.parameters():p.requires_grad_(False)
        return self

def fit_base(policy,rgb,records,actions,updates=500,batch_size=64,seed=0,lr=1e-3):
    """Train on training-episode RGB/action demonstrations only, then freeze.

    Caller owns episode split provenance and counts demonstration generation.
    The adapter basis is fixed from initialization; theta starts at zero.
    """
    policy.train();params=list(policy.encoder.parameters())+list(policy.base.parameters())
    opt=torch.optim.Adam(params,lr=lr);gen=torch.Generator().manual_seed(seed);logs=[]
    for step in range(updates):
        idx=torch.randint(len(rgb),(batch_size,),generator=gen).to(rgb.device)
        pred=policy(rgb[idx],None if records is None else records[idx])
        loss=((pred-actions[idx])/.03).square().mean()
        opt.zero_grad();loss.backward();opt.step()
        if step%50==0 or step==updates-1:logs.append({'update':step+1,'normalized_bc_mse':float(loss.detach())})
    policy.freeze_base();return logs
