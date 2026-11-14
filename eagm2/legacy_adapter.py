"""Frozen EAGMBase behavior behind the EAGM2 runner API.

Only adapts call/return structure. Training on the new common-data protocol is
not a reproduction of the historical protocol; that is a separate experiment.
"""
from __future__ import annotations
import torch
from torch import nn
from models.eagm import EAGMBase


class LegacyWorldAdapter(nn.Module):
    """Exact legacy command/dynamics equations, with no hidden-state inputs."""
    memory = 'none'
    prior = 'legacy'
    obs_dim = 28
    width = 160
    max_increment = 0.03
    is_privileged = False
    _forwarded_stats = frozenset(('lower','upper','alpha','xmean','xstd','ymean','ystd','umean','ustd',
                                  'data_lower_margin','data_upper_margin'))

    def __init__(self, stats):
        super().__init__()
        required=('lower','upper','alpha','xmean','xstd','ymean','ystd','umean','ustd')
        missing=[key for key in required if key not in stats]
        if missing: raise ValueError(f'legacy statistics missing {missing}')
        if any(torch.as_tensor(stats[key]).shape != (28,) for key in ('xmean','xstd','ymean','ystd')):
            raise ValueError('LegacyWorldAdapter only supports original 28D physical observations')
        # Freeze semantics, not learned parameters: the wrapped model can train.
        self.legacy = EAGMBase(stats, width=160)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name in self._forwarded_stats:
                return getattr(super().__getattr__('legacy'), name)
            raise

    def step(self, x, c, a, f_prev=None, h=None):
        if x.shape[-1] != 28: raise ValueError('legacy model requires physical 28D input')
        if h is not None: raise ValueError('legacy feedforward model has no execution latent')
        newx, u = self.legacy(x,c,a)
        return newx,u,None

    forward=step

    def initial_state(self,x):
        return None

    def observe(self,h_prior,x,c,prev_a):
        if h_prior is not None:raise ValueError('legacy model has no hidden state')
        return None

    def burnin(self,xhist,chist,ahist,detach=True):
        return None

    def parameter_counts(self):
        return {'total':sum(p.numel() for p in self.parameters()),
                'trainable':sum(p.numel() for p in self.parameters() if p.requires_grad)}
