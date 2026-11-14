"""Explicit PAML-critic / VaGraM-inspired objective adaptations.

These are objective-level adaptations for EAGM's deterministic actor and partial
observable interface, not full reproductions of the published RL algorithms.
Critics must be trained only on training trajectories and their cost logged.
"""
from __future__ import annotations
from contextlib import contextmanager
from collections.abc import Callable, Iterable
import torch
from torch import nn


class QCritic(nn.Module):
    """Small smooth Q critic; caller owns normalization and target construction."""
    def __init__(self, obs_dim: int, action_dim: int = 14, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim+action_dim,width),nn.SiLU(),
                                 nn.Linear(width,width),nn.SiLU(),nn.Linear(width,1))
    def forward(self, observation, action):
        return self.net(torch.cat([observation,action],dim=-1)).squeeze(-1)


class ValueCritic(nn.Module):
    def __init__(self, obs_dim: int, width: int = 64):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(obs_dim,width),nn.SiLU(),
                               nn.Linear(width,width),nn.SiLU(),nn.Linear(width,1))
    def forward(self, observation):
        return self.net(observation).squeeze(-1)


@contextmanager
def frozen_parameters(module):
    """Freeze critic weights without detaching observation/action derivatives."""
    parameters=list(module.parameters()) if isinstance(module,nn.Module) else []
    flags=[p.requires_grad for p in parameters]
    try:
        for p in parameters: p.requires_grad_(False)
        yield
    finally:
        for p,f in zip(parameters,flags): p.requires_grad_(f)


def paml_loss(critic: nn.Module, actor_fn: Callable, true_next: torch.Tensor,
              pred_next: torch.Tensor, actor_parameters: Iterable[torch.Tensor],
              *, reduction: str = 'mean') -> torch.Tensor:
    """Match actor-parameter gradients of -Q(s', pi(s')) on real/model states.

    The target branch uses detached observed next states. Critic weights are
    constants during this objective. The model branch is create_graph=True so
    gradients propagate through the induced actor gradient into model weights.
    Actor parameters are not stepped by this function. The caller freezes the
    critic and actor optimizer while updating the model.

    Additional observed context can be captured in actor_fn/critic wrappers;
    it must be identical in real and predicted branches. No simulator FD target
    enters this function. Per-batch gradient matching follows the official
    one-step path; it is not a full multi-step PAML reproduction.
    """
    if true_next.shape != pred_next.shape:
        raise ValueError('real and predicted next-state shapes must match')
    params=tuple(actor_parameters)
    if not params or any(not p.requires_grad for p in params):
        raise ValueError('actor_parameters must be nonempty differentiable tensors')
    if reduction not in ('mean','sum'):
        raise ValueError('reduction must be mean or sum')
    with frozen_parameters(critic):
        real=true_next.detach()
        j_real=-critic(real,actor_fn(real)).mean()
        j_model=-critic(pred_next,actor_fn(pred_next)).mean()
        if not j_real.requires_grad or not j_model.requires_grad:
            raise ValueError('critic objective does not depend on actor parameters')
        real_grad=torch.autograd.grad(j_real,params,allow_unused=True)
        model_grad=torch.autograd.grad(j_model,params,create_graph=True,allow_unused=True)
        chunks=[]
        used=False
        for p,gr,gm in zip(params,real_grad,model_grad):
            used |= gr is not None or gm is not None
            gr=torch.zeros_like(p) if gr is None else gr.detach()
            gm=torch.zeros_like(p) if gm is None else gm
            chunks.append((gm-gr).reshape(-1))
        if not used:
            raise ValueError('actor callback does not use provided actor_parameters')
        discrepancy=torch.cat(chunks)
        result=discrepancy.square().mean() if reduction=='mean' else discrepancy.square().sum()
        # Preserve a valid model graph also when gradients happen to be constant.
        return result + pred_next.sum()*0.0


def vagram_loss(value_fn: Callable, true_next: torch.Tensor,
                pred_next: torch.Tensor, *, state_scale=None,
                normalize_weights: bool = False, weight_floor: float = 0.0,
                epsilon: float = 1e-12) -> torch.Tensor:
    """Coordinatewise value-gradient squared weighting, NOT squared dot product.

    The raw loss is mean[(dV/ds_j * (pred_j-real_j))^2]. If state_scale is
    supplied, both gradients and errors are transformed consistently to the
    normalized coordinate, so the raw physical loss remains invariant.
    Weight normalization/floor are optional explicit adaptations, default off;
    they must be logged rather than called original VaGraM settings.
    """
    if true_next.shape != pred_next.shape:
        raise ValueError('real/predicted shapes differ')
    if weight_floor < 0 or epsilon <= 0:
        raise ValueError('invalid floor/epsilon')
    real=true_next.detach().clone().requires_grad_(True)
    with frozen_parameters(value_fn):
        values=value_fn(real)
        if not values.requires_grad:
            raise ValueError('value_fn must depend differentiably on state')
        gradient=torch.autograd.grad(values.sum(),real,allow_unused=True)[0]
    if gradient is None:
        raise ValueError('value_fn does not use supplied state')
    gradient=gradient.detach()
    scale=torch.ones_like(real) if state_scale is None else torch.as_tensor(state_scale,dtype=real.dtype,device=real.device)
    if not torch.isfinite(scale).all() or bool((scale<=0).any()):
        raise ValueError('state_scale must be positive and finite')
    error=(pred_next-real.detach())/scale
    weights=(gradient*scale).square()
    if normalize_weights:
        weights=weights/weights.mean(dim=-1,keepdim=True).clamp_min(epsilon)
    if weight_floor:
        weights=weights.clamp_min(weight_floor)
    return (weights.detach()*error.square()).mean()


def critic_regression_loss(critic, observations, actions, targets):
    """Plain train-only return/TD target regression; target source is external."""
    prediction=critic(observations,actions)
    target=targets.detach().to(prediction)
    if prediction.shape != target.shape:
        raise ValueError('critic target shape differs')
    return (prediction-target).square().mean()
