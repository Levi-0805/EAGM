"""Privileged Controller-B command-stack reference, NOT a deployable world model.

Initialization requires actual FIFO, smoothing, rate-state and active parameters
from Controller-B.snapshot(). No physics state transition or learned head exists.
"""
from dataclasses import dataclass, replace
import torch


@dataclass(frozen=True)
class PrivilegedState:
    target: torch.Tensor
    filtered: torch.Tensor
    queue: torch.Tensor
    extra_queue: tuple
    u_prev: torch.Tensor
    u_smooth: torch.Tensor
    delay: int


@dataclass(frozen=True)
class PrivilegedExecution:
    lower: torch.Tensor
    upper: torch.Tensor
    alpha: float
    max_increment: float
    gain: float
    extra_lag: int
    rate_limit: float
    smooth_beta: float

    @classmethod
    def from_privileged_snapshot(cls, snapshot, lower, upper, *, dtype=torch.float64, device='cpu'):
        """Explicit oracle-only initialization; no inference from observables."""
        if 'controller_b' not in snapshot:
            raise ValueError('true Controller-B hidden-state snapshot is required')
        b = snapshot['controller_b']; a = b['active']
        tensor = lambda x: torch.as_tensor(x, dtype=dtype, device=device).clone().detach()
        lower, upper = tensor(lower), tensor(upper)
        if lower.shape != (14,) or upper.shape != (14,) or not torch.all(lower < upper):
            raise ValueError('expected ordered 14-joint limits')
        lag = int(a['extra_lag']); delay = int(snapshot['delay'])
        if lag < 0 or lag != a['extra_lag'] or delay not in (0,1,2):
            raise ValueError('invalid discrete execution delays')
        model = cls(lower, upper, float(snapshot['alpha']), float(snapshot['max_increment']),
                    float(a['gain']), lag, float(a['rate_limit']), float(a['smooth_beta']))
        state = PrivilegedState(tensor(snapshot['qtarget']), tensor(snapshot['filtered']),
                                tensor(snapshot['queue']), tuple(tensor(q) for q in b['extra_queue']),
                                tensor(b['u_prev']), tensor(b['u_smooth']), delay)
        if state.queue.shape != (2,14) or any(x.shape != (14,) for x in
                (state.target,state.filtered,state.u_prev,state.u_smooth,*state.extra_queue)):
            raise ValueError('malformed privileged execution state')
        return model, state

    def step(self, state, action, *, delay=None):
        """Pure differentiable step; branch choices are discrete fixed configs."""
        if action.shape != (14,) or not torch.isfinite(action).all():
            raise ValueError('action must be a finite 14-vector')
        d = state.delay if delay is None else delay
        if d not in (0,1,2): raise ValueError('delay must be 0, 1 or 2')
        applied = action.clamp(-self.max_increment,self.max_increment)
        target = torch.clamp(state.target+applied,self.lower,self.upper)
        filtered = self.alpha*target+(1-self.alpha)*state.filtered
        nominal = filtered if d==0 else state.queue[2-d]
        queue = torch.stack((state.queue[1],filtered))
        gained = nominal*self.gain
        if self.extra_lag == 0:
            extra = (); delayed = gained
        else:
            extra = (*state.extra_queue,gained)
            if len(extra)<=self.extra_lag: delayed = state.u_prev
            else: delayed,extra = extra[0],extra[1:]
        limited = state.u_prev+(delayed-state.u_prev).clamp(-self.rate_limit,self.rate_limit)
        smooth = state.u_smooth
        if self.smooth_beta>0:
            smooth = self.smooth_beta*limited+(1-self.smooth_beta)*smooth
            output = smooth
        else: output = limited
        executed = torch.clamp(output,self.lower,self.upper)
        new_state = PrivilegedState(target,filtered,queue,extra,executed,smooth,int(d))
        return new_state, dict(nominal=nominal,executed=executed,applied=applied,
                               gained=gained,delayed=delayed,rate_limited=limited)
