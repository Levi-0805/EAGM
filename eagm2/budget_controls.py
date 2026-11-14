"""Exploratory physical FD/SPSA controls with an inclusive training-step cap.

These are optimization controls, not reproductions of published algorithms.
Query means one mean-return evaluation across all development training starts.
The budget unit is simulator control steps, and final evaluation is separate.
"""
from pathlib import Path
import random
import time

import numpy as np
import torch

from eagm2.gradients import finite_difference, spsa_gradient
from eagm2.runtime import check_seed, dump, stream_seed


def optimize_query_budget(objective, dimension, query_cap, steps_per_query, *,
                          method, seed, max_updates=50, output=None):
    """Optimize using complete iterations only; objective must have fixed cost.

    FD stencil planning evaluates a constant dummy function, never the simulator.
    Planning therefore counts actual distinct queries including shared centers.
    SPSA uses eight predeclared independent directions per update. Its stencil
    is explicitly unconstrained (at most 0.01 beyond the parameter box); policy
    updates are projected. This is not a clipped finite-difference estimator.
    """
    if method not in ('direct_fd', 'spsa'):
        raise ValueError('unknown direct control')
    if dimension <= 0 or steps_per_query <= 0 or query_cap < 0:
        raise ValueError('invalid dimension or budget')
    if any(int(x) != x for x in (dimension, steps_per_query, query_cap, max_updates)):
        raise ValueError('budget dimensions must be integers')
    if not 0 <= max_updates <= 50:
        raise ValueError('at most 50 updates authorized')
    root = None if output is None else Path(output)
    if root is not None:
        root.mkdir(parents=True, exist_ok=False)
        (root/'checkpoints').mkdir()
    theta = torch.zeros(dimension, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=.03)
    ledger, query_log = [], []
    used = 0
    start_time = time.perf_counter()
    seeds = [stream_seed(seed, f'{method}:update:{i}') for i in range(max_updates)]
    def save(state, error=None):
        result = dict(method=method, phase='exploratory_development', theta=theta.detach().numpy().copy(),
                      updates=len(ledger), query_cap_control_steps=int(query_cap),
                      training_control_steps=used, unused_control_steps=int(query_cap)-used,
                      objective_queries=len(query_log), steps_per_objective_query=steps_per_query,
                      wall_seconds=time.perf_counter()-start_time, stop_reason=state,
                      ledger=ledger, error=error, iteration_seeds=seeds,
                      protocol=dict(lr=.03, max_updates=max_updates, epsilon=.01,
                                    fd_scales=[1.], spsa_directions=8,
                                    parameter_projection=[-1., 1.],
                                    spsa_stencil_bounds=None, test_selection=False,
                                    budget_note='Different update counts follow from equal simulator-step caps; no unused world-model data charged.'))
        if root is not None:
            dump(root/'optimization.json', result)
            dump(root/'query_ledger.json', query_log)
            checkpoint = dict(theta=theta.detach().clone(), optimizer=optimizer.state_dict(),
                            completed_updates=len(ledger), ledger=ledger,
                            training_control_steps=used, iteration_seeds=seeds,
                            python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                            torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
            torch.save(checkpoint, root/'checkpoint.pt')
            torch.save(checkpoint, root/'checkpoints'/f'update_{len(ledger):03d}.pt')
        return result
    save('initialized')
    try:
        for update in range(max_updates):
            t = theta.detach().numpy().copy()
            expected = (finite_difference(lambda _: 0., t, scales=(1.,))['queries']
                        if method == 'direct_fd' else 16)
            if used + expected*steps_per_query > query_cap:
                return save('insufficient_budget_for_complete_iteration')
            first_query = len(query_log)
            iteration_start = time.perf_counter()
            def guarded(x):
                nonlocal used
                if used + steps_per_query > query_cap:
                    raise RuntimeError('query budget guard rejected simulator evaluation')
                row = dict(update=update, theta=x.copy(), status='started',
                           reserved_control_steps=steps_per_query)
                query_log.append(row)
                # Reserve before execution: failed calls cannot free spent budget.
                used += steps_per_query
                begun = time.perf_counter()
                try:
                    value = float(objective(x))
                    if not np.isfinite(value):
                        raise ValueError('nonfinite physical objective')
                    row.update(value=value, status='completed')
                    return value
                finally:
                    row['wall_seconds'] = time.perf_counter()-begun
            estimate = (finite_difference(guarded, t, scales=(1.,)) if method == 'direct_fd'
                        else spsa_gradient(guarded, t, directions=8, seed=seeds[update], bounds=None))
            if len(query_log)-first_query != expected:
                raise RuntimeError('stencil query count disagrees with reservation')
            g = np.asarray(estimate['gradient'])
            if not np.isfinite(g).all():
                raise ValueError('nonfinite physical gradient')
            optimizer.zero_grad()
            theta.grad = -torch.as_tensor(g, dtype=theta.dtype)
            optimizer.step()
            with torch.no_grad():
                theta.clamp_(-1., 1.)
            ledger.append(dict(update=update, theta_before=t, theta_after=theta.detach().numpy().copy(),
                               gradient=g, objective_queries=expected,
                               control_steps=expected*steps_per_query,
                               cumulative_control_steps=used,
                               wall_seconds=time.perf_counter()-iteration_start))
            save('running')
        return save('max_updates_reached')
    except Exception as exc:
        save('failed', repr(exc))
        raise


def run_direct_controls(env, starts, test_starts, initial, seed, task, horizon,
                        eval_horizon, budget, out, query_cap, vision_policy=None):
    """Run both controls with independent equal caps and final-only test access."""
    check_seed(task, seed)
    from eagm2.rollout import physical
    from eagm2.evaluation import evaluate_returns
    if not starts or not test_starts or horizon <= 0 or eval_horizon <= 0:
        raise ValueError('nonempty starts and positive horizons required')
    train_ids = {s['episode_id'] for s in starts}
    if train_ids & {s['episode_id'] for s in test_starts}:
        raise ValueError('training/test episode leakage')
    results = {}
    for name in ('direct_fd', 'spsa'):
        kind = f'{name}_training'
        before = budget.steps.get(kind, 0)
        seconds_before = budget.seconds.get(kind, 0.)
        def objective(theta):
            return np.mean([physical(env, theta, s, horizon, budget, kind,
                                     task=task, vision_policy=vision_policy) for s in starts])
        root = Path(out)/name
        try:
            result = optimize_query_budget(objective, 32 if vision_policy is not None else 28,
                                           query_cap, len(starts)*horizon, method=name,
                                           seed=seed, output=root)
        finally:
            if root.exists():
                dump(root/'actual_simulator_accounting.json',
                     dict(training_control_steps=budget.steps.get(kind, 0)-before,
                          training_simulator_wall_seconds=budget.seconds.get(kind, 0)-seconds_before,
                          query_cap_control_steps=query_cap))
        actual = budget.steps.get(kind, 0)-before
        if actual != result['training_control_steps']:
            raise RuntimeError('physical cost differs from fixed-horizon query budget')
        result['actual_training_control_steps'] = actual
        result['physical'] = evaluate_returns(env, result['theta'], test_starts, eval_horizon,
                                             budget, task, kind=f'{name}_final_evaluation',
                                             vision_policy=vision_policy)
        result['delta_return_vs_initial'] = result['physical']['mean_return']-initial['mean_return']
        dump(root/'results.json', result)
        results[name] = result
    return results
