# Privileged execution-stack reference: definition and limitations

This extension implements a known Controller-B **command stack**, initialized from its true hidden execution state. It is not a deployable EAGM input path, a learned dynamics model, or a complete privileged world-model baseline. No training or policy-performance experiment is included.

## Audited source and snapshot contract

- `controllers/partial_unknown.py:160`: action clipping, target update, nominal filter, declared delay, FIFO update.
- `controllers/partial_unknown.py:199`: gain precedes extra FIFO, followed by rate limit, extra smoothing, joint clipping.
- `controllers/partial_unknown.py:228`: actual hidden fields `active`, `extra_queue`, `u_prev`, `u_smooth`.
- `vendor/legacy_sim/sim_env.py:214`: named snapshot fields `qtarget`, `filtered`, `queue`, `delay`, `alpha`, `max_increment`. The nominal queue is a two-by-fourteen array. No packed-vector indices are guessed.

The explicit constructor `PrivilegedExecution.from_privileged_snapshot(snapshot, lower, upper)` rejects snapshots missing `controller_b`; it copies and detaches initial tensors. It reads active episode-specific parameters rather than nominal configuration defaults. It does not read the simulator integration vector. Joint limits must be supplied from the same environment. The returned state is immutable at the dataclass level and steps construct new tensors rather than changing snapshot contents.

## Exact recurrence

Let the pre-step target and filter be `r,f`, nominal queue be `[q0,q1]`, and the action bound be `m`. First:

```
a_bar = clip(a, -m, m)
r_next = clip(r + a_bar, lower, upper)
f_next = alpha * r_next + (1 - alpha) * f
nominal = f_next if d=0 else queue[2-d]
queue_next = [q1, f_next]
```

The nominal queue read occurs **before** its shift. Full queue initialization supports alpha=1 without any inverse-filter reconstruction. Declared delay may change among 0,1,2 without resetting the queue.

Then `gained = gain * nominal`. With extra lag zero, its queue is cleared and `delayed = gained`. With positive extra lag L, append `gained`; while the queue length is at most L, return the previous executed command as `delayed`; otherwise pop its oldest element. This reproduces the startup transient and retains already-gained past commands.

```
limited = u_prev + clip(delayed - u_prev, -rate_limit, rate_limit)
if beta > 0:
    u_smooth_next = beta * limited + (1 - beta) * u_smooth
    output = u_smooth_next
else:
    u_smooth_next = u_smooth
    output = limited
executed = clip(output, lower, upper)
u_prev_next = executed
```

The meaning of beta is the actual code's new-input weight; beta=0 bypasses smoothing rather than freezing the output. Smoothing state is stored before final joint clipping, while `u_prev` stores the clipped executed command. These distinct states must not be conflated.

## Validation design

`test_privileged_execution.py` runs CPU MuJoCo and the actual Controller-B `step` method. The value test crosses alpha {0.2,0.5,1}, extra lag {0,1,2}, gain {0.9,1.1}, rate limit {0.01,0.08}, and beta {0,0.25,0.8}. Declared delay cycles 0,1,2 over twelve steps for each configuration, including actual action clipping. Every intermediate execution state and both nominal/executed commands are compared. At step six the differentiable reference is reinitialized from the actual midtrajectory snapshot.

The derivative test compares autograd from the first action to the final executed command against central finite differences of the **real** controller, restoring the same true snapshot for every query. Its horizon d+extra_lag+3 allows delayed influence to arrive. It crosses alpha {0.2,0.5,1}, d {0,1,2}, extra lag {0,1,2}, beta {0,0.25}; gain=1 and rate limit=0.08 keep the small action perturbations off clipping boundaries. FD radius is 1e-6. This verifies multi-step command Jacobians, not physics-state or policy-return Jacobians.

All three numerical tests passed in 42.507 seconds. Across 108 configurations and 1,296 real controller steps, nominal/executed-command maximum absolute discrepancy was exactly zero at float64 precision, with all internal-state comparisons passing. All 54 multi-step gradient probes were nonzero and agreed with real-controller finite differences; maximum absolute Jacobian discrepancy was 7.627137810217732e-11. These are command-stack verification results only.

Exact test outcomes are in `privileged_execution_validation.json` and `privileged_execution_tests.log`. Test RNG values 880301–880303 are numerical validation fixtures, not research training/confirmation roots.

## Scope restrictions

- Inputs explicitly contain deployment-unavailable true hidden states and active parameters; this interface must remain separately labeled privileged/oracle-only.
- The reference does not propagate robot physics, visual observations, contact state, reward, or policy parameters. No trained oracle return or world-model superiority is established.
- Differentiation is with respect to continuous actions and retained state tensors. Declared delay and extra-lag branches are discrete; no derivative with respect to lag is defined.
- Clamps are piecewise differentiable. Tests make no differentiability claim exactly at action, target, rate, or joint-boundary kinks.
- Active gain/rate/smoothing/extra-lag are fixed for a reference instance. An external controller configuration reset must be followed by fresh privileged initialization; silently changing only parameters would fail to reproduce the reset's state changes.
- Current implementation is one 14-joint execution trajectory, not a batched trainable module. GPU correctness/performance is untested.
- This extension was kept outside the frozen development implementation, configuration, and test trees. It neither alters the running development_v1 experiment nor launches training.
