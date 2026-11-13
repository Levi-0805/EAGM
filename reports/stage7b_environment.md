# Stage 7B — Reach Environment

**Status:** Unit tests PASS (`tests/test_stage7_reach.py`)  
**Code:** `tasks/reach.py`, `configs/stage7_reach.json`  
**Rollout hook:** `training/rollout.py` dispatches on `start["task_id"]=="reach"`

## Design

| Choice | Detail |
|--------|--------|
| Robot / control | Same OpenArm pilot effective model + Controller B |
| EE sites | `left_ee_control_point`, `right_ee_control_point` |
| World model | **Unchanged** — predicts proprioceptive x∈R²⁸ |
| Actor | **Unchanged** — joint-ref PD-style with θ∈R²⁸ |
| Task change | Reward + goal/ref generation only |

### Reward

\[
R = -\|e_{\mathrm{ee}}\|_2^2/6 - 5\cdot10^{-4}\|\dot q\|^2/14 - 0.05\|a-a_{\mathrm{prev}}\|^2/14
\]

with \(e_{\mathrm{ee}}\in\mathbb{R}^6\) stacking left/right EE position errors. Discount 0.99.

### Matched imagined ↔ physical

Both use a **frozen MuJoCo-fitted FK MLP surrogate** (`FKSurrogate`) so Dir FD labels and imagined returns share one EE map. Physical eval also logs secondary `true_mujoco_ee_return` (true sites) for diagnostics — **not** used as the primary Dir/policy metric.

### Joint refs for the actor

Solve damped least-squares IK **once** toward EE goals, then linearly blend
`q0 → q*` over `ref_blend_horizon` and hold (optional sine perturbation for
`traj_shape`). Per-frame Jacobian IK was avoided for collection cost.

## Planned ablations (config keys)

| ID | Variation |
|----|-----------|
| B1 | Fixed EE goals |
| B2 | Random EE goals (`goal_noise`) |
| B3 | Larger init joint jitter |
| B4 | Sine-perturbed joint refs |
| B5 | Controller B episode randomization |

## Tests covered

- EE sites exist and move under action  
- FK-only helper restores full MjData  
- FK surrogate RMSE gate  
- Episode collect + imagined grad + physical return + true EE metric  
- Reward monotonicity in EE error  
- Fixed vs random goal sampling  

## Explicit non-goals (this stage)

- No Pick-and-Place / contact  
- No vision  
- No architecture changes  
- No 10-seed confirmatory (awaits 7C smoke + human OK for 7D)

## Next gate (7C)

Single-seed smoke: Mc / Base / Dir-Random / Aux under matched budget on Reach. Proceed to 7D only after human review.
