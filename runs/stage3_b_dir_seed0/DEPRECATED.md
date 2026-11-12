# DEPRECATED — Stage 3 exploratory (protocol bugs)

Do not cite as method evidence. Known issues:
1. `u_true` was read **before** `env.step(a)` → wrong command timing.
2. Transition-level random train/test split (leakage across adjacent frames).
3. `imagined_eagm` reward ≠ physical reward (missing vel/action-smooth terms and 0.99 discount).
4. EAGMBase had **no** history encoder despite EAGM naming.
5. Mu used for unfair comparisons; Mc lacks Controller-B hidden state.

Superseded by `runs/stage3_1_seed0_*` after protocol repair.
