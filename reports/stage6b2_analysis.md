# Stage 6B.2 — Analysis

**Decision: DROP_HISTC**

HistC may look slightly better than Hist+Dir on average, but it fails capacity-matched and shuffled-c_obs controls (n_pos≤1/5). Extra parameters / feature volume explain the gap, not temporal execution-state estimation. Do NOT start ExecState (6B.3). Prefer Stage 6C (saturation / path-aware Dir) after human review.

## What would justify ExecState (6B.3)?

- HistC beats Hist+Dir **and** capacity/shuffle controls on paired seed returns.
- Probe shows HistC encodes privileged scalars better than Hist / shuffle.
- Residual errors remain that a latent z_t could explain beyond c_obs history.

If HistC fails controls, drop it — do not invent ExecState to decorate the paper.

## Saturation reminder

θ-sat remains high for Dir-family models; Stage 6C stays deferred until HistC decision is clear.

