# Stage 8 paper outline

1. Problem and scope: partial execution dynamics in observable software state; distinguish commanded action from executed target. Figure: controller/state/command dependencies. Specify simulator-reset access and command-label availability.
2. Observation: state prediction accuracy does not ensure policy gradient quality. Show matched models, fixed objectives and full/stable metrics; use historical gradient/return figure and Reach failure as a boundary case.
3. Method: U0 plus learned command residual, matched-θ three-scale masked normalized Huber gradient supervision. Clearly acknowledge magnitude supervision. Add a reliability extension only if experiments support it.
4. Analysis: chain-rule decomposition of dynamics Jacobian error, reward Jacobian error and trajectory mismatch. State smoothness assumptions; separate clipping boundaries. No universal monotonicity claim.
5. Evaluation protocol: paired initialization/minibatches, episode splits, seed-level inference, control-step plus actual compute accounting, all-start physical returns. Include frozen protocol table and artifact provenance.
6. Results: Stage4 exploratory; Stage5A independent confirmation; Stage5B budget; Stage5C architecture; Stage5E/6B1 interpolation/extrapolation separately. Show every seed and LOO sensitivity.
7. Ablations and limitations: HistC, Path, saturation; new method validation only if it passes; negative Reach True-EE audit. Required future geometry-Jacobian intervention before any transfer claim. Do not hide failed extensions.
8. Scope: low-dimensional fixed controller family, short horizon, simulator resets, no hardware/vision validation. External baselines have not been run; no SOTA statement.

Submission readiness: core method can be drafted now as a narrow study, but a high-level method claim requires stronger novelty against PAML/GAMPS/AWM and a robust true-objective task. New-stage labels do not themselves make a contribution.
