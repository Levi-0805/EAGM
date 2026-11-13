# Stage 8 research decision

P0 completed as a bounded audit: preserved prior workspace, independently aggregated raw CSVs with provenance, seed/bootstrap/config sensitivity, evidence ledger and targeted Reach numerical/geometry/dynamics/optimization diagnosis. Full independent historical retraining was not performed; past overwrites before the initial snapshot cannot be excluded.

P1 executed: three mechanisms designed; C1 implemented and trained on two fresh engineering seeds; C2 equal-capacity linear feature screen; C3 train-label scalar calibration feasibility. C1 seed41 fails joint gradient+return gate. No five-seed, ten-seed, visual or contact expansion launched. No core claim changed.

Strongest supported finding remains Stage4–6 tracking Dir versus Base, with limited task/controller coverage and n=5 inference. Stage6C Base batch mismatch weakens that specific contrast. Reach failure is not explained by tested FD epsilon instability. Reward geometry and Jacobian mismatch remain important unresolved mechanisms, not a uniquely established causal explanation.

Recommended next controlled intervention: exact differentiable robot FK or a verified MuJoCo Jacobian-backed reward derivative, tested against current-state FK and directional finite differences on train/validation poses; then fixed-data frozen-model gradient comparison before retraining. This can distinguish reward-Jacobian error from dynamics-Jacobian error. Estimate: 0 control steps for FK/Jacobian validation; <=20k control steps for frozen-policy comparisons, CPU minutes, no large GPU. Any retraining should retain Base/Dir seed pairing and fresh validation episodes. This cycle has not implemented that interface or claimed a successful rescue.

Budget proposal only if a future method passes two-seed gate: five seeds approximately 5×110s CPU for this small four-model validation setup, ~160,800 collected/label/evaluation control steps; full OOD and diagnostic grids add cost. Independent 10-seed confirmation requires a frozen candidate and human approval. Do not extrapolate tiny-network runtime to visual agents.

Literature screening highlights GAMPS/PAML and action-conditioned world models as direct novelty challenges; recent execution-interface work also narrows broad claims. Equation-level comparisons and external baseline adaptation remain open. Venue assessment favors robot-learning venues for current scope; CVPR needs a real perception contribution, not an appended image encoder.
