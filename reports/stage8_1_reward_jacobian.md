# Reward Jacobian decomposition

Exploratory seed20 frozen-model causal diagnostic. No new world-model training, no independent confirmation. Sources: runs/stage8_1_geometry/; script analysis/stage8_1_reports.py.

Actual reward: −mean(EE_error²) −0.0005 mean(qdot²) −0.05 mean((a−a_prev)²), discount 0.99. Position gradient is −Jᵀ(p−goal)/3 with J∈R^(6×14), W=I/6. There is **no EE velocity reward**, so no omitted derivative of J(q) qdot. qdot derivative=−0.001 qdot/14; action derivative=−0.1(a−a_prev)/14; previous action has opposite derivative and remains in the BPTT graph. Fixed horizon, unchanged control_dt and no new termination rule.

| Distribution | native reward grad cos | true J / FK residual cos | FK J / true residual cos |
|---|---:|---:|---:|
| old_final_imagined | 0.84207 | 0.83052 | 0.95781 |
| old_final_physical | 0.82754 | 0.81967 | 0.95502 |
| random_legal | 0.90829 | 0.99356 | 0.90398 |
| test | 0.44101 | 0.40947 | 0.95719 |
| theta0_imagined | 0.84370 | 0.83237 | 0.95801 |
| theta0_physical | 0.81812 | 0.80929 | 0.95465 |
| train | 0.48272 | 0.48301 | 0.92855 |

Reward-value errors and goals are stored per sample in static.json. This factorial decomposition changes J and positional residual independently at the state level. Different goals come from original episodes; random configurations cycle held-out goals and do not constitute independent task seeds. state derivative identities checked in reward_derivative_checks.json. No causal claim follows solely from these correlations.

Total added physical control steps including actual-state coverage: **19488/20000**. Main and supplemental raw budgets are preserved separately; combined accounting is total_budget.json. Static metrics CSV, component/goal/configuration summaries and SHA-256 inventory are in the run directory. No prior file changed relative to Stage8.1 preservation manifest.
