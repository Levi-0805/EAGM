# Proof and assumption review checklist

| Statement | Proof status | Code correspondence | Outstanding applicability obligation |
|---|---|---|---|
| Nominal FIFO d0/1/2 | Direct update identity | prior.observable_step/nominal_command; ControllerB.step | Valid public cache/reset; only declared delays0/1/2 |
| Immediate versus total Jacobian | Chain rule, clipping masks | ControllerB queue selection before advancement | No ordinary derivative at clipping ties; no differentiation of discrete delay |
| Hidden B stack sensitivities | Explicit rate/smoothing/lag recurrence | ControllerB._apply_unknown | Startup queue and β=0 bypass must remain distinct; fixed execution parameters |
| Forward-fit non-implication | Smooth counterexample supplied | Motivates separate gradient diagnostics | Not empirical evidence of network oscillation |
| State error bound | Add/subtract map and induction | Imagined rollouts | Uniform transition error; common coordinates; valid neighborhoods |
| Sensitivity error bound | Recurrence subtraction and operator norms | model.step plus observable_step | State alignment/lifting for GRU remains unproved; derivative envelopes unknown |
| Initial state estimate | e0 and E0 explicitly retained | observed burn-in, detached during optimization | Frozen-start objective only; theta-dependent histories require their derivatives |
| Reward-gradient bound | Product decomposition and sum | rollout.imagined reward after predicted next state | Reward derivative errors and policy regularity; learned visual reward is not exact |
| FD central/nonuniform | Taylor moments and remainder | gradients.finite_difference | C³ over whole stencil, uniform M3, bounded evaluation error |
| FD exact boundary | Signed integral kernel | forward/backward three-point modes | Must retain two feasible distinct points; no kink theorem |
| Update lower bound | Smoothness integral + Cauchy–Schwarz | Adam/projection actual displacement | Certified δ,LJ absent; different start distributions add error |
| Contact/vision extension | Explicitly conditional, not proved | physical simulator; latent encoder/reward | Contact switching, image/latent sufficiency and reward errors unestablished |

Artifacts:

- `causal_jacobian_derivation.tex`: controller timing and execution Jacobians.
- `gradient_error_bound.tex`: full constants, recurrences, rewards, FD proofs and update inequality.
- `theoretical_analysis.md`: standalone readable proofs and scope.
- `check_derivations.py`, `derivation_checks.json`: seven arithmetic checks, all passing; not simulator validation.

Changes relative to proposal: distinguish actual β=0 smoothing bypass; include explicit initial state/sensitivity errors and state-distribution propagation; expose transition, policy and reward constants; include implemented exact-boundary FD stencil; retain the common-coordinate limitation. Recurrent training and low prediction error do not discharge these assumptions.

Before any paper theorem claim: another reviewer should independently inspect each proof and the lifting limitation; empirical sections must not describe conditional bounds as a numerical guarantee. No original-paper or frozen training code was edited.

Compilation completed locally: `causal_jacobian_derivation.pdf` 2 pages; `gradient_error_bound.pdf` 3 pages. Initial builds failed on uncached font downloads; genuine failure logs are retained (`causal_build.log`, `gradient_build.log`, `causal_build_t1.log`, `causal_build_final.log`). Using cached T1 fonts and equivalent indicator notation avoided missing font dependencies. Successful logs are `causal_build_cached.log` and `gradient_build_t1.log`. No network dependency was downloaded.
