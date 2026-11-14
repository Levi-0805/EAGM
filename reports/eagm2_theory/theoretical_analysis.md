# EAGM 2.0: conditional theory and implementation correspondence

These are elementary, auditable conditional derivations, not a claim of a new theorem, convergence, certified physical improvement, or proved GRU state sufficiency. They extend `reports/eagm2_plan/03_mathematical_derivation.tex` against the actual frozen implementation. No training or paper files were changed.

## Causal execution

Before each call let c=(r,f,d), b=previous observed filter. The real controller first clips the action, clips the incremented target, computes f⁺=αr⁺+(1−α)f, selects nominal v=f⁺/f/b for d=0/1/2, then advances FIFO [b,f] to [f,f⁺]. Public cache b⁺=f is valid also at α=1. Algebraic inversion b=(f−αr)/(1−α) requires the preceding update invariant and α≠1. Delay switching does not change FIFO advancement, but the imagined code holds delay fixed and does not anticipate external switching.

On an open fixed clipping regime let Da and Dr be diagonal unsaturated-coordinate masks. Holding the pre-call state fixed, ∂v/∂a=1[d=0]αDrDa. Proof: direct differentiation; positive delays select old values. This is not the total derivative with respect to policy parameters. Writing R,F,B,A for their total sensitivities gives R⁺=Dr(R+DaA), F⁺=αR⁺+(1−α)F, B⁺=F and V=1[d=0]F⁺+1[d=1]F+1[d=2]B. Positive-delay policy sensitivities can therefore be nonzero through history.

Controller B subsequently applies gain, its independent extra queue, rate limit, smoothing and final clipping. If W is sensitivity of the extra-queue-selected command, Uprev is previous executed-command sensitivity and Dρ is the rate mask, limited-command sensitivity is Λ=DρW+(I−Dρ)Uprev. For β>0, smoothing sensitivity is S=βΛ+(1−β)Sprev and U=DuS. For β≤0 the code bypasses smoothing, U=DuΛ: the branch at β=0 is not the positive-β recurrence's continuous limit. After queue fill and at constant gain/lag, Wt=gV(t−lag); startup uses the previous executed command. The standalone nominal prior does not implement these hidden sensitivities. The residual/GRU must learn whatever it can from public history.

Code correspondence: `controllers/partial_unknown.py:step,_apply_unknown`; `eagm2/prior.py:observable_step,nominal_command`; `eagm2/models.py:burnin,observe,step`. The posterior consumes actual observations during burn-in, whereas imagination advances the learned transition without future physical observations. A latent vector is not by construction a complete execution state.

## Why forward fit does not bound gradient error

Take F(x)=0 and Fhatε(x)=ε sin(x/ε²). Uniform forward error is ≤ε, but derivative error at x=0 is 1/ε. Both functions are smooth. Hence forward prediction accuracy alone does not control Jacobian accuracy; a finite-sample NMSE cannot imply a uniform derivative bound either. This counterexample establishes a non-implication, not actual network oscillation.

## Closed-loop gradient bound with initial estimation error

Assume a common complete state coordinate z and the same differentiable policy πθ, with true/model maps Tθ(z)=F(z,πθ(z)) and Thatθ(z)=Fhat(z,πθ(z)). A lower-dimensional latent model needs a justified differentiable lifting to use this statement; none is established by the present training. All constants below must hold uniformly on a neighborhood covering both trajectories and connecting segments, not merely a few logged starts.

Let true/model transition derivative bounds be Lz, La; policy derivative bounds be Pz, Pθ. Let same-input transition/Jacobian discrepancies be εF, εz, εa. Let true Fz, Fa be Lipschitz in the joint norm ||Δz||+||Δa|| with constants Kz, Ka, and let πz, πθ be Lipschitz in z with constants Qz, Qθ. Define

\[
L=L_z+L_aP_z,\quad B=L_aP_\theta,
\quad \epsilon_A=\epsilon_z+\epsilon_aP_z,
\quad \epsilon_B=\epsilon_aP_\theta,
\]
\[
K_A=(K_z+K_aP_z)(1+P_z)+L_aQ_z,
\quad K_B=K_a(1+P_z)P_\theta+L_aQ_\theta.
\]

These follow by the chain rule Tz=Fz+Faπz, Tθ=Faπθ and adding/subtracting a common-factor product. The policy increases joint state-action distance by at most 1+Pz.

Set et=||zhat−z||, St=∂zt/∂θ, st=||St||, Et=||Shat−S||. Initial history estimation can give e0>0 and E0>0. Then

\[
e_{t+1}\le Le_t+\epsilon_F,\quad s_{t+1}\le Ls_t+B,
\]
\[
E_{t+1}\le LE_t+(\epsilon_A+K_Ae_t)s_t+\epsilon_B+K_Be_t.
\]

Proof: add and subtract the true map at the predicted state to bound e. Sensitivities obey S⁺=AS+B. Their difference equals Ahat(Shat−S)+(Ahat−A)S+(Bhat−B). Same-input derivative errors plus state variation bound the latter differences by εA+KAe and εB+KBe. Take operator norms and apply the triangle inequality. Induction gives

\[
e_t\le L^te_0+\epsilon_F\sum_{i=0}^{t-1}L^i,\quad
s_t\le L^ts_0+B\sum_{i=0}^{t-1}L^i,
\]
\[
E_t\le L^tE_0+\sum_{k=0}^{t-1}L^{t-1-k}
[(\epsilon_A+K_Ae_k)s_k+\epsilon_B+K_Be_k].
\]

When optimizing from a fixed saved start/history, the initial state does not depend on the currently optimized theta, so s0=E0=0. This does not eliminate hidden-state estimation error e0. The burn-in detach implements this conditional-start interpretation, not a derivative of a theta-dependent history/data-collection distribution.

For an augmented per-step reward R(z,a), which may include next-state reward through F, assume true/model reward derivative bounds Rz*, Ra*, same-input derivative errors εRz, εRa, and true joint derivative Lipschitz constants KRz, KRa. Set

\[
C=R_z^*+R_a^*P_z,\quad\epsilon_C=\epsilon_{Rz}+\epsilon_{Ra}P_z,
\quad\epsilon_D=\epsilon_{Ra}P_\theta,
\]
\[
K_C=(K_{Rz}+K_{Ra}P_z)(1+P_z)+R_a^*Q_z,
\quad K_D=K_{Ra}(1+P_z)P_\theta+R_a^*Q_\theta.
\]

For horizon H and discount γ∈[0,1],

\[
\|\nabla\hat J-\nabla J\|\le\sum_{t=0}^{H-1}\gamma^t
[CE_t+(\epsilon_C+K_Ce_t)s_t+\epsilon_D+K_De_t].
\]

Proof: each exact reward-gradient contribution is CtSt+Dt, where Ct=∂zrθ and Dt=∂θrθ. Subtract modeled and true contributions and add/subtract ChatSt. Bound the sensitivity term by CEt; the remaining terms follow from the same product-difference argument. Sum the discounted bounds. Learned reward derivative errors cannot be omitted. Analytic reward composed with an imperfect next-state prediction can still have derivative discrepancy. If initial-state distributions differ, an additional distribution-gradient error is needed. L>1 can make the bound uninformative; constants have not been certified on these simulations.

## Finite differences and actual implemented stencils

Let f(s)=J(θ+sej) be C³ over the entire feasible stencil, with |f'''|≤M3 and objective evaluation errors bounded by η. Then:

- Symmetric ±h: error ≤M3h²/6+η/h.
- Unequal offsets −a,0,b: weights −b/[a(a+b)], (b−a)/(ab), a/[b(a+b)] give error ≤M3ab/6+ηΣ|w|.
- Exact lower boundary: weights (−3,4,−1)/(2h) at 0,h,2h give error ≤M3h²/3+4η/h. Upper boundary follows by reflection.

Proof for central/nonuniform cases: weighted constants and quadratics cancel, weighted linear coefficient is one. Absolute cubic remainders sum to M3(ba²+ab²)/[6(a+b)]=M3ab/6. Noise contributes η times the absolute weight sum. For one-sided differences the integral Taylor remainder kernel is [4(h−s)²−(2h−s)²]/(4h) on [0,h], and −(2h−s)²/(4h) on [h,2h]. Both are nonpositive, and their combined integral is −h²/3; therefore its absolute contribution is ≤M3h²/3. Noise weight sum is 4/h.

`eagm2/gradients.py:finite_difference` implements feasible stencils, reduces one-sided h to fit the interval, counts cached center queries, selects the scale nearest1 and uses spread as a stability mask. The mask is not a confidence interval. At a clipping/contact kink, C³ fails. Agreement across scales does not prove the missing regularity. Stochastic objective error needs a separate noise model before a probability statement is possible.

## Policy updates and contact limitations

If the evaluation objective has LJ-Lipschitz gradient on the actual update segment and ||ghat−∇J||≤δ, then

\[
J(\theta+d)-J(\theta)\ge\hat g^Td-\delta\|d\|-L_J\|d\|^2/2.
\]

Proof: integrate the gradient on the segment, use its Lipschitz bound for the quadratic remainder, and bound (∇J−ghat)ᵀd by Cauchy–Schwarz. The actual Adam/projected step d must be used. Cosine alone supplies neither δ nor a safe magnitude. Discrete success/completion-time metrics are not differentiable rewards.

A smooth contact branch is covered only if the actual discrete simulator map is differentiable there and the considered trajectories/perturbations satisfy the required neighborhood assumptions. Contact-event changes, solver branch changes, saturation ties, unknown integer delay changes, and active-set changes are not automatically covered. A formal hybrid-system analysis would need additional event assumptions and sensitivities; none is claimed here.

## Verification and outstanding work

Seven independent arithmetic checks in `check_derivations.py` passed: quadratic reproduction and tight cubic bounds for three nonuniform and three one-sided stencils, plus an explicit closed-loop example with initial estimation error. `derivation_checks.json` records results. These check algebra, not physical regularity. The source controller tests remain separate empirical evidence.

Outstanding scientific obligations: prove or measure a suitable state representation/lifting, estimate useful Jacobian/reward-error envelopes, inspect contact-switch sensitivity and FD-radius dependence, and independently review proofs. No new theorem novelty, certified GRU sufficiency, contact-wide differentiability, or acceptance readiness is asserted.

Both PDFs compiled successfully using locally cached fonts: causal derivation2 pages, gradient bound3 pages. Initial missing-font/network errors remain in build logs; final successful logs are `causal_build_cached.log` and `gradient_build_t1.log`.
