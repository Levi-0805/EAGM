# Primary-source novelty review (2026-11-14)

This is a targeted review, not a completed novelty certification. No external method was run.

- **PAML:** [paper, pp.6–7, Eqs.(7)–(9)](https://arxiv.org/pdf/2003.00030). It directly defines a policy-gradient discrepancy; the empirical construction compares real/model trajectories with a critic learned from real transitions. EAGM's masked finite-difference target and command bottleneck differ in interface/estimator. Generic gradient-matching novelty is unavailable. A matched adaptation remains necessary.
- **GAMPS:** [Section4.1, Eq.(3)](https://arxiv.org/html/1909.04115v2). Weighted transition likelihood includes discount, importance ratio and accumulated score norms. EAGM instead penalizes normalized policy-gradient coordinate discrepancies. Different objectives do not establish superiority; same-interface experiments remain missing.
- **AWM:** [ICML primary source](https://proceedings.mlr.press/v235/ma24i.html). Action-conditioned trajectory modeling changes gradient paths. Relevant architecture competitor; no EAGM implementation/result yet.
- **VaGraM:** [primary paper](https://arxiv.org/abs/2204.01464). Value-gradient model-error weighting is a distinct decision-aware objective. Scope reviewed at abstract level here; no exhaustive equivalence analysis.
- **Execution interfaces:** [Mind the Execution Gap](https://arxiv.org/abs/2610.06582). Future-sequence and applied-action interfaces overlap the broad execution motivation. Exact algorithm/code comparison remains open.
- **SuperMimic:** [primary paper](https://arxiv.org/abs/2610.28480). Learned differentiable world model and short imagined BPTT for tracking. Controller-input and actuator-map code audit remains pending. No claim that it lacks execution-state access; no copied performance numbers presented as EAGM benchmarks.

Submission-critical: characterize overlap with the already submitted diagnostic, compare objective functions on matched command inputs/budgets, and review recent implementation details before asserting an algorithmic novelty boundary. CoRL/ICRA remain plausible subject fits; CVPR requires a genuine visual research contribution, not a nominal image input. No deadlines or current venue rules are asserted in this draft.
