# Phase-start and strict legacy audit

## Contact/vision training coverage correction

`eagm2/data.py` previously selected only frame32 for every policy-training start, although snapshots32/64/96 already existed. Training contact/vision starts now rotate eligible saved snapshots with ordinal `(episode + episode//3) % number_of_legal_snapshots`. With nine episodes this crosses all three delays with all three phases, avoiding complete phase/delay confounding from naive episode-modulo-three selection. Validation/development-test starts remain32, as do all tracking starts. Each start's observable history contains exactly frames t−31…t; actions are previous actions at those frames. Snapshot, reference offset and fprev follow the selected frame. Short trajectories only select snapshots actually saved with at least32 future steps; no synthetic snapshot is fabricated.

Ten tests passed in `tests_phase_starts.log`: phase/delay crossing, observable-history and snapshot alignment, unchanged heldout protocol, short-trajectory fallback, tracking protocol, and six extra-data regression tests. These tests verify data selection/causality, not contact success or improved policy performance. Prior runs with frame32-only policy starts remain evidence for the earlier implementation and must not be silently merged with new attempts.

## Original-protocol reproduction

- Root200001: original test episodes `[0,3,4,6,15,19]` contain no episode with delay2 (episode index modulo3 equals2). Original `pick_starts` consequently yields an empty policy-test pool. Saved results label completion but policy returns, baseline, delta and gradient cosine are null, with empty per-start lists. One-step test dynamics/command values remain numerically defined; they do not establish policy improvement. Preserve original artifacts unchanged and classify policy/gradient evaluation invalid.
- Root200002: original test episodes `[4,5,10,17,19,23]` contain three delay2 episodes. Base cosine0.5657970, Dir0.9630567; mean returns Base−0.001957767, Dir−0.001659265; shared theta0−0.003141852. These are exploratory single-root results, not confirmation or statistical superiority.
- Root200003: original train episodes `[0,1,2,5,6,7,10,14,15,17,22,23]` contain six delay2 episodes; test `[3,8,11,13,16,20]` contains three. Empty-pool guard should pass. The existing runner can execute a fresh immutable original-protocol attempt with3000 world updates and50 policy updates. No training launched by this audit; runtime success is not implied by split feasibility.

## Independent holdout for root200001

`scripts/eagm2/evaluate_legacy_holdout.py` freezes existing final policy parameters before generating an independent named holdout substream, evaluates30 paired starts (10 per delay) with the original ControllerB configuration and original horizon16, and reports all-delay and delay2 strata separately. Zero training and zero policy optimization. Source SHA256 and unchanged-content assertion protect the original results. This is a new evaluation distribution and cannot replace original-protocol results, be combined with root200002's old evaluation, or be called confirmation. Output uses a new immutable attempt `legacy_fresh_holdout_v1`. No gradient-fidelity claim is inferred from these return-only evaluations.

Collection now records completed simulator steps in a `finally` block on every episode, including exceptions. The extra-data cap wrapper separately records issued-but-failed calls, without double-counting completed steps or wall time. Fault injection at call17 verifies16 completed plus1 issued failure.

Fresh root200001 holdout completed: all30 theta0 mean−0.001530734, Base−0.002280479, Dir−0.002174749. Delay2 n10 theta0−0.001763133, Base−0.002325726, Dir−0.002190826. Both frozen learned policies underperform theta0 on this new holdout, despite Dir exceeding Base numerically. Preserve this negative generalization result; no claim of repaired original-policy evidence. Source content unchanged.
