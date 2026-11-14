# EAGM 2.0 research execution checklist — adopted 2026-11-14

This record operationalizes the author's Phase 0–9 checklist. It does not
authorize confirmation, paper changes or a new method based on partial outcomes.
Reference commit: f3a998e; actual frozen development implementation: 5037c49.

## Phase 0: baseline and provenance

- [x] Independent branch, GitHub push, historical paper/data protection.
- [x] 78 discovered tests and three task smoke gates (see unit-test report).
- [x] Baseline commit, complete tracked diff, implementation diff and source archive.
- [x] Python/PyTorch/MuJoCo/CUDA metadata and pip package snapshot.
- [x] Hash repository MJCFs and full existing external OpenArm v2 assets tree.
- [x] Register actual attempts with seed/method/config_hash/attempt_id in an independent CSV.
- [x] Inventory existing local tensor artifacts and retained failures.
- [ ] Final immutable archive and hashes after all running writes finish.
- [ ] Clean-environment reproduction; a package snapshot alone does not prove this.

Files: `baseline_snapshot_v1/{project_baseline_manifest.json,environment_lock.json,
source_hash_manifest.json,experiment_registry.csv,artifact_inventory.json}`.
Queued roots have prospective registrations in batch plan.json, not invented
actual config hashes. Current inventory hashes are timestamped snapshots.

## Phase 1: nine exploratory roots

- [ ] Complete tracking 200001–200003, contact 201001–201003, vision 202001–202003.
- [ ] Verify completed/failed/missing methods, update counts, 30 paired test starts,
  checkpoints, split/config/source hashes and all retained attempts.
- [ ] Privileged reference experiments: interface tested, full experiment absent.
- [ ] RGB-only policy comparison: interface tested, full experiment absent.
- [ ] Full failed contact trajectory visualization, rebound and stability analysis.

The frozen main matrix has 21 tracking/contact conditions and 18 vision
conditions. It includes D/R/G, capacity, gradient-loss, adaptation and query
controls. It does not cover all conditions in the new research checklist.
Vision shares one RGB+record policy with a frozen encoder and 32-D adapter;
world-model recurrence is not a separately trained recurrent RGB policy.
Existing methods remain fixed throughout this batch. Missing conditions require
separate versioned development attempts and cannot be retroactively inserted.

## Phase 2: analysis and selection

- [ ] Produce per-root outcomes, paired method comparisons and failure analysis.
- [ ] Separate full-coordinate/masked gradients, finite coverage, undefined probes,
  coordinate errors, delay/task/policy-position strata.
- [ ] Report model metrics sampled from up to 256 evenly spaced sequences;
  full-test evaluation remains additional work, not implied by that sample.
- [ ] D/R/G effects, D×R, D×G, R×G, capacity and extra-data comparisons.
- [ ] Apply validation-only selection: mean return gain >0, positive 2/3 roots,
  unseen-probe mean cosine gain >0; contact/vision additionally success gain >0
  and nonnegative 2/3 roots, with fairness/test gates.
- [ ] Prefer simpler passing candidates; record no-go if none qualifies.

The existing postrun analyzer is descriptive and does not choose a winner.
D×G/R×G, complete-test model evaluation and automatic selection are not promised
by its current output. Base-extra-data NMSE has a different train-scale denominator.
For now show both `ff_dir` and frozen `legacy_dir` reference comparisons; the
author's generic “Dir” must resolve to one exact implementation/protocol in the
selection record before confirmation. Never resolve this by picking the better result.

## Phase 3: corrections and boundaries

- [x] Actual Controller B FIFO/prior Cartesian tests, nominal/executed distinction.
- [x] Observable-only causal memory API and fixed 32-frame history tests.
- [x] Independent postrun methods=None and budget-reader correction, tested on
  synthetic structures; frozen original analyzer is retained.
- [ ] Memory-length and initial-filter approximation experiments.
- [ ] Contact threshold/hold/timeout boundary matrix and future floor/ceil rule.
- [ ] Physical-substep accounting and partial-integration failure accounting.
- [ ] Optimizer/RNG resumed-training equivalence across full task checkpoints.
- [ ] Vision encoder/reward error, perturbations and information-input comparisons.

Development retains its existing 10.030s final-step and 9.996s last-success
sampling semantics. A confirmation correction must be frozen for every method.
CUDA event elapsed is not GPU-active kernel time. Shared data/critic/encoder
costs must be disclosed separately from per-method logical costs.

## Phases 4–6: confirmation and additional evidence

- [ ] Exact architecture, optimizer, losses, FD scales/mask, horizons, data/updates.
- [ ] Freeze metrics, comparisons, statistical family, failure rules and episode IDs.
- [ ] Freeze 200101–200115 / 201101–201115 / 202101–202115 after collision checks.
- [ ] Estimate chosen roster from actual runtime, plus 25% reserve and rerun costs.
- [ ] Author's second approval. **Current authorization remains false.**
- [ ] Execute authorized confirmation to the fixed stopping rule; retain failures.
- [ ] Execution single/joint-factor OOD, vision perturbations, hidden-state controls,
  shuffled histories, memory length and efficiency studies under frozen budgets.

The proposed 16/16/6 roster totals 570 method-seed runs but is a prospective
design, not the existing development matrix. The vision endpoint names
RGB+observable-record+gradient as comparator while the six-condition roster says
RGB+observable-record without explicitly specifying gradient. Resolve its exact
loss condition before budget/protocol freezing. Sparse tracking/contact OOD
probes do not complete the full OOD or vision-generalization checklist.

## Phases 7–9: theory, paper and independent review

- [ ] Define observable/hidden execution state, actions, nominal/executed commands.
- [ ] Verify causal Jacobians, horizon error bounds and FD assumptions/proofs.
- [ ] Match theoretical quantities to implemented model and measured errors.
- [ ] Obtain and hash designated Final manuscript and correct Supplementary sources.
- [ ] Authorize paper reconstruction after validated evidence is available.
- [ ] Update claims/evidence/ledger, report negatives and computational limits.
- [ ] Three independent reviewers, clean reproduction and final submission audit.

Final manuscript and correct supplementary files are still prerequisites to the
paper phase. No acceptance rating or research-complete status follows from
passing engineering tests, scheduled runs or automated artifact generation.

## Next review deliverables

`development_v1_summary.md`, `development_v1_per_seed.csv`,
`development_v1_method_comparison.csv`, `development_v1_ablation.csv`,
`development_v1_failure_analysis.md`, `development_method_selection.md`,
`development_acceptance_report.md`, `selected_model_architecture.md` are required
handoff targets. Their existence and scientific adequacy must be checked after
the actual batch; the current automation's differently named tables are inputs,
not an assertion that all these decisions and reports have already been made.
