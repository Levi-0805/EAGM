# EAGM manuscript draft

Independent writing directory; CVPR2027 is untouched. Eight planned body pages plus references, using an explicit page break for each substantive section. This is a research draft, not submission-ready or an accepted-paper claim. All eight sections contain draft prose; tables/plots are generated from archived data. Stage8.3 mixed results are included, not deferred or described as success.

## Build

From repo root: `python paper_eagm/build_assets.py`.
Then `tectonic --keep-logs paper_eagm/main.tex`, or inside paper_eagm run pdflatex, bibtex, pdflatex twice.

This computer has a cached Tectonic binary outside PATH; its ICU79 runtime is required. Build logs document any missing package/network failures. Consult `build_status.json` for the final actual compilation outcome/page count; do not infer a successful build merely from this README.

## Page plan

1. Abstract and Introduction: restricted method contribution, no SOTA claim.
2. Related Work: PAML/GAMPS objective distinctions, AWM, VaGraM, execution interfaces; novelty limits.
3. Problem Formulation: observable/hidden execution state, full sensitivity recurrence, conditional smoothness inequality.
4. Method: U0+Residual, actual normalized Huber gradient loss, matching theta and masks.
5. Setup: split/budget/pairing, true reward, evidence levels and five-seed limitations.
6. Main Results: five historical within-protocol contrasts, raw-linked seed table and figure.
7. Ablations/Generalization: OOD limits, failed HistC/Path, corrected Reach and geometry failure.
8. Analysis/Limitations: dynamics interventions, Stage8.3 validation/fresh-test conflict, no new-method promotion.
References follow separately; no conference style or official eight-page compliance is claimed.

## Before submission

- Complete external baseline adaptations and formula/code-level novelty review.
- Establish the distinction from the submitted Execution Memory diagnostic; record overlapping tasks/data and any disclosure requirements of the chosen venue.
- Resolve missing real-robot and visual evidence; do not manufacture a visual contribution for CVPR.
- Replace provisional author line, select venue template and audit layout.
- Any new method needs independent physical-return evidence; neither oracle nor a selected start is sufficient.

Data provenance: `data/source_manifest.json`, `data/core_seed_differences.csv`, `build_assets.py`, `claims_ledger.md`. Original data/configs are not rewritten. Stage4/5 are separate evidence blocks; Stage6 protocol caveats are retained. No pooled seed count across protocols.
