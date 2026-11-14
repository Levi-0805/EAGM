# EAGM 2.0 authorized development

Current scope: nine exploratory roots; **45 confirmation roots are not authorized**.
No historical paper or raw experiment data is modified.

Start with:

* `05_implementation_report.md`: implemented modules and scientific limits.
* `05_unit_test_report.md` and `tests_full_discovery_final.log`: 78-test gate.
* `05_frozen_pipeline_review.md`: independent review and unresolved boundaries.
* `05_development_acceptance_status.md`: acceptance scope, not a positive verdict.
* `../../runs/eagm2_batches/development_v1/experiment_status.csv`: live nine-root status.
* `development_completion_worker.json`: live postprocessing/publication status.
* `github_transport_diagnosis.md`: actual GitHub transport fix and successful push.

The fixed batch uses 1000 model updates, 50 model-policy updates, and 30 test
starts per root; Direct FD/SPSA can complete fewer updates under query caps.
Tracking/contact have 21 method variants/controls each, vision 18. Test starts
are repeated measurements, not additional independent seeds. Failed and smoke
attempts remain preserved and cannot be relabeled confirmation evidence.

After the batch, the independent completion worker is configured to generate:

* `development_v1_postrun_analysis/`: explicit-attempt per-seed, paired,
  factorial, capacity, gradient, resource and budget tables; no winner selection.
* `ood_*_v1/`: fixed sparse tracking/contact execution-shift diagnostics;
  vision OOD is not implemented and will not be claimed.
* `05_reproducibility_manifest.json`: hashes of raw local artifacts and archives.
* `06_compute_resource_report.md`: measured successful-root resource totals.
* `06_development_handoff.md`: review handoff, not confirmation authorization.
* `../../runs/eagm2_artifacts/`: checkpoint archives for Git publication.

These are planned output paths until actual files exist. If training fails,
the scheduler stops subsequent roots; the worker saves partial analysis and a
needs-review status, without inventing results or restarting altered protocols.
The original frozen automatic-analysis failure is retained separately from the
corrected postrun report. Neither program changes the paper or launches any
confirmation root. Scientific candidate selection and the full confirmation
configuration remain subject to evidence review and a second author approval.
