# Stage 8.2 reproduction and artifact guide

Use a separate fresh workspace with the preserved source tree and Stage7C2 source artifacts. Run scripts refuse existing experiment directories. Do not delete or overwrite historical runs to rerun.

Interpreter: `/home/aaa/miniconda3/envs/eagm_gpu/bin/python`.
Set `OPENARM_MODEL_PATH=/home/aaa/HIL-OpenArm/third_party/openarm_mujoco/v2/openarm_bimanual.xml` and `PYTHONPATH=.:vendor/legacy_sim`.

Execution order:

1. `python -B tests/test_stage8_2_dynamics.py` (96 physical steps).
2. `python -B analysis/stage8_2_run.py` (A-C, hard cap40000).
3. If gate passes, `python -B analysis/stage8_2_policy.py` (D, hard cap20000).
4. `python -B analysis/stage8_2_localize.py` (conditional actuator F and train/test horizon checks; D remainder enforced).
5. `python -B analysis/stage8_2_stability.py` (three-scale policy references, local derivative checks; extra cap15000).
6. `python -B analysis/stage8_2_reports.py` (no simulation; derived CSV, five reports, figures).
7. `python -B tests/test_stage8_2_offline.py` (no simulation).

Actual aggregate A-C remained below40000, including the supplemental runs. For other data, ensure the aggregate limit before running step5; its local cap is not a substitute for aggregate accounting. `total_budget.json` records actual counts, cached queries and the failed localization attempt. No failed outputs were overwritten. The first localization attribute-name failure (one step) is retained in `runs/stage8_2_localization/`; successful code writes a separate v2 path. A clean reproduction without that failed attempt consumes one fewer step, and should record that distinction rather than fabricate a failure.

Raw sources:

- `stage8_2a_single_step`: full 154x168 physical/model matrices, 28D gradients, 28x28 state sensitivities, common forward states, gate, numerical chain check.
- `stage8_2_numerical`: independent .005/.01/.02 policy stability and .5/2 local Jacobian scales. Use its validated metrics for stable-coordinate statistics. Original A-C stable fields were all-coordinate placeholders.
- `stage8_2d_frozen_policy`: saved actors, one-step training gradients, per-start physical returns, final true FD vectors. Final FD is single-scale and has no validated stability mask.
- `stage8_2_localization_v2`: conditional F(x,u) matrices, two training/six test starts, H8/H16 outcomes, true training gradients.
- `stage8_2_summary`: derived CSV and decision/budget/provenance; `reports/figures/stage8_2` contains generated plots.

A/B/C/D mean reward/dynamics derivative combinations: original/learned, true/learned, original/true, true/true. All forward values coincide at common theta. Here B/D mean complete true reward derivative, not true geometric J alone.

Source code, existing checkpoints and model XML checksums are in runtime/source_inputs and the preservation manifest. No model training, new architecture, independent seed replication, paper-core claim change, remote push or main merge occurred. Full-oracle gradient recovery did not translate to held-out mean physical return improvement in the matched one-update experiment.
