# Module inventory (legacy vs required)

## Present (usable for Stage 1)

| Module | Path | Notes |
|---|---|---|
| ArmEnv / Controller A | `vendor/legacy_sim/sim_env.py` | Exact accumulate/filter/FIFO; \(u_t\) reconstructible |
| World models Mx/Mc/Mh/Mu | `vendor/legacy_sim/experiment.py` | CPU FP32; no learned bottleneck |
| Dir pilot Mu+Dir | `vendor/legacy_sim/eagm_pilot.py` | Stability mask returned but **not** used in loss |
| Stage2 / 3k runners | `eagm_stage2.py`, `eagm_mudir3k.py` | Exact-controller setting only |
| CF diagnostics | `counterfactual_gradient.py` | Evaluation, not training loss |
| Pilot data | `runs/pilot_20261101/` | transitions, starts, env meta |
| 1k/3k checkpoints | `extended_...`, `revision4_...` | For Mc/Mu references |
| EAGM archive + stop report | `EAGM_archive_20261112/` | Stop condition documented |

## Missing (must build in `eagm_gpu/`)

| Module | Target | Priority |
|---|---|---|
| Controller B partial-unknown | `controllers/partial.py` | High (Stage 2) |
| Controller C history-only | `controllers/history.py` | High (Stage 2) |
| Execution memory encoder | `models/memory_encoder.py` | High (Stage 3) |
| Learned command bottleneck | `models/command_bottleneck.py` | High (Stage 3) |
| Masked + policy-region Dir loss | `training/dir_loss.py` | High (Stage 3) |
| CF consistency loss | `training/cf_loss.py` | Medium (Stage 3) |
| Fair budget accounting | `evaluation/budget.py` | Medium |
| Visual EAGM | later | Stage 6 only after gains |

## Stop-condition reminder

Do not scale exact \(M_u+\mathrm{Dir}\) 3k as a new method claim. New scientific leverage is Controllers B/C + proper supervision + policy return under matched sim budget.
