# Geometry and Jacobian audit

Exploratory seed20 frozen-model causal diagnostic. No new world-model training, no independent confirmation. Sources: runs/stage8_1_geometry/; script analysis/stage8_1_reports.py.

| Distribution | Position pooled RMSE m | J RMSE | relative J error | J cosine | reward gradient cosine |
|---|---:|---:|---:|---:|---:|
| old_final_imagined | 0.023385 | 0.029039 | 0.3000 | 0.9652 | 0.8421 |
| old_final_physical | 0.022589 | 0.028890 | 0.2989 | 0.9651 | 0.8275 |
| random_legal | 0.030688 | 0.025180 | 0.3263 | 0.9442 | 0.9083 |
| test | 0.022436 | 0.028680 | 0.2996 | 0.9638 | 0.4410 |
| theta0_imagined | 0.023370 | 0.029015 | 0.2998 | 0.9652 | 0.8437 |
| theta0_physical | 0.022994 | 0.028955 | 0.2994 | 0.9651 | 0.8181 |
| train | 0.028895 | 0.031845 | 0.3287 | 0.9606 | 0.4827 |

Geometry FD max error across three eps and five random configurations: 6.61e-08. Static phase control steps=0. Mapping:

```json
{
  "nq": 14,
  "nv": 14,
  "njnt": 14,
  "sites": [
    1,
    2
  ],
  "joints": [
    "openarm_left_joint1",
    "openarm_left_joint2",
    "openarm_left_joint3",
    "openarm_left_joint4",
    "openarm_left_joint5",
    "openarm_left_joint6",
    "openarm_left_joint7",
    "openarm_right_joint1",
    "openarm_right_joint2",
    "openarm_right_joint3",
    "openarm_right_joint4",
    "openarm_right_joint5",
    "openarm_right_joint6",
    "openarm_right_joint7"
  ],
  "qpos_addresses": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13
  ],
  "dof_addresses": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13
  ],
  "units": "positions: world-frame metres; hinge q: radians; Jacobian m/rad",
  "calls": 917,
  "seconds": 0.04305084799489123
}
```

All joints asserted hinge, nq=nv=14; qpos/dof addresses identical. Jacobians are world translation metres/radian, ordered left xyz then right xyz. Robot body tree controls which arm columns contribute; no manual column permutation. Left/right and per-joint errors in static_summary.json. Independent MjData is used for mj_forward/mj_jacSite. Train/test samples retain original episode IDs; there is no new adjacent-frame split. Random states respect limits. Imagined theta0/old-final states are explicitly model predictions, not physical trajectories. Supplemental actual theta-state observations, if present, are separately labeled and charged to the control budget.


Total added physical control steps including actual-state coverage: **19488/20000**. Main and supplemental raw budgets are preserved separately; combined accounting is total_budget.json. Static metrics CSV, component/goal/configuration summaries and SHA-256 inventory are in the run directory. No prior file changed relative to Stage8.1 preservation manifest.
