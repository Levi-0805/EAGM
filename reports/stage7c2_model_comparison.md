# Stage 7C.2 model comparison

Seed20 is exploratory and was used to discover this protocol issue. No multiseed experiment is authorized by this report. Old Stage 7C.1 artifacts are unchanged.

| Model | True R0 | True Rfinal | Δ true | FailRate | Surrogate Rfinal | NMSE | Command MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| mc | -0.01545618 | -0.01548891 | -0.00003273 | 0.500 | -0.03105088 | 1.49788 | None |
| base | -0.01545618 | -0.01544864 | 0.00000755 | 0.167 | -0.03187282 | 0.85143 | 0.01959952525794506 |
| dir_random | -0.01545618 | -0.01571363 | -0.00025745 | 0.667 | -0.03142499 | 0.99958 | 0.01977790892124176 |
| aux | -0.01545618 | -0.01602261 | -0.00056643 | 0.667 | -0.03207142 | 3.33715 | 0.014616236090660095 |

True ranking: ['base', 'mc', 'dir_random', 'aux']. Surrogate ranking: ['mc', 'dir_random', 'base', 'aux'].

Dir−Base per-start differences: [0.00011058935400562551, -0.0026910030718864664, -0.0006067311897038144, 0.0010205756613434515, -0.0003316656886934999, 0.0009082683532589085]; positive 3/6. Starts are not independent training seeds.

All models use identical trajectories/splits and batch seed 50. Base/Dir share exact initialization; different architectures cannot share identical parameter tensors. Old 7C.1 Mc used batch seed 40, so this is a corrected protocol, not exact replication.

Budget:

```json
{
  "collection_steps": 7200,
  "fk_fit_samples": 4000,
  "fk_fit_steps": 800,
  "train_label_steps": 16128,
  "train_label_queries": 2016,
  "gradient_audit_steps": 86016,
  "total_control_steps": 111264,
  "evaluation_control_steps": 87936,
  "note": "All env.step calls counted. FK/IK queries are kinematic evaluations, not control steps. All models share collection; only Dir consumes training FD labels.",
  "train_dynamics_steps": 4800,
  "validation_collection_steps": 1200,
  "test_collection_steps": 1200,
  "dir_total_training_steps": 20928,
  "gradient_audit_queries": 10752,
  "policy_eval_queries": 192,
  "all_fd_and_eval_queries": 12960
}
```

Primary artifacts: `runs/stage7c2_true_ee_current_sites_seed20/`. True reward refreshes site positions from current qpos in separate MjData, preserving live integration/warm-start state. Earlier cached-site preliminary diagnostics remain under `runs/stage7c2_true_ee_seed20/` and are superseded. Each run used 111264 control steps; both runs together used 222528, excluding separately recorded replay tests.
