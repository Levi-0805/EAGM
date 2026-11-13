SUPERSEDED: preliminary cached-site diagnostic; use current-site final reports.

# Stage 7C.2 model comparison

Seed20 is exploratory and was used to discover this protocol issue. No multiseed experiment is authorized by this report. Old Stage 7C.1 artifacts are unchanged.

| Model | True R0 | True Rfinal | Δ true | FailRate | Surrogate Rfinal | NMSE | Command MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| mc | -0.01546561 | -0.01549816 | -0.00003255 | 0.500 | -0.03105088 | 1.49788 | None |
| base | -0.01546561 | -0.01545796 | 0.00000766 | 0.167 | -0.03187282 | 0.85143 | 0.01959952525794506 |
| dir_random | -0.01546561 | -0.01558339 | -0.00011777 | 0.833 | -0.03173820 | 0.99063 | 0.019993219524621964 |
| aux | -0.01546561 | -0.01602924 | -0.00056362 | 0.667 | -0.03207142 | 3.33715 | 0.014616236090660095 |

True ranking: ['base', 'mc', 'dir_random', 'aux']. Surrogate ranking: ['mc', 'dir_random', 'base', 'aux'].

Dir−Base per-start differences: [0.0008737993417840357, -0.0009236557187213869, -0.00039167346517373844, 0.00026999286053436267, -0.00020834894193899958, -0.00037270281039668285]; positive 2/6. Starts are not independent training seeds.

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
  "gradient_audit_queries": 10752,
  "policy_eval_queries": 192,
  "all_fd_and_eval_queries": 12960
}
```
