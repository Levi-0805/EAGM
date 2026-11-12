# Stage 3.2 Protocol Audit

Branch: `stage3-protocol-repair`  
Date: 2026-11-12

## P0 findings from Stage 3.1 review (confirmed in code)

1. **θ mismatch:** `sim_grad_masked` FD at θ=0 while training `autograd` used non-zero θ → fixed via `training/gradients.py::sim_grad_masked(..., theta)` and matched label packs.
2. **RNG:** no `torch.manual_seed` in Stage 3.1 → fixed via `training/seeds.py::seed_everything` + paired batch generators.
3. **Imagined limits:** `fit_stats` used data ±0.05 as lower/upper → now uses `env.lower`/`env.upper`; data margins kept only for audit.
4. **Paired Base/Dir:** `training/paired_train.py` clones init and shares minibatch stream; only Dir loss differs.

## Variants this round

| ID | Architecture | Dir labels |
|---|---|---|
| Mc | Controller-A memory | — |
| EAGM-Base | Markov residual | none |
| EAGM-Dir-ZeroTheta | Markov | ∇J_sim(θ=0) vs ∇Ĵ(θ=0) |
| EAGM-Dir-MatchedTheta | Markov | ∇J_sim(θ_j) vs ∇Ĵ(θ_j) |
| EAGM-History | GRU history | none |
| EAGM-History+Dir-MatchedTheta | GRU | matched θ |

## Tests

```bash
python tests/test_stage3_2_protocol.py
python tests/test_protocol_p0.py
```

Both passed before Stage 3.2 seed0 launch.
