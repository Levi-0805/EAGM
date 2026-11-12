# Stage 3.1 Repaired Seed-0 Results

Branch: `stage3-protocol-repair`  
Run dir: `runs/stage3_1_seed0/`  
Old exploratory runs: **deprecated** (`runs/stage3_b_*/DEPRECATED.md`).

Fixed Controller B: gain=1.05, extra_lag=1, rate_limit=0.045, smooth_beta=0.25.  
Split: 12/6/6 trajectories. Held-out: 6 separately collected randomized-B episodes.  
Device: CPU FP32 (protocol). Direction FD cost: 32,256 sim control steps (labels).

## vs deprecated Stage 3 (protocol-broken)

| Item | Old exploratory | Stage 3.1 @ 3k |
|---|---|---|
| u timing | before step (wrong) | after step (=ctrl) |
| split | frame shuffle | trajectory IDs |
| reward | EAGM pos-only | unified pos+vel+Δa+0.99 |
| history | none (misnamed) | GRU hist_len=8 |
| Mc return | −0.00563 | −0.00184 |
| EAGM-Base return | **−0.00386 (appeared best)** | −0.00292 (**worse than Mc**) |
| EAGM-Dir cosine | 0.833 | 0.893 |
| EAGM-Dir return | −0.00432 | **−0.00149** (best among deployable≈HistDir) |

**Q1:** After timing fix, EAGM-Base’s old “dynamics/policy advantage over Mc” **does not hold** on independent test trajectories for policy return. Base dyn NMSE can still beat Mc, but return does not.

## 1,000 updates (seed 0)

| Model | Dyn NMSE | Cmd MAE | Cosine | Bal. dir acc | Policy return (test) | Held-out return |
|---|---:|---:|---:|---:|---:|---:|
| Mc (A-memory) | 0.283 | — | 0.623 | 0.805 | **−0.00144** | −0.0154 |
| Mu one-step | 0.280 | — | — | — | *(no policy)* | — |
| EAGM-Base | 0.278 | 0.00336 | 0.434 | 0.562 | −0.00446 | −0.0160 |
| EAGM-Dir | 0.273 | 0.00277 | 0.884 | 1.000 | −0.00157 | −0.0150 |
| EAGM-Hist | 0.263 | 0.00283 | 0.781 | 0.572 | −0.00277 | −0.0150 |
| EAGM-Hist+Dir | 0.259 | 0.00306 | 0.891 | 1.000 | −0.00157 | −0.0150 |

## 3,000 updates (seed 0)

| Model | Dyn NMSE | Cmd MAE | Cosine | Bal. dir acc | Policy return (test) | Held-out return |
|---|---:|---:|---:|---:|---:|---:|
| Mc | 0.301 | — | 0.591 | 0.808 | −0.00184 | **−0.0145** |
| Mu one-step | 0.294 | — | — | — | — | — |
| EAGM-Base | 0.244 | 0.00280 | 0.722 | 0.555 | −0.00292 | −0.0148 |
| EAGM-Dir | 0.256 | 0.00305 | 0.893 | 0.920 | −0.00149 | −0.0151 |
| EAGM-Hist | 0.179 | 0.00379 | 0.750 | 0.716 | −0.00187 | −0.0150 |
| EAGM-Hist+Dir | **0.168** | 0.00392 | **0.941** | **0.941** | **−0.00148** | −0.0152 |

Held-out **dynamics NMSE ≈ 2.4–2.8** for all EAGM variants (severe OOD under randomized B).

## Answers

**Q1:** Timing-fixed EAGM-Base does **not** retain the old policy-return win over Mc.  
**Q2:** Unified reward: Dir still raises cosine a lot; at 3k Dir/HistDir **also** beat Mc on iid test return (seed 0 only). At 1k, Mc still leads return.  
**Q3:** On independent test trajectories @ 3k, Hist+Dir / Dir > Mc on return (single seed). Base does not.  
**Q4:** History clearly helps **iid dyn NMSE** (0.17 vs Base 0.24); helps cosine with Dir; held-out B still fails.  
**Q5:** Protocol A–E **PASS**. Criterion F: **partial** — interpretable iid signal at 3k for Dir/HistDir, but held-out B negative and single-seed only.

## Stage 4 gate decision

**Do not launch the full Stage-4 five-seed × held-out matrix as a method claim.**

Allowed next step (narrow): five-seed **fixed** Controller B @ 3k for Mc vs EAGM-Dir vs EAGM-Hist+Dir, with trajectory-cluster CIs — only to test whether the iid return edge replicates.  
Blocked until improved: claiming OOD/held-out execution robustness; SOTA language; scaling visual Stage 6.
