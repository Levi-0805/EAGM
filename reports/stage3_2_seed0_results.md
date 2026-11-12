# Stage 3.2 Seed-0 Results (matched-θ Dir)

Dir: `runs/stage3_2_fixedB/seed0/`  
Wall: 518.7 s · Dir FD cost: 64,512 control steps · Base twin param L2² = **0** (paired init/batches OK)

Controller B fixed: gain=1.05, extra_lag=1, rate_limit=0.045, smooth_beta=0.25 · 3000 updates

| Model | Dyn NMSE | Cosine | Policy return | Δ vs θ=0 | ‖Δθ‖ |
|---|---:|---:|---:|---:|---:|
| Mc | 0.301 | 0.525 | −0.001589 | +0.000926 | 5.04 |
| EAGM-Base | 0.259 | 0.553 | −0.002673 | −0.000158 | 4.93 |
| EAGM-Dir-Zeroθ | 0.259 | **0.846** | −0.001530 | +0.000985 | 5.18 |
| EAGM-Dir-Matchedθ | 0.256 | 0.780 | **−0.001317** | **+0.001198** | 5.08 |
| EAGM-Hist | 0.171 | 0.714 | −0.002210 | +0.000305 | 4.79 |
| EAGM-Hist+Dir-Matchedθ | 0.174 | 0.869 | −0.001602 | +0.000913 | 5.20 |

θ=0 baseline return: −0.002515. Final θ differ across Dir variants (‖DirMatch−DirZero‖≈1.21) — not identical collapse.

## Q1–Q4 (seed 0)

**Q1:** Yes — after θ alignment, Dir still raises cosine (Base 0.55 → Zero 0.85 / Match 0.78 / HistDir 0.87).  
**Q2:** Yes for **Dir-Matched** vs Mc on iid return (−0.00132 vs −0.00159). Base/Hist alone do not.  
**Q3:** Hist+Dir does **not** beat Dir-Matched on return; History helps NMSE, Dir-Matched drives the iid return edge here. θ moved ~5 in all cases (real updates, not θ=0).  
**Q4:** 64k FD steps (~comparable to prior Stage 3.1 label cost); acceptable for this diagnostic.

## Gate to Stage 4 (fixed B five-seed)

**PASS (conditional):** P0 tests green; paired Base twin exact; Dir-Matched shows both cosine↑ and iid return↑ on seed0.  
Proceed with frozen config: Mc / Base / Dir-Matched / Hist / Hist+Dir-Matched @ 3k × seeds 0–4.
