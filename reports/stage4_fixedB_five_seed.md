# Stage 4 — Fixed Controller B Five-Seed Results

Config frozen from Stage 3.2: gain=1.05, extra_lag=1, rate_limit=0.045, smooth_beta=0.25 · 3000 updates · matched-θ Dir · paired Base/Dir · physical limits · trajectory splits.  
Runs: `runs/stage4_fixedB_five_seed/` · wall **1886 s**

## Seed-mean metrics

| Model | Return mean | Return std | Cosine mean | #seeds > Mc (return) |
|---|---:|---:|---:|---:|
| Mc | −0.002379 | 0.000627 | 0.492 | — |
| EAGM-Base | −0.002709 | 0.000350 | 0.628 | 2/5 |
| **EAGM-Dir-Matched** | **−0.001703** | 0.000362 | **0.826** | **4/5** |
| EAGM-Hist | −0.002216 | 0.000338 | 0.687 | 4/5 |
| EAGM-Hist+Dir-Matched | −0.002003 | 0.000295 | 0.862 | 3/5 |

## Paired seed-level bootstrap (n=5 seeds)

| Contrast | Return Δ mean | 95% CI | #pos | Cosine Δ mean | Cosine CI |
|---|---:|---|---:|---:|---|
| Dir-Matched − Mc | **+6.75×10⁻⁴** | **[+1.24×10⁻⁴, +1.48×10⁻³]** | 4/5 | +0.334 | excludes 0 (5/5) |
| Dir-Matched − Base | **+1.01×10⁻³** | **[+5.77×10⁻⁴, +1.30×10⁻³]** | **5/5** | — | — |
| Hist − Mc | +1.63×10⁻⁴ | crosses 0 | 4/5 | +0.196 | excludes 0 |
| Hist+Dir − Mc | +3.76×10⁻⁴ | crosses 0 | 3/5 | +0.370 | excludes 0 |
| Base − Mc | −3.30×10⁻⁴ | crosses 0 | 2/5 | +0.137 | excludes 0 |

**Caution:** n=5 seeds; CI is indicative. Per-start rows are nested in trajectories — do not treat pooled starts as i.i.d.

## Final judgment (fixed Controller B)

**Yes — with matched-θ Dir and protocol repairs, EAGM-Dir-Matched shows a reliable five-seed paired improvement in real policy return over Mc, not only higher cosine.**

- Cosine gain is large and consistent (5/5).  
- Return gain vs Mc has CI excluding 0 (4/5 seeds).  
- Dir’s return gain vs its paired Base is even clearer (5/5, CI excludes 0) → Dir loss itself matters under matched batches.  
- History alone / Hist+Dir: cosine↑ but return vs Mc **not** yet reliable at n=5.

**Not claimed:** held-out/random Controller B generalization; SOTA; public benchmarks; that cosine alone equals policy success.

## Held-out failure (from Stage 3.1; unchanged this round)

Training was fixed-B only. Randomized held-out B previously showed dyn NMSE ~2.4–2.8 and no return edge. That is expected coverage failure, not a refutation of fixed-B Dir gains. Separate train-time randomization + frozen OOD test is required next if pursuing OOD.

## Next (only if continuing)

1. Optional: fresh held-out trajectories for confirmatory fixed-B test (if Stage 3.2/4 test starts were explored).  
2. Train-time Controller B randomization experiment (new data, new held-out).  
3. CF loss only after fixed-B Dir remains stable.
