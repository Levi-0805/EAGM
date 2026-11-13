# Stage 6B.2 — HistC Execution-State History Results

Observable `c_obs` history only (target/filter/delay). Privileged B state used **only** for analysis probes.
Seeds [10, 11, 12, 13, 14] · k_main=8 · wall≈11868s

## FailRate caveat (cross-stage)

Do not compare FailRate levels across Stage 6A corrected (seeds 5–9) and Stage 6B.1/6B.2 (seeds 10–14) as a Stage progression. Differences mix new seeds and (if any) evaluation-sample changes. Within-stage Dir vs Base contrasts are the claim.

Within this stage, compare models on the **same** seeds and starts. Do not narrate FailRate as improving from Stage 6A→6B.1→6B.2.

## Primary contrasts (policy return)

### Interpolation
- HistC+Dir − Hist+Dir: mean=0.000313264±0.000537149, CI[-0.000127807,0.000734763], n_pos=4/5
- HistC+Dir − Mc: mean=0.000821476±0.00185362, CI[-0.000221785,0.00247795], n_pos=3/5
- HistC+Dir − HistWide+Dir (capacity): mean=-9.8331e-05±0.000829958, CI[-0.00064234,0.000592587], n_pos=1/5
- HistC+Dir − HistCShuf+Dir (time structure): mean=-0.000141607±0.000877267, CI[-0.000912709,0.000557869], n_pos=1/5
- Dir − Base: mean=0.00236393±0.000906451, CI[0.00166425,0.0030636], n_pos=5/5

### Extrapolation
- HistC+Dir − Hist+Dir: mean=0.000318196±0.000421632, CI[-1.27609e-05,0.000649153], n_pos=3/5
- HistC+Dir − Mc: mean=0.00088584±0.0019847, CI[-0.000340081,0.00268422], n_pos=4/5
- HistC+Dir − HistWide+Dir: mean=-0.000184325±0.000854002, CI[-0.000730265,0.000556482], n_pos=1/5
- HistC+Dir − HistCShuf+Dir: mean=-0.000147873±0.00090223, CI[-0.000923578,0.000583822], n_pos=1/5

## Pool means (k=8)

| model | interp ret | ΔR | FailRate | θ-sat | NMSE | cosF | cosS |
|-------|------------|----|----------|-------|------|------|------|
| mc | -0.004777 | 0.000814299 | 0.22 | 0.94 | 0.262 | 0.537 | 0.543 |
| eagm_base | -0.006073 | -0.00048151 | 0.67 | 0.69 | 0.253 | 0.226 | 0.232 |
| eagm_dir_matched | -0.003709 | 0.00188242 | 0.00 | 0.89 | 0.253 | 0.679 | 0.689 |
| eagm_hist | -0.004588 | 0.00100363 | 0.07 | 0.85 | 0.232 | 0.604 | 0.606 |
| eagm_hist_dir | -0.004269 | 0.00132251 | 0.00 | 0.86 | 0.240 | 0.615 | 0.620 |
| eagm_histc_dir | -0.003956 | 0.00163578 | 0.02 | 0.95 | 0.229 | 0.659 | 0.664 |
| eagm_hist_wide_dir | -0.003857 | 0.00173411 | 0.00 | 0.87 | 0.239 | 0.712 | 0.718 |
| eagm_histc_shuf_dir | -0.003814 | 0.00177738 | 0.00 | 0.94 | 0.232 | 0.730 | 0.734 |

### Extrapolation pool

| model | ret | ΔR | FailRate | θ-sat | NMSE |
|-------|-----|----|----------|-------|------|
| mc | -0.006187 | 0.000544583 | 0.33 | 0.94 | 0.511 |
| eagm_base | -0.007591 | -0.000859065 | 0.78 | 0.69 | 0.493 |
| eagm_dir_matched | -0.005183 | 0.00154897 | 0.17 | 0.89 | 0.495 |
| eagm_hist | -0.005994 | 0.000737729 | 0.25 | 0.85 | 0.457 |
| eagm_hist_dir | -0.005620 | 0.00111223 | 0.17 | 0.86 | 0.471 |
| eagm_histc_dir | -0.005302 | 0.00143042 | 0.12 | 0.95 | 0.453 |
| eagm_hist_wide_dir | -0.005117 | 0.00161475 | 0.05 | 0.87 | 0.467 |
| eagm_histc_shuf_dir | -0.005154 | 0.0015783 | 0.08 | 0.94 | 0.457 |

## Per-seed HistC−Hist / HistC−Mc (interp)

| seed | HistC−Hist | HistC−Mc | HistC−Wide | HistC−Shuf |
|------|------------|----------|------------|------------|
| 10 | 6.7221e-05 | -5.15512e-05 | -3.34267e-05 | -5.68397e-05 |
| 11 | 0.000351705 | 0.000526021 | -0.000593795 | -0.00148329 |
| 12 | 0.00101943 | -0.000527798 | -0.00022773 | -9.78805e-05 |
| 13 | -0.000420348 | 9.23759e-05 | -0.00089819 | -5.13801e-05 |
| 14 | 0.000548313 | 0.00406833 | 0.00126149 | 0.000981355 |

## Hidden-state probe (|corr| mean)

- seed 10: eagm_hist_dir:0.4907466255480092, eagm_histc_dir:0.541932275186869, eagm_histc_shuf_dir:0.5620828392359892
- seed 11: eagm_hist_dir:0.4609070077615952, eagm_histc_dir:0.5490669057473095, eagm_histc_shuf_dir:0.5712842595328257
- seed 12: eagm_hist_dir:0.5106980345764565, eagm_histc_dir:0.6124887661076625, eagm_histc_shuf_dir:0.6146952773183763
- seed 13: eagm_hist_dir:0.5728990254387106, eagm_histc_dir:0.6765202620028773, eagm_histc_shuf_dir:0.6600184876648834
- seed 14: eagm_hist_dir:0.5489052173626036, eagm_histc_dir:0.6507684750994754, eagm_histc_shuf_dir:0.652520344364775

## Decision (pre-registered style)

**DROP_HISTC.** HistC may look slightly better than Hist+Dir on average, but it fails capacity-matched and shuffled-c_obs controls (n_pos≤1/5). Extra parameters / feature volume explain the gap, not temporal execution-state estimation. Do NOT start ExecState (6B.3). Prefer Stage 6C (saturation / path-aware Dir) after human review.

Stage 6B.3 / 6C were **not** auto-started.

