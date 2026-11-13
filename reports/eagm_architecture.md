# EAGM architecture (Stage 5 complete · Stage 6A protocol)

Frozen Stage-5 tip: `6f66823` · working branch: `stage6-exec-generalization`.

## Status

| Piece | Status |
|---|---|
| CompatibleArmEnv (Controller A) | Done |
| Controller B partial-unknown (+ train-only `privileged_exec_state`) | Done |
| `EAGMBase` U0+Δ command bottleneck | Done (Stage 4/5 confirmed) |
| `EAGMHistory` GRU over `(x,a)` history | Done |
| Matched-θ Dir loss + paired Base/Dir train | Done |
| Fair baselines (ObsMLP, AuxCommandMLP) | Done (Stage 5C) |
| OOD train-time B randomization | Done (Stage 5E); metrics corrected in Stage 6A |
| HistC: history includes `c_obs` | **Planned Stage 6B** |
| ExecState recursive latent + aux state loss | **Planned Stage 6B** |
| Path-aware Dir + saturation-aware policy opt | **Planned Stage 6C** |
| CF consistency loss | Deferred (only if residual CF errors remain) |

## Controller observability

### B — partial unknown (`PartialUnknownArmEnv`)

After nominal A-command: gain, extra lag FIFO, rate limit, optional smooth.

- **Deployable inputs:** \(x\), \(a\), \(c^{obs}=(\mathrm{target},\mathrm{filter},\mathrm{delay})\), optional history of these.
- **Train-only labels:** `u_exec`, full `c`, `privileged_exec_state()` (gain/lag/rate/smooth, `u_prev`, `u_smooth`, lag pad). **Never** as deploy inputs.
- **Oracle leak:** `known_config_vector()` — debug only.

## Models (implemented)

\[
\hat u_t = U_0(c_t^{obs}, a_t) + \Delta_\omega(\cdot),\quad
\hat x_{t+1} = F_\phi(x_t, \hat u_t)
\]

- **Base:** \(\Delta(c^{obs}, a)\).
- **History:** \(m_t = H_\omega(x_{t-k:t}, a_{t-k:t})\), \(\Delta(m_t, a)\). Hist length \(k=8\) in Stage 5.
- **Mc:** full Controller-A 57-D memory (strong baseline; not B-privileged).

Params (width 160): Base/Dir ≈ 40.9k · Hist ≈ higher · ObsMLP ≈ 41.8k · AuxMLP ≈ 53.0k.

## Losses (Stage 5)

\[
\mathcal L = \mathcal L_{\mathrm{dyn}} + \lambda_u \mathcal L_{\mathrm{cmd}} + \lambda_d \mathcal L_{\mathrm{dir}}
\]

with \(\lambda_u=1\), \(\lambda_d=0.1\). Dir uses three-scale stability mask + matched non-zero \(\theta\) labels. CF **not** used.

## Stage 6A metric conventions

- **Fail rate:** \(\mathrm{FailRate}=\frac1N\sum_i\mathbf{1}[R_i(\theta_{\mathrm{final}})-R_i(\theta_0)<-\varepsilon]\) on **paired** starts.
- **θ param saturation** ≠ **action clip fraction** (recorded separately).
- **Gradient quality:** report `cosine_full` and `cosine_stable` (mask) together.
- **Train budget:** train dynamics steps + train dir-label steps only; val / IID holdout / OOD FD are listed separately.

## Stage 5 evidence (short)

- Fixed-B confirmatory seeds 5–9: Dir−Base policy return **5/5**.
- Budget-matched: Dir still beats Base+extra-dyn **5/5**.
- Structured bottleneck beats AuxMLP **5/5**.
- OOD: randomization fixes catastrophic NMSE; Hist+Dir vs Mc **not** stably positive (3/5).
