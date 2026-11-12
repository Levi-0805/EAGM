# EAGM architecture (draft — implementation in progress)

## Status

| Piece | Status |
|---|---|
| CompatibleArmEnv (Controller A on pilot XML) | Done |
| Controller B partial-unknown | Done (smoke OK) |
| Controller C history-only interface | Pending |
| `models/eagm.py` HistoryEncoder + CommandBottleneck | Scaffolded |
| Masked Dir loss + policy-θ sampling | Pending |
| CF consistency loss | Pending |
| Single-seed Controller B experiment | Smoke ran (`runs/stage3_b_min_seed0`); residual cmd regression needs fix |

## Controller observability

### A — exact known (legacy)

Full \(c_t\) (57-D); \(u_t=U(c_t,a_t)\) exact.

### B — partial unknown (`PartialUnknownArmEnv`)

Adds after nominal A-command: gain, extra lag FIFO, rate limit, optional smooth.
Episode-level randomization optional for held-out configs.

- **Inference / model inputs:** \(x\), \(a\), \(c^{obs}=(\mathrm{target},\mathrm{filter},\mathrm{delay})\), optional declared config vector.
- **Train-only labels:** true \(u\), full \(c\), B internals.
- **Fairness:** all baselines face the same B plant; only privileged labels differ.

### C — history inference (planned)

No hidden queue at inference; recover execution belief from \((o,a)\) history only.

## Model

\[
m_t=H_\omega(\text{hist}),\quad
\hat u_t=U_0(c_t^{obs},a_t)+\Delta_\omega(m_t,a_t),\quad
\hat x_{t+1}=F_\phi(x_t,\hat u_t)
\]

Parameter count will be reported vs width-matched MLP once training scripts land.

## Losses (planned)

\(\mathcal L=\mathcal L_{dyn}+\lambda_u\mathcal L_{cmd}+\lambda_d\mathcal L_{dir}+\lambda_c\mathcal L_{cf}\)

Dir loss **must** apply the three-scale stability mask (legacy pilot did not).
