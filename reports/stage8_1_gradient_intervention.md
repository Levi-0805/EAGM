# Controlled gradient interventions

Exploratory seed20 frozen-model causal diagnostic. No new world-model training, no independent confirmation. Sources: runs/stage8_1_geometry/; script analysis/stage8_1_reports.py.

| Model | condition | full cosine | stable cosine | relative gradient error |
|---|---|---:|---:|---:|
| mc | A | -0.02738 | -0.02738 | 1.50210 |
| mc | B | 0.00473 | 0.00473 | 1.19831 |
| mc | C | 0.00473 | 0.00473 | 1.19831 |
| mc | D | -0.02738 | -0.02738 | 1.50210 |
| base | A | 0.06890 | 0.06890 | 2.84692 |
| base | B | 0.10189 | 0.10189 | 1.89580 |
| base | C | 0.10189 | 0.10189 | 1.89580 |
| base | D | 0.06890 | 0.06890 | 2.84692 |
| dir_random | A | -0.13211 | -0.13211 | 1.05807 |
| dir_random | B | -0.11552 | -0.11552 | 1.02694 |
| dir_random | C | -0.11552 | -0.11552 | 1.02694 |
| dir_random | D | -0.13211 | -0.13211 | 1.05807 |

All rows share theta0, held-out starts, H=8, fixed dynamics and controller recursion. A=surrogate value/gradient, B=true value/gradient, C=surrogate value/true complete reward gradient, D=true value/surrogate complete reward gradient. C/D use scalar stop-gradient corrections. A vs C isolates complete reward derivative at identical forward reward values; it does not isolate geometric J from position residual. Static factorial analysis supplies that distinction. Native-rollout value/gradient parity and A/C, B/D forward and A/D, B/C backward equality tested. Full 28-coordinate gradient arrays and norm/sign diagnostics stored in intervention.json. Analytical TrueFK backward is first-order only; it must not be used for second-order Dir model training without extending/testing it.


Total added physical control steps including actual-state coverage: **19488/20000**. Main and supplemental raw budgets are preserved separately; combined accounting is total_budget.json. Static metrics CSV, component/goal/configuration summaries and SHA-256 inventory are in the run directory. No prior file changed relative to Stage8.1 preservation manifest.
