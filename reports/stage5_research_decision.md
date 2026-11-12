# Stage 5 — Research Decision

Branch `stage5-confirm-budget` · frozen Stage-4 SHA `c386ed3193bea0ea676b4d014be33a3bfe9fe24b`.  
CVPR2027/ diagnostic paper was **not** modified.

## Answers to Q1–Q6

**Q1 — Confirmatory Dir−Base reproduction?**  
**Yes.** Seeds 5–9, new trajectories: Dir−Base mean **+5.14×10⁻⁴**, **5/5** positive; CI excludes 0 (n=5 caveat). Cosine Base→Dir ~0.72→0.87.

**Q2 — Is direction-label sim cost worth it?**  
**Yes under matched control-step budget.** Dir still beats Base+extra-dynamics (C−B mean **+2.59×10⁻⁴**, **5/5**). Not a free lunch in wall-clock/FLOPs, but not explained away as “just more data.”

**Q3 — Structured command bottleneck vs fair aux baselines?**  
**Yes.** EAGM-Base beats ObsMLP (+9.4e-4, 5/5) and AuxCommandMLP (+7.4e-4, 5/5) at similar capacity; Dir adds further lift over Aux (+1.26e-3, 5/5). Gains are not reducible to “extra u_exec labels alone.”

**Q4 — Gradient improvement on the policy path?**  
**Mostly yes early/mid.** Path Δcos at θ0/mid positive (5/5 and 4/5); θ_final mixed. True update returns still favor Dir 5/5. Not probe-only.

**Q5 — Train-time B randomization for unseen configs?**  
**Partial.** Fixes catastrophic OOD NMSE; Dir>Base holds on interp & extrap; Hist+Dir vs Mc on OOD return is **not** reliably positive (3/5). Extrap harder than interp. No CF loss yet.

**Q6 — Contribution scope / what would support public-benchmark claims?**  
Current scope: **partially unknown execution (Controller B), low-dim proprioceptive OpenArm, fixed and mildly randomized B**, with confirmatory + budget-matched evidence that Matched-θ Dir improves policy optimization beyond paired Base and beyond fair aux-command MLPs.

Still **missing** for public robot-benchmark leadership: multi-task / visual observations, larger seeds, stronger external baselines (Dreamer-style, TD-MPC, etc.), hardware or standard sim suites (ManiSkill/Isaac), and reliable OOD wins vs Mc-class memory models. Do **not** claim SOTA.

## Stop-condition check

| Condition | Outcome |
|-----------|---------|
| 5A confirmatory Dir−Base fails | **Not triggered** (5/5) |
| Budget-matched Base ≥ Dir | **Not triggered** (Dir wins 5/5 vs B) |
| Path gradients unrelated to returns | **Not triggered** (early/mid path + returns align) |
| Extreme seed/trajectory sensitivity | Mild (effect sizes vary; sign stable) |
| Bottleneck ≤ aux-command baseline | **Not triggered** |

→ Stage 5E was allowed and run. Further CF-loss work remains **optional**, not forced.

## Level assessment

EAGM now has **confirmed** (independent seeds) method evidence on fixed-B Dir-Matched policy gains, plus budget-fairness and structured-baseline support. OOD is improved by train-time randomization but not solved vs Mc. Ready for careful paper drafting of a **narrow** execution-gap / gradient-consistency claim — not vision SOTA expansion.

## Reproduce

```bash
conda activate eagm_gpu
export OPENARM_MODEL_PATH=.../openarm_bimanual.xml
bash scripts/run_stage5_all.sh          # 5A→5D
python scripts/run_stage5e_ood.py       # after 5A–5C pass
python scripts/write_stage5_reports.py
```

Raw JSON/CSV/checkpoints under `runs/stage5{a_confirm,b_budget,c_baselines,d_gradpath,e_ood}/`.
