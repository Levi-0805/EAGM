# Stage 6 — GitHub Issue / Task Checklist

Branch: `stage6-exec-generalization` (from Stage 5 `6f66823`).  
Do **not** overwrite Stage 5 frozen artifacts under `runs/stage5*`.

## Issue 1 — Stage 6A protocol repair (P0 · now)

- [x] Branch `stage6-exec-generalization`
- [x] Fix paired fail rate (`ΔR_i = R_final − R_θ0`)
- [x] Separate θ-param saturation vs action clip
- [x] Full vs stable-mask gradient metrics API
- [x] Budget: train 24 / val 6 / IID 6 / FD diagnostic
- [x] Unit tests `tests/test_stage6a_metrics.py`
- [x] OOD config list + compound configs `configs/stage6_ood_configs.json`
- [x] Architecture doc update
- [x] Re-eval Stage 5E with corrected metrics → `runs/stage6a_stage5e_corrected/`
- [x] Write `reports/stage5e_corrected_stats.md`
- [x] Push branch (`f80c69d` protocol; tip `bf99aff` checklist)

**Reproduce**
```bash
conda activate eagm_gpu
export OPENARM_MODEL_PATH=.../openarm_bimanual.xml
python -c "import tests.test_stage6a_metrics as t; ..."  # or install pytest
python scripts/plot_stage6a_saturation.py
python scripts/run_stage6a_reeval_stage5e.py
```

## Issue 2 — Stage 6B.1 reproduce Stage 5E on seeds 10–14 (P0)

- [x] Freeze Stage 6A metrics; seeds 10–14; Stage 5E legacy (2 eps/config).
- [x] Five models unchanged; no HistC/ExecState.
- [x] Reports: `stage6b1_results.md`, `stage6b1_analysis.md`; **stop for human review** (no auto 6B.2).
- **Outcome (Case C):** Dir−Base **5/5** interp+extrap; Hist+Dir−Mc **3/5** interp (CI includes 0). θ-sat Dir≈0.89 remains high.

## Issue 3 — Stage 6B.2 HistC ablation (P0)

- Add `c_obs` history to encoder; k ∈ {4,8,16}.
- Compare Base / Dir / Hist / Hist+Dir / HistC+Dir / Mc.

## Issue 4 — Stage 6B.3 ExecState prototype (P1 after HistC)

- Latent z + optional privileged state aux loss (train-only labels).
- No CF loss yet.

## Issue 5 — Stage 6C path-aware Dir + saturation (P1)

- Path θ sampling with matched FD budget.
- LR ablation 0.03 / 0.015 / 0.0075; val-based early stop.

## Issue 6 — Stage 6D confirm seeds 20–29 + task transfer (gated)

- Only after 6B/6C pass pre-registered criteria.
