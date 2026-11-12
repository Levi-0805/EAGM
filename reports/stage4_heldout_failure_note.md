# Held-out Controller B — failure attribution (brief)

Stage 3.1 held-out (randomized B, models trained on fixed B):

- EAGM dyn NMSE ≈ 2.4–2.8 (vs ~0.17–0.30 iid)
- No method beat Mc on held-out return

Likely causes (not mutually exclusive):

1. **Train/test controller mismatch:** fixed (gain, lag, rate, smooth) vs random — distribution shift in u dynamics.  
2. **Command residual under-specified for OOD lag/gain** without seeing those configs in training.  
3. **History encoder** improves iid lag identification but cannot invent unseen gain/rate maps.  
4. **Policy θ** optimized under fixed-B imagined dynamics transfers poorly when plant B changes.

This is **not** evidence that matched-θ Dir is useless on the training controller family; Stage 4 fixed-B five-seed supports Dir there. OOD requires explicit randomization during data collection.
