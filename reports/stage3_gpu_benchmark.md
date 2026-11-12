# Stage 3 GPU/CPU benchmark

```json
{
  "cuda_available": true,
  "cpu": {
    "device": "cpu",
    "steps": 500,
    "seconds": 1.564,
    "updates_per_s": 319.6,
    "peak_mem_mb": null
  },
  "cuda": {
    "device": "cuda",
    "steps": 500,
    "seconds": 1.115,
    "updates_per_s": 448.5,
    "peak_mem_mb": 19.02
  },
  "recommendation": "For Stage-3.1 OpenArm MLP + MuJoCo FD labels, CPU training is often comparable; use CUDA when batch/history/visual encoders grow. Protocol runs use CPU FP32."
}
```

MuJoCo stepping remains CPU-bound. Stage 3.1 protocol training uses **CPU FP32** to keep imagined/dir graphs device-consistent with snapshot labels.
