#!/usr/bin/env python3
"""Compare CPU vs CUDA throughput for EAGM-sized training (no protocol change)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def bench(device, steps=500, batch=128):
    m = nn.Sequential(
        nn.Linear(42, 160), nn.SiLU(), nn.Linear(160, 160), nn.SiLU(), nn.Linear(160, 28)
    ).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    x = torch.randn(batch, 42, device=device)
    y = torch.randn(batch, 28, device=device)
    for _ in range(20):
        opt.zero_grad(set_to_none=True)
        ((m(x) - y) ** 2).mean().backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        ((m(x) - y) ** 2).mean().backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    peak = None
    if device.type == "cuda":
        peak = round(torch.cuda.max_memory_allocated() / 1e6, 2)
    return dict(device=str(device), steps=steps, seconds=round(dt, 3), updates_per_s=round(steps / dt, 1), peak_mem_mb=peak)


def main():
    out = {
        "cuda_available": torch.cuda.is_available(),
        "cpu": bench(torch.device("cpu")),
    }
    if torch.cuda.is_available():
        out["cuda"] = bench(torch.device("cuda"))
        out["recommendation"] = (
            "For Stage-3.1 OpenArm MLP + MuJoCo FD labels, CPU training is often comparable; "
            "use CUDA when batch/history/visual encoders grow. Protocol runs use CPU FP32."
        )
    path = ROOT / "reports" / "stage3_gpu_benchmark.md"
    path.write_text(
        "# Stage 3 GPU/CPU benchmark\n\n"
        f"```json\n{json.dumps(out, indent=2)}\n```\n\n"
        "MuJoCo stepping remains CPU-bound. Stage 3.1 protocol training uses **CPU FP32** "
        "to keep imagined/dir graphs device-consistent with snapshot labels.\n"
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
