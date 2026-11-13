#!/usr/bin/env python3
"""Stage 7A — freeze Dir-Random method; audit Stage 6C recoverability + pilot doc."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

EAGM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EAGM_ROOT))
sys.path.insert(0, str(EAGM_ROOT / "vendor" / "legacy_sim"))

from models.baselines import AuxCommandMLP, count_params
from models.eagm import EAGMBase, McWorld
from training.paired_train import clone_model
from training.stage6c_common import PARENT_COMMIT as STAGE6C_PARENT

OUT = EAGM_ROOT / "runs" / "stage7a_baseline_audit"
REF = "43cfc79c0c38c9e2a322f7f4a75d8a100dc04aea"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=EAGM_ROOT).decode().strip()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    frozen = json.loads((EAGM_ROOT / "configs" / "stage7_frozen.json").read_text())
    c2 = json.loads((EAGM_ROOT / "runs" / "stage6c2_pathdir" / "results.json").read_text())
    c2_proto = json.loads((EAGM_ROOT / "runs" / "stage6c2_pathdir" / "protocol.json").read_text())
    audit0 = json.loads((EAGM_ROOT / "runs" / "stage6c0_audit" / "audit.json").read_text())
    sat_md = (EAGM_ROOT / "reports" / "stage6c_saturation_analysis.md").read_text()
    path_md = (EAGM_ROOT / "reports" / "stage6c_pathdir_results.md").read_text()
    src_6c2 = (EAGM_ROOT / "scripts" / "run_stage6c2_pathdir.py").read_text()

    # --- Path-Dir pilot discrepancy ---
    pilot_uses_base = "path_probe_thetas_from_pilot(\n        base," in src_6c2 or (
        "path_probe_thetas_from_pilot(" in src_6c2 and "Train Base alone" in src_6c2
    )
    sat_had_dir_wording = "frozen Dir" in sat_md
    sat_clarified = "Path-Dir label" in sat_md and "trained Base" in sat_md
    path_says_base = "frozen trained Base" in path_md or "Pilot on frozen" in src_6c2

    pilot_resolution = dict(
        stage6c2_code_pilot_model="trained_Base",
        stage6c0_audit_path_probes_model="stage6b1_Dir_checkpoint_for_diagnostics_only",
        saturation_report_originally_said_frozen_Dir=sat_had_dir_wording,
        saturation_report_clarified=sat_clarified,
        pathdir_results_consistent_with_code=path_says_base or "Base" in path_md,
        paper_safe_statement=(
            "Path-Dir labels in Stage 6C.2/6C.3 were generated from a policy-optimization "
            "pilot on a frozen trained EAGM-Base; Stage 6C.0 used a Dir checkpoint only "
            "for saturation/projection smoke diagnostics."
        ),
        resolved=bool(pilot_uses_base and sat_clarified),
    )

    # --- Artifact recoverability ---
    required = [
        EAGM_ROOT / "runs/stage6c2_pathdir/results.json",
        EAGM_ROOT / "runs/stage6c2_pathdir/protocol.json",
        EAGM_ROOT / "runs/stage6c1_lr_ablation/results.json",
        EAGM_ROOT / "runs/stage6c_manifest.json",
        EAGM_ROOT / "reports/stage6c_research_decision.md",
        EAGM_ROOT / "tests/test_stage6c_fd_boundary.py",
    ]
    artifacts = {}
    for p in required:
        artifacts[str(p.relative_to(EAGM_ROOT))] = dict(
            exists=p.exists(),
            sha256=sha256_file(p) if p.exists() else None,
            bytes=p.stat().st_size if p.exists() else 0,
        )

    # Per-seed JSON present (checkpoints .pt gitignored — noted)
    seed_json = {}
    for s in range(15, 20):
        rp = EAGM_ROOT / f"runs/stage6c2_pathdir/seed{s}/results.json"
        seed_json[f"seed{s}"] = dict(exists=rp.exists(), has_pt=(EAGM_ROOT / f"runs/stage6c2_pathdir/seed{s}/eagm_dir_random.pt").exists())

    # --- Reproduce paired-stats numbers from raw seed contrasts ---
    diffs = [c2["seed_results"][i]["contrasts"]["interp_path_minus_random"] for i in range(5)]
    # seed_results may be compact list
    if isinstance(c2["seed_results"][0], dict) and "contrasts" in c2["seed_results"][0]:
        diffs = [s["contrasts"]["interp_path_minus_random"] for s in c2["seed_results"]]
    reported = c2["paired_seed_stats"]["interp_path_minus_random"]
    recompute = dict(
        mean=float(np.mean(diffs)),
        n_positive=int(np.sum(np.asarray(diffs) > 0)),
        matches_report_mean=abs(float(np.mean(diffs)) - reported["mean"]) < 1e-12,
        matches_report_n_pos=int(np.sum(np.asarray(diffs) > 0)) == reported["n_positive"],
        decision=c2["decision_gate"]["decision"],
    )

    # --- Frozen model inventory (constructible) ---
    dummy_stats = dict(
        xmean=np.zeros(28), xstd=np.ones(28),
        umean=np.zeros(14), ustd=np.ones(14),
        ymean=np.zeros(28), ystd=np.ones(28),
        lower=np.full(14, -2.0), upper=np.full(14, 2.0), alpha=np.asarray(0.5),
    )
    models = {
        "mc": McWorld(dummy_stats),
        "eagm_base": EAGMBase(dummy_stats),
        "eagm_dir_random": clone_model(EAGMBase(dummy_stats)),
        "aux_command_mlp": AuxCommandMLP(dummy_stats),
    }
    model_info = {k: dict(kind=getattr(m, "kind", k), n_params=count_params(m)) for k, m in models.items()}

    # Env versions
    try:
        import mujoco
        mj_ver = mujoco.__version__
    except Exception as e:
        mj_ver = f"error:{e}"

    payload = dict(
        stage="7A",
        git_head=git_head(),
        reference_commit=REF,
        stage6c_parent_recorded=STAGE6C_PARENT,
        platform=dict(
            python=sys.version,
            platform=platform.platform(),
            torch=torch.__version__,
            cuda=torch.cuda.is_available(),
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            mujoco=mj_ver,
        ),
        frozen_method=frozen["frozen_method"],
        path_dir_pilot_resolution=pilot_resolution,
        artifacts=artifacts,
        seed_json=seed_json,
        stage6c2_protocol=c2_proto,
        stage6c0_status=audit0.get("status"),
        recompute_path_minus_random=recompute,
        model_inventory=model_info,
        notes=[
            ".pt checkpoints remain gitignored; local seed*.pt exist for restore when present.",
            "Do not launch Stage 7D multi-seed until 7B unit tests + 7C single-seed gate pass.",
            "Dir-Random is the frozen primary method; Path-Dir remains STOP.",
        ],
        wall_seconds=time.time() - t0,
        status="PASS" if pilot_resolution["resolved"] and recompute["matches_report_mean"] and all(a["exists"] for a in artifacts.values()) else "FAIL",
    )
    (OUT / "audit.json").write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps(payload, indent=2, default=str))
    print("status", payload["status"], flush=True)


if __name__ == "__main__":
    main()
