"""Wait for development_v1, then postrun → gates → selection → freeze → optional confirmation.

Safe to launch while the resume scheduler is still running. Does not modify hashed
training sources until the development batch has finished.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/eagm2_dev"
BATCH = ROOT / "runs/eagm2_batches/development_v1"
PY = "/home/aaa/miniconda3/envs/eagm_gpu/bin/python"
STATE = REPORT / "continue_pipeline_state.json"
EXPECTED = {
    ("tracking", 200001),
    ("tracking", 200002),
    ("tracking", 200003),
    ("contact", 201001),
    ("contact", 201002),
    ("contact", 201003),
    ("vision", 202001),
    ("vision", 202002),
    ("vision", 202003),
}


def save(obj) -> None:
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(STATE)


def run(cmd, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x") as f:
        return subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode


def wait_for_batch(state: dict) -> list:
    state.update(status="waiting_for_development_batch", pid=os.getpid(), started_unix=time.time())
    save(state)
    while True:
        rows = json.loads((BATCH / "experiment_status.json").read_text())
        manifest = json.loads((BATCH / "batch_manifest.json").read_text())
        done = {(r["task"], r["seed"]) for r in rows if r.get("status") == "completed"}
        state.update(
            completed_roots=len(done),
            batch_status=manifest.get("status"),
            checked_unix=time.time(),
        )
        save(state)
        if done == EXPECTED and manifest.get("status") in {"completed", "analysis_failed"}:
            return rows
        # Also accept completed if analysis not yet done but all roots completed and scheduler gone.
        if done == EXPECTED and not _scheduler_alive():
            return rows
        if time.time() - state["started_unix"] > 72 * 3600:
            raise TimeoutError("development batch wait exceeded 72h")
        time.sleep(60)


def _scheduler_alive() -> bool:
    out = subprocess.run(
        ["pgrep", "-f", "resume_authorized_batch.py --batch-dir runs/eagm2_batches/development_v1"],
        capture_output=True,
        text=True,
    )
    return out.returncode == 0


def postrun(state: dict) -> Path:
    state["status"] = "postrun_analysis"
    save(state)
    out = REPORT / "development_v1_postrun_analysis"
    if out.exists():
        # Keep first successful analysis; create versioned sibling if needed.
        out = REPORT / "development_v1_postrun_analysis_continue"
    code = run(
        [
            PY,
            str(REPORT / "analyze_development_postrun.py"),
            "--attempt",
            "development_v1",
            "--output-dir",
            str(out),
            "--report-tag",
            "development_v1_continue",
        ],
        REPORT / "continue_postrun.console.log",
    )
    state["postrun_exit_code"] = code
    save(state)
    if code:
        raise RuntimeError("postrun analysis failed")
    return out


def selection(state: dict) -> tuple[Path, Path]:
    state["status"] = "selection"
    save(state)
    gates_path = REPORT / "selection_gates_development_v1.json"
    code = run(
        [
            PY,
            str(REPORT / "build_selection_gates.py"),
            "--attempt",
            "development_v1",
            "--tests-log",
            str(REPORT / "tests_full_discovery_final.log"),
            "--output",
            str(gates_path),
        ],
        REPORT / "continue_build_gates.console.log",
    )
    if code:
        raise RuntimeError("gate construction failed")
    sel_path = REPORT / "development_method_selection.json"
    if sel_path.exists():
        sel_path = REPORT / "development_method_selection_continue.json"
    code = run(
        [
            PY,
            str(REPORT / "evaluate_method_selection.py"),
            "--attempt",
            "development_v1",
            "--gates",
            str(gates_path),
            "--output",
            str(sel_path),
        ],
        REPORT / "continue_selection.console.log",
    )
    state["selection_exit_code"] = code
    save(state)
    if code:
        raise RuntimeError("selection failed")
    return gates_path, sel_path


def freeze_from_selection(sel_path: Path, state: dict) -> Path:
    sel = json.loads(sel_path.read_text())
    tasks = {}
    any_authorized = False
    for task, baselines in sel.get("tasks", {}).items():
        # Prefer ff_dir baseline resolution path; vision has no legacy_dir.
        preferred = "ff_dir" if "ff_dir" in baselines else next(iter(baselines))
        entry = baselines[preferred]
        selected = entry.get("selected")
        status = entry.get("status")
        if status == "pass" and selected:
            # Confirm selected candidate + Dir baseline only (primary comparison).
            methods = [selected, preferred]
            # Keep unique order.
            seen = set()
            methods = [m for m in methods if not (m in seen or seen.add(m))]
            tasks[task] = {
                "status": "authorized",
                "selected": selected,
                "baseline": preferred,
                "methods": methods,
                "selection_status": status,
            }
            any_authorized = True
        else:
            tasks[task] = {
                "status": "no_go" if status == "no_go" else "not_authorized",
                "selected": selected,
                "baseline": preferred,
                "methods": [],
                "selection_status": status,
            }
    freeze = {
        "phase": "confirmation_frozen",
        "confirmation_authorized": bool(any_authorized),
        "no_go": not any_authorized,
        "attempt_source": "development_v1",
        "selection_path": str(sel_path),
        "updates": 1000,
        "policy_updates": 50,
        "primary_rule": "selected vs Dir baseline on confirmatory seeds; validation-only selection already applied",
        "author_authorization": str(REPORT / "author_continue_authorization.json"),
        "tasks": tasks,
        "created_unix": time.time(),
    }
    path = REPORT / "confirmation_freeze_v1.json"
    if path.exists():
        path = REPORT / "confirmation_freeze_v1_continue.json"
    path.write_text(json.dumps(freeze, indent=2, ensure_ascii=False) + "\n")
    # Human-readable handoff
    lines = [
        "# Confirmation freeze",
        "",
        f"- selection: `{sel_path}`",
        f"- confirmation_authorized: `{freeze['confirmation_authorized']}`",
        f"- no_go: `{freeze['no_go']}`",
        "",
        "| Task | Status | Selected | Baseline | Methods |",
        "|---|---|---|---|---|",
    ]
    for task, spec in tasks.items():
        lines.append(
            f"| {task} | {spec['status']} | {spec.get('selected')} | {spec.get('baseline')} | {','.join(spec.get('methods') or [])} |"
        )
    lines += [
        "",
        "If no_go, confirmation seeds will not launch. Negative exploratory evidence is retained.",
    ]
    (REPORT / "confirmation_freeze_v1.md").write_text("\n".join(lines) + "\n")
    state["freeze_path"] = str(path)
    state["confirmation_authorized"] = freeze["confirmation_authorized"]
    state["no_go"] = freeze["no_go"]
    save(state)
    return path


def maybe_confirm(freeze_path: Path, state: dict) -> None:
    freeze = json.loads(freeze_path.read_text())
    os.environ.setdefault('OPENARM_MODEL_PATH', '/home/aaa/HIL-OpenArm/third_party/openarm_mujoco/v2/openarm_bimanual.xml')
    if freeze.get("no_go") or not freeze.get("confirmation_authorized"):
        state["status"] = "completed_no_go"
        state["note"] = "No confirmation launched; scientific selection failed or pending."
        save(state)
        (REPORT / "06_development_handoff_continue.md").write_text(
            "# Development continue handoff — no-go\n\n"
            "Nine exploratory roots finished. Validation-only selection did not authorize confirmation.\n"
            "Read postrun analysis and selection JSON. Do not treat exploratory means as confirmatory.\n"
        )
        return
    state["status"] = "confirmation_running"
    save(state)
    batch = ROOT / "runs/eagm2_batches/confirmation_v1"
    code = run(
        [
            PY,
            str(REPORT / "run_confirmation_batch.py"),
            "--freeze",
            str(freeze_path),
            "--batch-dir",
            str(batch),
            "--attempt",
            "confirmation_v1",
        ],
        REPORT / "continue_confirmation.console.log",
    )
    state["confirmation_exit_code"] = code
    if code:
        state["status"] = "confirmation_failed"
        save(state)
        raise RuntimeError("confirmation batch failed; partial results retained")
    state["status"] = "confirmation_completed"
    save(state)


def write_progress_summary(state: dict) -> None:
    (REPORT / "continue_pipeline_status.md").write_text(
        "# Continue pipeline status\n\n"
        f"- status: `{state.get('status')}`\n"
        f"- completed_roots: `{state.get('completed_roots')}`\n"
        f"- confirmation_authorized: `{state.get('confirmation_authorized')}`\n"
        f"- no_go: `{state.get('no_go')}`\n"
        f"- freeze: `{state.get('freeze_path')}`\n"
        f"- state_json: `{STATE}`\n"
    )


def main() -> None:
    os.environ.setdefault('OPENARM_MODEL_PATH', '/home/aaa/HIL-OpenArm/third_party/openarm_mujoco/v2/openarm_bimanual.xml')
    if STATE.exists():
        existing = json.loads(STATE.read_text())
        if existing.get("status") not in {"needs_review", "confirmation_failed"} and existing.get("pid"):
            # Allow explicit restart if previous pid dead.
            try:
                os.kill(int(existing["pid"]), 0)
                raise SystemExit(f"continue pipeline already running pid={existing['pid']}")
            except OSError:
                pass
    state = {
        "status": "starting",
        "pid": os.getpid(),
        "author_authorization": str(REPORT / "author_continue_authorization.json"),
    }
    save(state)
    try:
        wait_for_batch(state)
        postrun(state)
        _, sel = selection(state)
        freeze = freeze_from_selection(sel, state)
        maybe_confirm(freeze, state)
        write_progress_summary(state)
    except BaseException as exc:
        state.update(status="needs_review", error=repr(exc), finished_unix=time.time())
        save(state)
        write_progress_summary(state)
        raise


if __name__ == "__main__":
    main()
