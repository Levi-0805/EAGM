"""Resume an interrupted nine-root exploratory batch without deleting evidence.

Rules:
- Completed roots are left untouched.
- An incomplete root directory is renamed to *_interrupted_* (never deleted).
- Remaining planned roots are executed under the same attempt name.
- Source hashes must still match the original batch freeze.
- No confirmation seeds are launched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'eagm2'))
from run_authorized_batch import persist, source_hashes, write_json  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive_incomplete(run_dir: Path, reason: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = run_dir.with_name(f"{run_dir.name}_interrupted_{stamp}")
    if dest.exists():
        raise FileExistsError(dest)
    run_dir.rename(dest)
    note = {
        "status": "interrupted",
        "reason": reason,
        "archived_from": str(run_dir),
        "archived_to": str(dest),
        "archived_unix": time.time(),
        "note": "Partial exploratory attempt retained; not a confirmation result.",
    }
    write_json(dest / "interruption_record.json", note)
    # Mark incomplete manifests if present.
    manifest = dest / "run_manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text())
            data["status"] = "interrupted"
            data["interruption"] = note
            write_json(manifest, data)
        except (OSError, ValueError):
            pass
    return dest


def is_complete(run_dir: Path) -> bool:
    manifest = run_dir / "run_manifest.json"
    results = run_dir / "results.json"
    if not (manifest.exists() and results.exists()):
        return False
    try:
        return (
            json.loads(manifest.read_text()).get("status") == "completed"
            and json.loads(results.read_text()).get("status") == "completed"
        )
    except (OSError, ValueError):
        return False


def resume(batch: Path, *, python: str, dry_run: bool = False) -> None:
    frozen = json.loads((batch / "source_hashes.json").read_text())
    current = source_hashes(ROOT)
    if current != frozen:
        raise RuntimeError("training/source files changed since batch preflight; refusing resume")

    rows = json.loads((batch / "experiment_status.json").read_text())
    state = json.loads((batch / "batch_manifest.json").read_text())
    state.update(
        status="resuming",
        resume_scheduler_pid=os.getpid(),
        resume_started_unix=time.time(),
        confirmation_authorized=False,
    )
    write_json(batch / "batch_manifest.json", state)
    persist(batch, rows)

    for row in rows:
        run_dir = Path(row["run_dir"])
        if row.get("status") == "completed" and is_complete(run_dir):
            continue

        # Dead "running" or failed incomplete root: archive and re-run.
        if run_dir.exists():
            if is_complete(run_dir):
                row.update(status="completed", exit_code=0)
                # Reconstruct wall time if missing.
                if "wall_seconds" not in row or row["wall_seconds"] is None:
                    try:
                        row["wall_seconds"] = json.loads((run_dir / "results.json").read_text()).get("wall_seconds")
                    except (OSError, ValueError, TypeError):
                        pass
                persist(batch, rows)
                continue
            archived = archive_incomplete(
                run_dir,
                reason=f"resume after status={row.get('status')} pid={row.get('worker_pid')}",
            )
            row["archived_incomplete"] = str(archived)
            console = batch / f"{row['task']}_{row['seed']}.console.log"
            if console.exists():
                console.rename(console.with_name(console.name + f".interrupted_{archived.name.split('_')[-1]}"))

        begun = time.perf_counter()
        row.update(status="running", worker_pid=None, exit_code=None, wall_seconds=None)
        persist(batch, rows)
        if dry_run:
            row.update(status="planned_resume_dry_run")
            persist(batch, rows)
            continue

        if source_hashes(ROOT) != frozen:
            row["status"] = "blocked_source_drift"
            persist(batch, rows)
            raise RuntimeError("source drift detected before launching worker")

        log_path = batch / f"{row['task']}_{row['seed']}.console.log"
        with log_path.open("x") as log:
            try:
                worker = subprocess.Popen(row["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            except Exception as exc:
                row.update(status="failed_to_launch", error=repr(exc))
                persist(batch, rows)
                raise
            row["worker_pid"] = worker.pid
            persist(batch, rows)
            try:
                code = worker.wait()
            except BaseException:
                worker.terminate()
                worker.wait()
                row.update(
                    status="interrupted",
                    exit_code=worker.returncode,
                    wall_seconds=time.perf_counter() - begun,
                )
                persist(batch, rows)
                raise
        row.update(exit_code=code, wall_seconds=time.perf_counter() - begun)
        completed = is_complete(Path(row["run_dir"]))
        row["status"] = "completed" if code == 0 and completed else "failed"
        persist(batch, rows)
        if row["status"] != "completed":
            raise RuntimeError(f"worker {row['task']}:{row['seed']} failed; later roots not launched")

    if dry_run:
        state["status"] = "validated_resume_dry_run"
        write_json(batch / "batch_manifest.json", state)
        return

    if source_hashes(ROOT) != frozen:
        raise RuntimeError("source changed before analysis; stopping")
    command = [
        str(python),
        str(ROOT / "scripts/eagm2/analyze_development.py"),
        "--attempt",
        rows[0]["attempt"],
        "--output-dir",
        str(batch / "analysis"),
        "--report-tag",
        "authorized_batch_resume",
    ]
    with (batch / "analysis_resume.console.log").open("x") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    write_json(batch / "analysis_resume_status.json", dict(command=command, exit_code=result.returncode))
    state.update(status="completed" if result.returncode == 0 else "analysis_failed", resume_finished_unix=time.time())
    write_json(batch / "batch_manifest.json", state)
    if result.returncode:
        raise RuntimeError("analysis failed; raw experiments retained")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    resume(args.batch_dir.resolve(), python=sys.executable, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
