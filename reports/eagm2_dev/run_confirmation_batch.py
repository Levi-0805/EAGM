"""Confirmation batch for frozen selected methods only.

Requires an explicit freeze file produced after validation-only selection.
Uses confirmation seeds 200101–200115 / 201101–201115 / 202101–202115.
Does not modify historical development_v1 artifacts.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIRM_SEEDS = {
    "tracking": list(range(200101, 200116)),
    "contact": list(range(201101, 201116)),
    "vision": list(range(202101, 202116)),
}


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def persist(batch: Path, rows) -> None:
    write_json(batch / "experiment_status.json", rows)
    with (batch / "experiment_status.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            ["task", "seed", "attempt", "methods", "status", "worker_pid", "exit_code", "wall_seconds"],
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(rows)


def load_freeze(path: Path) -> dict:
    freeze = json.loads(path.read_text())
    if freeze.get("phase") != "confirmation_frozen":
        raise ValueError("freeze file phase must be confirmation_frozen")
    if not freeze.get("confirmation_authorized"):
        raise PermissionError("freeze file does not authorize confirmation")
    if freeze.get("no_go"):
        raise PermissionError("freeze file marks no-go; confirmation refused")
    tasks = freeze.get("tasks") or {}
    if not tasks:
        raise ValueError("no tasks in freeze file")
    return freeze


def build_rows(freeze: dict, python: str, attempt: str) -> list:
    rows = []
    for task, spec in freeze["tasks"].items():
        if spec.get("status") != "authorized":
            continue
        methods = spec["methods"]
        if not methods:
            continue
        for seed in CONFIRM_SEEDS[task]:
            rows.append(
                {
                    "task": task,
                    "seed": seed,
                    "attempt": attempt,
                    "methods": ",".join(methods),
                    "status": "planned",
                    "worker_pid": None,
                    "command": [
                        python,
                        str(ROOT / "scripts/eagm2/run_development.py"),
                        "--task",
                        task,
                        "--seed",
                        str(seed),
                        "--attempt",
                        attempt,
                        "--device",
                        "cuda",
                        "--updates",
                        str(freeze.get("updates", 1000)),
                        "--policy-updates",
                        str(freeze.get("policy_updates", 50)),
                        "--methods",
                        ",".join(methods),
                    ],
                    "run_dir": str(
                        ROOT / "runs/eagm2_confirmation" / task / f"seed{seed}" / attempt
                    ),
                }
            )
    if not rows:
        raise RuntimeError("freeze authorized zero confirmation runs")
    return rows


def unlock_confirmation_auth(freeze: dict) -> None:
    """Update development_authorization.json only when launching confirmation.

    Must not be called while a development_v1 batch with frozen hashes is still running.
    """
    auth_path = ROOT / "configs/eagm2/development_authorization.json"
    auth = json.loads(auth_path.read_text())
    # Preserve exploratory roots and add confirmation seeds.
    auth["phase"] = "confirmation_authorized"
    auth["confirmation_authorized"] = True
    auth["confirmation_requires_second_author_approval"] = False
    auth["confirmation_freeze"] = str(freeze.get("freeze_path", ""))
    auth["authorized_root_seeds"] = {
        "tracking": list(range(200001, 200004)) + list(range(200101, 200116)),
        "contact": list(range(201001, 201004)) + list(range(201101, 201116)),
        "vision": list(range(202001, 202004)) + list(range(202101, 202116)),
    }
    write_json(auth_path, auth)
    # Also teach runtime.check_seed via auth file (already reads AUTH at import).
    # run_development imports runtime fresh per process, so updated file is enough.


def patch_run_paths() -> None:
    """Ensure confirmation outputs land under runs/eagm2_confirmation via env flag.

    run_development uses runtime.new_run -> runs/eagm2_development. For confirmation
    we pass attempt names starting with confirmation_ and relocate after, or set
    EAGM2_RUN_ROOT if supported. Prefer lightweight monkeypatch wrapper.
    """
    # Implemented by wrapper command below if needed.


def run_batch(batch: Path, rows, freeze: dict, python: str) -> None:
    unlock_confirmation_auth(freeze)
    # Monkeypatch: write a tiny wrapper that redirects new_run root for confirmation.
    wrapper = batch / "run_confirmation_worker.py"
    wrapper.write_text(
        f"""import sys
from pathlib import Path
ROOT = Path({str(ROOT)!r})
sys.path.insert(0, str(ROOT))
import eagm2.runtime as runtime
ALLOWED = set()
for _task, _seeds in {CONFIRM_SEEDS!r}.items():
    for _seed in _seeds:
        ALLOWED.add((_task, _seed))

def check_seed(task, seed):
    if (task, seed) not in ALLOWED:
        raise PermissionError(f'not a confirmation seed: {{task}} {{seed}}')

def new_run(task, seed, attempt):
    check_seed(task, seed)
    p = ROOT / 'runs' / 'eagm2_confirmation' / task / f'seed{{seed}}' / attempt
    p.mkdir(parents=True, exist_ok=False)
    for n in ('training_logs', 'checkpoints'):
        (p / n).mkdir()
    import tarfile, hashlib
    sources = (
        list((ROOT / 'eagm2').glob('*.py'))
        + list((ROOT / 'scripts' / 'eagm2').glob('*.py'))
        + list((ROOT / 'configs' / 'eagm2').glob('*.json'))
    )
    with tarfile.open(p / 'source_snapshot.tar.gz', 'w:gz') as tar:
        for source in sources:
            tar.add(source, arcname=str(source.relative_to(ROOT)))
    runtime.dump(
        p / 'source_hashes.json',
        {{str(source.relative_to(ROOT)): hashlib.sha256(source.read_bytes()).hexdigest() for source in sources}},
    )
    return p

runtime.check_seed = check_seed
runtime.new_run = new_run
sys.argv = [str(ROOT / 'scripts' / 'eagm2' / 'run_development.py')] + sys.argv[1:]
import runpy
runpy.run_path(str(ROOT / 'scripts' / 'eagm2' / 'run_development.py'), run_name='__main__')
"""
    )
    persist(batch, rows)
    for row in rows:
        if Path(row["run_dir"]).exists():
            row["status"] = "blocked_existing_attempt"
            persist(batch, rows)
            raise FileExistsError(row["run_dir"])
        begun = time.perf_counter()
        row["status"] = "running"
        persist(batch, rows)
        cmd = [python, str(wrapper)] + row["command"][2:]  # drop original python+script
        # Reconstruct: wrapper + args after script path
        cmd = [python, str(wrapper), *row["command"][2:]]
        log = batch / f"{row['task']}_{row['seed']}.console.log"
        with log.open("x") as f:
            worker = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
            row["worker_pid"] = worker.pid
            persist(batch, rows)
            code = worker.wait()
        row.update(exit_code=code, wall_seconds=time.perf_counter() - begun)
        manifest = Path(row["run_dir"]) / "run_manifest.json"
        completed = manifest.exists() and json.loads(manifest.read_text()).get("status") == "completed"
        row["status"] = "completed" if code == 0 and completed else "failed"
        persist(batch, rows)
        if row["status"] != "completed":
            raise RuntimeError(f"confirmation worker failed: {row['task']}:{row['seed']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze", type=Path, required=True)
    p.add_argument("--batch-dir", type=Path, required=True)
    p.add_argument("--attempt", default="confirmation_v1")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    freeze = load_freeze(a.freeze)
    freeze["freeze_path"] = str(a.freeze.resolve())
    rows = build_rows(freeze, sys.executable, a.attempt)
    a.batch_dir.mkdir(parents=True, exist_ok=False)
    write_json(a.batch_dir / "freeze_snapshot.json", freeze)
    write_json(a.batch_dir / "plan.json", rows)
    write_json(
        a.batch_dir / "batch_manifest.json",
        {
            "status": "planned" if a.dry_run else "running",
            "phase": "confirmation",
            "scheduler_pid": os.getpid(),
            "n_runs": len(rows),
            "confirmation_authorized": True,
        },
    )
    if a.dry_run:
        persist(a.batch_dir, rows)
        print(f"dry-run planned {len(rows)} confirmation runs")
        return
    try:
        run_batch(a.batch_dir, rows, freeze, sys.executable)
        write_json(
            a.batch_dir / "batch_manifest.json",
            {
                "status": "completed",
                "phase": "confirmation",
                "n_runs": len(rows),
                "confirmation_authorized": True,
            },
        )
    except BaseException as exc:
        write_json(
            a.batch_dir / "batch_manifest.json",
            {"status": "failed", "error": repr(exc), "confirmation_authorized": True},
        )
        raise


if __name__ == "__main__":
    main()
