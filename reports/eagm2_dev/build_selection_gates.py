"""Build capacity/budget/tests gate attestations from completed development roots.

Uses only completed development_v1 results. Does not read test-primary outcomes
for selection criteria (those remain in evaluate_method_selection.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEEDS = {
    "tracking": (200001, 200002, 200003),
    "contact": (201001, 201002, 201003),
    "vision": (202001, 202002, 202003),
}
CANDIDATES = ("delay_dir", "recurrent_dir", "combined_dir")
BASELINES = ("ff_dir", "legacy_dir")


def read_params(run: Path, method: str):
    result = json.loads((run / method / "results.json").read_text())
    counts = result.get("parameter_counts") or {}
    budget = result.get("training_simulator_budget") or {}
    return int(counts.get("trainable", -1)), budget


def capacity_status(cand: int, base: int) -> dict:
    if cand <= 0 or base <= 0:
        return {"status": "fail", "evidence": f"missing parameter counts cand={cand} base={base}"}
    ratio = abs(cand - base) / base
    # Prefer within ±5%; larger recurrent/combined models must be disclosed as fail for
    # capacity-matched claims against ff_dir (they remain eligible vs capacity_* controls).
    status = "pass" if ratio <= 0.05 else "fail"
    return {
        "status": status,
        "evidence": f"trainable cand={cand} baseline={base} relative_diff={ratio:.6f} threshold=0.05",
    }


def budget_status(cand_budget: dict, base_budget: dict) -> dict:
    ck = int(cand_budget.get("collection", -1)) + int(cand_budget.get("gradient_labels", 0))
    bk = int(base_budget.get("collection", -1)) + int(base_budget.get("gradient_labels", 0))
    if ck < 0 or bk < 0:
        return {"status": "fail", "evidence": f"missing budgets cand={cand_budget} base={base_budget}"}
    # Same common collection + FD label protocol => pass; unequal => fail for fair claim.
    status = "pass" if ck == bk else "fail"
    return {
        "status": status,
        "evidence": f"training_query_units cand={ck} baseline={bk} (collection+gradient_labels)",
    }


def tests_status(tests_log: Path) -> dict:
    text = tests_log.read_text()
    ok = ("OK" in text.splitlines()[-5:]) or ("\nOK\n" in text) or text.rstrip().endswith("OK")
    status = "pass" if ok and "FAILED" not in text and "ERROR:" not in text else "fail"
    return {"status": status, "evidence": str(tests_log.resolve())}


def build(attempt: str, tests_log: Path) -> dict:
    gates: dict = {}
    test_gate = tests_status(tests_log)
    for task, seeds in SEEDS.items():
        gates[task] = {}
        for baseline in BASELINES:
            if task == "vision" and baseline == "legacy_dir":
                continue
            gates[task][baseline] = {}
            for candidate in CANDIDATES:
                # Aggregate across roots; require every completed root comparable.
                cap_notes = []
                bud_notes = []
                missing = []
                for seed in seeds:
                    run = ROOT / "runs/eagm2_development" / task / f"seed{seed}" / attempt
                    try:
                        c_params, c_bud = read_params(run, candidate)
                        b_params, b_bud = read_params(run, baseline)
                    except (OSError, ValueError, KeyError, TypeError) as exc:
                        missing.append({"seed": seed, "error": repr(exc)})
                        continue
                    cap_notes.append(capacity_status(c_params, b_params))
                    bud_notes.append(budget_status(c_bud, b_bud))
                if missing or len(cap_notes) != 3:
                    cap = {"status": "fail", "evidence": f"incomplete roots: {missing}"}
                    bud = {"status": "fail", "evidence": f"incomplete roots: {missing}"}
                else:
                    # All roots must pass the same gate.
                    cap = {
                        "status": "pass" if all(x["status"] == "pass" for x in cap_notes) else "fail",
                        "evidence": " | ".join(x["evidence"] for x in cap_notes),
                    }
                    bud = {
                        "status": "pass" if all(x["status"] == "pass" for x in bud_notes) else "fail",
                        "evidence": " | ".join(x["evidence"] for x in bud_notes),
                    }
                gates[task][baseline][candidate] = {
                    "capacity": cap,
                    "budget": bud,
                    "tests": test_gate,
                }
    return gates


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attempt", default="development_v1")
    p.add_argument("--tests-log", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    gates = build(a.attempt, a.tests_log)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.output.with_suffix(a.output.suffix + ".tmp")
    tmp.write_text(json.dumps(gates, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(a.output)
    print(a.output)


if __name__ == "__main__":
    main()
