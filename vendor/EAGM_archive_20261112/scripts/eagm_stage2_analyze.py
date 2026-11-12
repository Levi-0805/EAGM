"""Analyze the five-seed 1,000-update Stage-2 comparison.

Reports seed-level means and paired differences with start-level and
episode-level cluster bootstrap (12 starts nested in 6 trajectories).
"""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs" / "eagm_stage2_1k5_20261112"
rng = np.random.default_rng(0)
CLUSTER = np.repeat(np.arange(6), 2)


def boot(diff, unit, n=20000):
    u = np.unique(unit)
    means = np.empty(n)
    for i in range(n):
        pick = rng.choice(u, size=len(u), replace=True)
        means[i] = np.mean([diff[:, unit == c].mean() for c in pick])
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return float(diff.mean()), float(lo), float(hi), float(p)


def main():
    rows = json.loads((RUN / "results.json").read_text())["rows"]
    seeds = sorted({r["seed"] for r in rows})
    names = ["mx", "mc", "mu_base", "mu_dir"]
    ret = {n: np.array([[r["returns"] for r in rows
                        if r["seed"] == s and r["model"] == n][0]
                       for s in seeds]) for n in names}
    cos = {n: np.array([np.mean([r["cosine_median"] for r in rows
                                 if r["seed"] == s and r["model"] == n])
                        for s in seeds]) for n in names}
    out = {"per_model": {}, "pairwise": {}}
    for n in names:
        per_seed = ret[n].mean(1)
        out["per_model"][n] = {
            "return_mean": float(per_seed.mean()),
            "return_per_seed": per_seed.tolist(),
            "return_seed_sd": float(per_seed.std(ddof=1)),
            "cosine_mean": float(cos[n].mean()),
            "cosine_per_seed": cos[n].tolist(),
        }
        print(f"{n:8s} ret {per_seed.mean():+.8f} sd {per_seed.std(ddof=1):.2e} "
              f"cos {cos[n].mean():.4f}")
    for a, b in [("mu_dir", "mu_base"), ("mc", "mu_base"), ("mc", "mu_dir"),
                 ("mu_dir", "mx"), ("mu_base", "mx")]:
        d = ret[a] - ret[b]
        per_seed = d.mean(1)
        line = {"per_seed": per_seed.tolist(),
                "seeds_positive": int((per_seed > 0).sum()),
                "starts_positive": int((d.mean(0) > 0).sum())}
        for label, unit in [("start", np.arange(12)), ("episode", CLUSTER)]:
            m, lo, hi, p = boot(d, unit)
            line[label] = {"mean": m, "ci95": [lo, hi], "p": p}
        out["pairwise"][f"{a}_minus_{b}"] = line
        print(f"{a} - {b}: start {line['start']['mean']:+.3e} "
              f"[{line['start']['ci95'][0]:+.3e},{line['start']['ci95'][1]:+.3e}] "
              f"p={line['start']['p']:.3f} | episode "
              f"[{line['episode']['ci95'][0]:+.3e},{line['episode']['ci95'][1]:+.3e}] "
              f"p={line['episode']['p']:.3f} | seeds+ {line['seeds_positive']}/5")
    (RUN / "analysis.json").write_text(json.dumps(out, indent=2))
    print("wrote", RUN / "analysis.json")


if __name__ == "__main__":
    main()
