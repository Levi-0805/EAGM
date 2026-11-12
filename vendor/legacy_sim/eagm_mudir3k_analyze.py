"""Analyze the 3,000-update Mu / Mu+Dir / Mc comparison.

Resampling unit is the test start, not the seed: the twelve test starts are
shared across seeds, so a seed-level t-test overstates precision.
"""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs" / "eagm_mudir3k_20261112"
rng = np.random.default_rng(0)
# Twelve test starts are two frames of each of six trajectories.
EPISODE = np.repeat(np.arange(6), 2)


def cluster_bootstrap(diff, unit, n=20000):
    """diff: (seeds, starts). Resample the given clusters with replacement."""
    units = np.unique(unit)
    means = np.empty(n)
    for i in range(n):
        pick = rng.choice(units, size=len(units), replace=True)
        means[i] = np.mean([diff[:, unit == c].mean() for c in pick])
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return float(diff.mean()), float(lo), float(hi), float(p)


def main():
    data = json.loads((RUN / "results.json").read_text())
    rows = data["rows"]
    seeds = sorted({r["seed"] for r in rows})
    names = ["mc3k", "mu3k", "mudir3k"]
    returns = {n: np.array([[r["returns"] for r in rows
                             if r["seed"] == s and r["model"] == n][0]
                            for s in seeds]) for n in names}
    cosines = {n: np.array([np.mean([r["cosine_median"] for r in rows
                                     if r["seed"] == s and r["model"] == n])
                            for s in seeds]) for n in names}

    out = {"seeds": seeds, "n_starts": int(returns[names[0]].shape[1]),
           "per_model": {}, "pairwise": {}}
    for n in names:
        per_seed = returns[n].mean(1)
        out["per_model"][n] = {
            "return_mean": float(per_seed.mean()),
            "return_per_seed": per_seed.tolist(),
            "return_seed_sd": float(per_seed.std(ddof=1)),
            "cosine_mean": float(cosines[n].mean()),
            "cosine_per_seed": cosines[n].tolist(),
        }
    for a, b in [("mudir3k", "mu3k"), ("mudir3k", "mc3k"), ("mu3k", "mc3k")]:
        d = returns[a] - returns[b]
        per_seed = d.mean(1)
        t = float(per_seed.mean() / (per_seed.std(ddof=1) / np.sqrt(len(per_seed))))
        entry = {"per_seed": per_seed.tolist(),
                 "seeds_positive": int((per_seed > 0).sum()),
                 "seed_t": t,
                 "starts_positive": int((d.mean(0) > 0).sum())}
        for label, unit in [("start", np.arange(d.shape[1])), ("episode", EPISODE)]:
            mean, lo, hi, p = cluster_bootstrap(d, unit)
            entry[label] = {"mean": mean, "ci95": [lo, hi], "p": p}
        out["pairwise"][f"{a}_minus_{b}"] = entry
        print(f"{a} - {b}: start {entry['start']['mean']:+.3e} "
              f"[{entry['start']['ci95'][0]:+.3e},{entry['start']['ci95'][1]:+.3e}] "
              f"p={entry['start']['p']:.3f} | episode "
              f"[{entry['episode']['ci95'][0]:+.3e},{entry['episode']['ci95'][1]:+.3e}] "
              f"p={entry['episode']['p']:.3f} | seeds+ "
              f"{entry['seeds_positive']}/{len(seeds)} starts+ "
              f"{entry['starts_positive']}/{d.shape[1]} seed_t={t:.2f}")
    (RUN / "analysis.json").write_text(json.dumps(out, indent=2))
    print("wrote", RUN / "analysis.json")


if __name__ == "__main__":
    main()
