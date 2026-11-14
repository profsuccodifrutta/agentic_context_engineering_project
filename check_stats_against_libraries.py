"""Cross-check compare_arms.py's hand-written statistics against scipy and statsmodels.

Read-only: reads the finished runs under results/

    uv run --with statsmodels python check_stats_against_libraries.py

Prints the tables in docs/STATS_VALIDATION.md.
"""

import glob
import itertools
import random
import statistics
from collections import defaultdict
from fractions import Fraction

import numpy as np
import statsmodels.api as sm
from scipy import stats as ss
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

import compare_arms as ca


def closed_form():
    """Formula-based pieces: must agree to floating-point precision."""
    rng = random.Random(1)
    rows = []

    pairs = []
    for total in (10, 50, 200, 441):
        for correct in range(total + 1):
            mine = ca.wilson_interval(correct, total)
            ref = proportion_confint(correct, total, alpha=0.05, method="wilson")
            pairs += [(mine[0], ref[0]), (mine[1], ref[1])]
    rows.append(("Wilson interval", "statsmodels `proportion_confint(method=\"wilson\")`",
                 "every count at n = 10, 50, 200, 441", pairs))

    pairs = [(ca.mcnemar_exact(b, c), sm_mcnemar([[0, b], [c, 0]], exact=True).pvalue)
             for b in range(41) for c in range(41)]
    rows.append(("McNemar exact", "statsmodels `mcnemar(exact=True)`", "all b, c from 0 to 40", pairs))

    pairs = [(ca.mcnemar_exact(b, c), ss.binomtest(min(b, c), b + c, 0.5).pvalue)
             for b in range(41) for c in range(41) if b + c]
    rows.append(("McNemar exact", "scipy `binomtest`", "same", pairs))

    pairs = []
    for _ in range(2000):
        p = [rng.random() ** 3 for _ in range(rng.randint(1, 8))]
        pairs += list(zip(ca.holm(p), multipletests(p, method="holm")[1]))
    rows.append(("Holm", "statsmodels `multipletests(method=\"holm\")`",
                 "2,000 random families of 1-8 p-values", pairs))

    pairs = []
    for _ in range(2000):
        x = [rng.random() for _ in range(rng.randint(2, 6))]
        pairs.append((statistics.stdev(x), float(np.std(x, ddof=1))))
    rows.append(("across-seed sd", "numpy `std(ddof=1)`", "2,000 random samples", pairs))

    print("| repo function | library | inputs | largest difference |")
    print("|---|---|---|---|")
    for name, lib, inputs, pairs in rows:
        print(f"| {name} | {lib} | {inputs} | {max(abs(a - b) for a, b in pairs):.0e} |")


def exact_signflip_p(diffs, unit):
    """Exact two-sided sign-flip p-value: the whole null distribution, by dynamic programming.

    Possible because every cluster difference is a multiple of `unit`, so the 2^n sign patterns
    collapse onto a small set of integer sums.
    """
    ints = [round(Fraction(d).limit_denominator(1000) / unit) for d in diffs]
    dist = {0: 1}
    for m in ints:
        step = defaultdict(int)
        for s, w in dist.items():
            step[s + m] += w
            step[s - m] += w
        dist = step
    observed = abs(sum(ints))
    return float(Fraction(sum(w for s, w in dist.items() if abs(s) >= observed), 2 ** len(ints)))


def fmt_p(p):
    return "<0.0001" if p < 0.0001 else f"{p:.4f}"


def run_dir(path):
    return glob.glob(path + "/ace_run_*")[0]


def real_data(title, arm_dirs, scorer, clusters, unit):
    """Monte Carlo pieces on the real pooled data: must agree within simulation error."""
    by_arm = {arm: [{"preds": ca.load_predictions(run_dir(d), scorer)[0]} for d in dirs]
              for arm, dirs in arm_dirs.items()}
    scores, _, shared, refusal = ca.pooled_arm_scores(by_arm)
    assert refusal is None, refusal
    labels = clusters or {i: i for i in shared}  # no block structure: every row is its own cluster

    print(f"\n**{title}** ({len(set(labels[i] for i in shared))} clusters, {len(shared)} rows)\n")
    print("| comparison | p repo | p exact | p scipy | CI repo | CI scipy | SE bootstrap | SE statsmodels |")
    print("|---|---|---|---|---|---|---|---|")
    raw = []
    for left, right in itertools.combinations(arm_dirs, 2):
        # The p-value compare_arms quotes and feeds to Holm (report_multiplicity, 50,000 draws,
        # seed 0 -- the same random sequence as this call) and the CI it prints (10,000 draws).
        p_repo = ca.cluster_paired_stats(shared, scores[left], scores[right], labels, draws=50000)["p"]
        repo = ca.cluster_paired_stats(shared, scores[left], scores[right], labels)
        raw.append(p_repo)

        blocks = defaultdict(lambda: [0, 0.0, 0.0])
        for i in shared:
            b = blocks[labels[i]]
            b[0] += 1
            b[1] += scores[left][i]
            b[2] += scores[right][i]
        ordered = [blocks[k] for k in sorted(blocks)]
        rows = np.array([b[0] for b in ordered], float)
        x = np.array([b[1] for b in ordered])
        y = np.array([b[2] for b in ordered])
        total = rows.sum()

        p_exact = exact_signflip_p(list(y - x), unit)
        perm = ss.permutation_test((x, y), lambda a, b, axis: (b.sum(axis) - a.sum(axis)) / total,
                                   permutation_type="samples", vectorized=True, n_resamples=200000,
                                   alternative="two-sided", random_state=0)
        boot = ss.bootstrap((y - x, rows), lambda d, r, axis: d.sum(axis) / r.sum(axis), paired=True,
                            vectorized=True, method="percentile", n_resamples=200000, random_state=0)
        ci = boot.confidence_interval

        # Spread of compare_arms' own cluster resampling, against a regression-based estimator.
        rng = random.Random(0)
        spread = statistics.stdev(
            sum(b[2] - b[1] for b in pick) / sum(b[0] for b in pick)
            for pick in ([rng.choice(ordered) for _ in ordered] for _ in range(20000)))
        diff = np.array([scores[right][i] - scores[left][i] for i in shared])
        groups = np.array([sorted(blocks).index(labels[i]) for i in shared])
        ols = sm.OLS(diff, np.ones_like(diff)).fit(cov_type="cluster", cov_kwds={"groups": groups})
        assert abs(ols.params[0] - repo["diff"]) < 1e-12, "point estimates disagree"

        print(f"| {left} -> {right} | {fmt_p(p_repo)} | {fmt_p(p_exact)} | {fmt_p(perm.pvalue)} "
              f"| [{repo['ci'][0]:+.3f}, {repo['ci'][1]:+.3f}] | [{ci.low:+.3f}, {ci.high:+.3f}] "
              f"| {spread:.4f} | {ols.bse[0]:.4f} |")

    mine, ref = ca.holm(raw), multipletests(raw, method="holm")[1]
    print(f"\nHolm on this family: repo {', '.join(f'{v:.4f}' for v in mine)}; "
          f"statsmodels {', '.join(f'{v:.4f}' for v in ref)}")


if __name__ == "__main__":
    print("### Closed-form statistics\n")
    closed_form()
    print("\n### Monte Carlo statistics on the real pooled data")
    F = "results/formula"
    real_data("Formula",
              {"none": [f"{F}/preregistered/seed{s}/none" for s in range(3)],
               "text": [f"{F}/cocite_experiment/seed{s}/text" for s in range(3)],
               "cocite": [f"{F}/cocite_experiment/seed{s}/cocite" for s in range(3)]},
              ca._score_exact, ca.load_clusters("formula"), Fraction(1, 3))
    # The finer control is deterministic and entered three times, as in docs/FINER_RESULTS.md.
    real_data("FiNER, per-tag",
              {"none": ["results/finer/repaired_v2/seed0/none"] * 3,
               "text": [f"results/finer/final/seed{s}/text" for s in range(3)],
               "cocite": [f"results/finer/final/seed{s}/cocite" for s in range(3)]},
              ca._score_finer_per_tag, None, Fraction(1, 12))
