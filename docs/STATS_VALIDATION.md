# Validating the statistics against established libraries

## Claim

The statistics in `compare_arms.py` are written with the Python standard library only (`math`,
`random`, `statistics`). They return the same results as scipy and statsmodels: to floating-point
precision where the statistic has a closed form, and within Monte Carlo error where it is
estimated by resampling.

## Implementation

`check_stats_against_libraries.py` imports the repo's own functions and runs them beside the
library equivalents. It is read-only and writes nothing.

```
uv run --with statsmodels python check_stats_against_libraries.py
```

Two kinds of check:

- **Closed-form statistics** (Wilson, McNemar exact, Holm, across-seed sd) on synthetic inputs
  covering their range. Agreement should be exact up to rounding.
- **Monte Carlo statistics** (sign-flip test, cluster bootstrap) on the real pooled data of the
  two finished experiments, the same inputs the results files report on:
  - `p repo` is the sign-flip p-value `compare_arms.py` quotes and passes to Holm (50,000 draws);
    `CI repo` is the cluster bootstrap interval it prints (10,000 draws).
  - `p exact` is the sign-flip p-value with no simulation at all. Every cluster difference is a
    multiple of 1/3 (Formula) or 1/12 (FiNER), so the full null distribution over all 2^n sign
    patterns can be counted exactly.
  - `p scipy` and `CI scipy` are `scipy.stats.permutation_test` (paired) and
    `scipy.stats.bootstrap` (percentile) on the same clusters, 200,000 resamples each.
  - `SE bootstrap` is the sd of the repo's cluster resampling; `SE statsmodels` is the
    cluster-robust standard error of a regression of the row differences on a constant. The two
    are different estimators of the same quantity.

## Results

### Closed-form statistics

| repo function | library | inputs | largest difference |
|---|---|---|---|
| Wilson interval | statsmodels `proportion_confint(method="wilson")` | every count at n = 10, 50, 200, 441 | 9e-11 |
| McNemar exact | statsmodels `mcnemar(exact=True)` | all b, c from 0 to 40 | 6e-16 |
| McNemar exact | scipy `binomtest` | same | 6e-16 |
| Holm | statsmodels `multipletests(method="holm")` | 2,000 random families of 1-8 p-values | 0 |
| across-seed sd | numpy `std(ddof=1)` | 2,000 random samples | 1e-16 |

### Monte Carlo statistics on the real pooled data

**Formula** (50 clusters, 200 rows)

| comparison | p repo | p exact | p scipy | CI repo | CI scipy | SE bootstrap | SE statsmodels |
|---|---|---|---|---|---|---|---|
| none -> text | 0.0140 | 0.0147 | 0.0149 | [+0.022, +0.155] | [+0.023, +0.155] | 0.0336 | 0.0340 |
| none -> cocite | 0.0005 | 0.0005 | 0.0006 | [+0.052, +0.170] | [+0.052, +0.170] | 0.0305 | 0.0307 |
| text -> cocite | 0.3344 | 0.3358 | 0.3337 | [-0.020, +0.060] | [-0.018, +0.062] | 0.0202 | 0.0203 |

Holm on this family: repo 0.0280, 0.0016, 0.3344; statsmodels 0.0280, 0.0016, 0.3344.

**FiNER, per-tag** (441 rows, each its own cluster)

| comparison | p repo | p exact | p scipy | CI repo | CI scipy | SE bootstrap | SE statsmodels |
|---|---|---|---|---|---|---|---|
| none -> text | <0.0001 | <0.0001 | <0.0001 | [+0.055, +0.121] | [+0.055, +0.121] | 0.0168 | 0.0169 |
| none -> cocite | 0.0001 | <0.0001 | <0.0001 | [+0.036, +0.100] | [+0.036, +0.100] | 0.0163 | 0.0164 |
| text -> cocite | 0.0601 | 0.0602 | 0.0610 | [-0.041, +0.001] | [-0.041, +0.001] | 0.0106 | 0.0106 |

Holm on this family: repo 0.0001, 0.0001, 0.0601; statsmodels 0.0001, 0.0001, 0.0601.

Every point estimate also matches statsmodels exactly (asserted by the script). The p-values sit
within simulation error of the exact values, the intervals agree to within 0.002 at each end,
and the two standard errors agree within about 1%.

## Not covered

- **Westfall-Young** has no Python reference implementation (the standard one is R's `multtest`,
  `mt.maxT`), so it is checked only by `compare_arms.py --selftest`, against its known limiting
  case.
- These checks show the code computes what it claims. They do not measure how often the
  bootstrap interval covers the true effect at intermediate within-cluster correlation with 50
  clusters; that would need a simulation study. The sign-flip test needs no such study, since its
  error rate holds by construction under the null.
