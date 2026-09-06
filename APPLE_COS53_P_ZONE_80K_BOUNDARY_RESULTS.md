# Apple COS53 p64 production-zone boundary around 80K

Current baseline: `baseline/apple-cos53-current-vvcos-20260906` at exact tested SHA `70eea3cc4395d2643e9e4fea3824fee18d755c4f`.

Experiment branch: `exp/apple-cos53-p-sweep-40-100k-20260906`.

Relevant runs:
- focused 50K/80K/100K confirmation: `34018280521`
- coarse 60K-100K boundary: `34018578000`
- edge refinement: `34018726770`
- central 72.5K-87.5K: `34018894312`
- micro 78K-82K boundary: `34019110971`

All of the boundary/central/micro runs completed successfully on four GitHub-hosted Apple M1 runners.

## Why paired ratios are used for the final boundary

Absolute wall-time levels vary substantially between hosted M1 runner instances. For the micro-boundary decision, the robust comparison is the ratio of challenger to baseline measured within the same runner slot and same benchmark round. Baseline ordering was alternated before/after the p candidates across slot/round parity.

Each micro point has 20 paired observations (4 M1 slots x 5 rounds).

## Micro-boundary result

`p64` is the preferred route in the central zone. Ratios below are medians of same-slot/same-round paired ratios. A wall ratio below 1 means p64 is faster; CPU ratio below 1 means p64 consumes less process CPU.

| n | p64/baseline wall ratio | speed gain | p64/baseline CPU ratio | CPU change | p64 wall wins / 20 |
|---:|---:|---:|---:|---:|---:|
| 78,000 | 0.7386 | 35.4% | 0.9457 | -5.4% | 19 |
| 78,500 | 0.7806 | 28.1% | 1.0370 | +3.7% | 17 |
| 79,000 | 0.7942 | 25.9% | 0.9787 | -2.1% | 18 |
| 79,500 | 0.7902 | 26.6% | 0.9773 | -2.3% | 17 |
| 80,000 | 0.8330 | 20.0% | 0.9946 | -0.5% | 17 |
| 80,500 | 0.8547 | 17.0% | 1.0486 | +4.9% | 13 |
| 81,000 | 0.8102 | 23.4% | 1.0499 | +5.0% | 13 |
| 81,500 | 0.9144 | 9.4% | 1.0464 | +4.6% | 12 |
| 82,000 | 0.8741 | 14.4% | 1.0258 | +2.6% | 14 |

## Outside controls

The larger 72.5K-87.5K run provides the immediate outside controls:

- 77,500 p64/baseline paired median wall ratio = **1.0424**; only **7/20** wall wins: baseline preferred.
- 82,500 p64/baseline paired median wall ratio = **1.0379**; only **9/20** wall wins: baseline preferred.

Thus the observed transition is sharp at the tested 500-element resolution:

- `n <= 77,500`: keep baseline routing.
- `78,000 <= n <= 82,000`: **p64 dispatch candidate**.
- `n >= 82,500`: return to baseline routing.

A production rule can conservatively use `78000 <= n && n <= 82000` (or half-open `78000 <= n && n < 82500`) until finer-than-500 boundary testing is justified.

## Interpretation

This is not an isolated 80K point. The p64 route wins on wall time at every 500-element micro point from 78K through 82K under paired comparison, with median process-CPU change ranging from about -5.4% to +5.0%. The adjacent tested controls at 77.5K and 82.5K do not win, so extending p64 beyond this narrow interval is not supported by the current data.

The current baseline branch remains unchanged; this report records the candidate interval only.
