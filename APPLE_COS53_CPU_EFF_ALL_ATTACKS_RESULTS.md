# Apple COS53 CPU-efficiency all-attacks benchmark — 2026-09-06

Branch: `exp/apple-cos53-k1280-cpu-eff-all-attacks-20260906`
Candidate commit benchmarked: `a4d3328ddff2afa1525481bb8143d7c756aecebe`
Frozen comparison commit: `ddbeb33c40d9f4913ebce3537d501c1396d6ef29`
Workflow run: `33995471662`
Hardware: GitHub macOS-15 arm64 Apple M1 runners
Sampling: 4 independent slots × 3 rounds = 12 samples per mode/batch; values below are medians.

## Accuracy

- 9,600-case MPFR256 validator: exact 4,046; <=1 ULP 9,241; <=2 ULP 9,600; >2 ULP 0; max 2 ULP.
- 1,000,000-case stressed MPFR256 validator: exact 441,685; <=1 ULP 962,384; <=2 ULP 1,000,000; >2 ULP 0; max 2 ULP.

The first split-pi pass showed root-sensitive ULP failures. The final candidate retains the no-qpi-LUT common path and adds a rare root-only scalar repair for `j >= 2009`; the final validation above is the repaired candidate.

## Implemented attacks

1. Native AArch64/NEON split-pi range reduction; q*pi lookup table removed from the common path.
2. AoS coefficient layout: two 128-bit coefficient loads plus NEON zip/deinterleave.
3. Mathematically redundant j clamps removed (`0 <= j <= 2011` for K=1280 supported range).
4. Two independent 128-bit vectors per loop iteration (2× unroll).
5. Adaptive two-core helper: short spin then C++20 atomic wait/notify.
6. Persistent pthreadpool 2-thread and 3-thread modes, benchmarked alongside single, adaptive2, frozen, and Accelerate `vvcos`.

## Best median CPU mode subject to no median wall-time regression vs frozen

Negative wall change means faster. CPU reduction is lower process CPU ns/el relative to frozen. If no tested mode improved both wall and CPU medians, frozen is retained.

| n | selected | frozen wall ns/el | selected wall ns/el | wall change | frozen CPU ns/el | selected CPU ns/el | CPU reduction | vvcos wall ns/el |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| 100 | single | 2.0229 | 1.7926 | -11.4% | 4.0280 | 1.7927 | 55.5% | 1.3815 |
| 400 | frozen | 1.1982 | 1.1982 | 0.0% | 2.4319 | 2.4319 | 0.0% | 1.3113 |
| 700 | frozen | 1.1451 | 1.1451 | 0.0% | 2.1941 | 2.1941 | 0.0% | 1.3062 |
| 1,200 | frozen | 1.0599 | 1.0599 | 0.0% | 2.0780 | 2.0780 | 0.0% | 1.3073 |
| 3,000 | auto | 1.0399 | 1.0180 | -2.1% | 2.0423 | 1.9295 | 5.5% | 1.2947 |
| 5,000 | adapt2 | 1.0555 | 0.9982 | -5.4% | 2.0585 | 1.9268 | 6.4% | 1.2589 |
| 7,500 | auto | 0.9727 | 0.9370 | -3.7% | 1.9684 | 1.6581 | 15.8% | 1.3013 |
| 15,000 | auto | 0.9463 | 0.9175 | -3.0% | 1.9504 | 1.6497 | 15.4% | 1.2353 |
| 29,999 | adapt2 | 0.9527 | 0.9055 | -5.0% | 1.9461 | 1.7040 | 12.4% | 1.2664 |
| 30,000 | pool3 | 0.7368 | 0.6339 | -14.0% | 2.1210 | 1.9987 | 5.8% | 1.2956 |
| 40,000 | pool3 | 0.6955 | 0.6124 | -11.9% | 2.0440 | 1.7379 | 15.0% | 1.2836 |
| 40,001 | adapt2 | 1.0584 | 0.9213 | -13.0% | 2.4915 | 1.7957 | 27.9% | 1.3002 |
| 50,000 | auto | 1.1394 | 0.8950 | -21.4% | 2.4846 | 1.7780 | 28.4% | 1.2686 |
| 100,000 | auto | 0.8485 | 0.6201 | -26.9% | 2.1526 | 1.7380 | 19.3% | 1.2819 |
| 200,000 | pool3 | 0.7792 | 0.6184 | -20.6% | 2.0435 | 1.9429 | 4.9% | 1.2923 |
| 500,000 | frozen | 0.7116 | 0.7116 | 0.0% | 1.9597 | 1.9597 | 0.0% | 1.3273 |
| 1,000,000 | frozen | 0.6745 | 0.6745 | 0.0% | 1.9064 | 1.9064 | 0.0% | 1.3252 |

## Interpretation

The all-attacks candidate produced clear dual wins at multiple batch sizes: faster wall time and lower process CPU simultaneously. The strongest median examples are 50K (21.4% faster, 28.4% less CPU), 100K (26.9% faster, 19.3% less CPU), 40,001 (13.0% faster, 27.9% less CPU), and 7.5K/15K (~3–4% faster with ~15–16% less CPU).

The tested candidate does not dominate the frozen path everywhere. At 400, 700, 1,200, 500K and 1M, no tested candidate mode improved both median wall time and CPU, so those ranges should retain the frozen route for now.

`vvcos` remains faster than the selected candidate at n=100, but the custom kernel is faster than `vvcos` at every tested n from 400 upward in the median table.

No frozen branch was modified; all work remains on the experiment branch.
