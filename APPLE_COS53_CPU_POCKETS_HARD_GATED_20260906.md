# Apple COS53 CPU pockets — hard-gated configuration attack

Frozen incumbent baseline: `b47b2fff19ad2c323de8ded704cb6ddc6573541c` (`baseline/apple-cos53-current-contract-fast-p64-20260906`).

Experiment branch: `exp/apple-cos53-cpu-pockets-v1-20260906`.

Hard acceptance constraint:

- candidate wall time must not exceed the frozen baseline wall time;
- candidate process CPU time must be lower than frozen baseline;
- no math/kernel approximation change; configuration/scheduler/partition changes only.

Focused confirmation run: `34021918620`, 4 Apple M1 jobs x 5 rounds = 20 paired samples per candidate/batch. All four jobs succeeded and accuracy validation passed.

Initial integrated run: `34022156421`, 4 M1 x 3 rounds. It rejected the 82K replacement because its integrated pooled wall median was 2.69% slower despite saving CPU. 500K candidates also failed the wall constraint, so both 82K and 500K were restored to the exact frozen baseline route.

Final exact-map run: `34022399889`, exact tested commit `ff3576202519e55854bb2e118e08f0118cebd470`, 4 M1 x 3 rounds. All four jobs succeeded; route assertions succeeded; slot 1 accuracy validation succeeded.

Accuracy for both fast/off hot kernels remained identical:

- 9600 MPFR256: exact 4046, <=1 ULP 9241, <=2 ULP 9600, max 2 ULP.
- genuine 1M stress: exact 441685, <=1 ULP 962384, <=2 ULP 1000000, max 2 ULP.

## Final hard-gated map

| n | route |
|---:|:---|
| 78K | contract-fast existing Apple Workgroup 3, utility QoS (`wg3_utility`) |
| 80K | contract-fast aligned Workgroup 2, utility QoS, 8-yield spin (`wg2_utility_s8`) |
| 82K | **exact frozen baseline fallback** |
| 100K | original `ffp-contract=off` hot-loop native pool3 |
| 200K | contract-fast existing Apple Workgroup 3, utility QoS (`wg3_utility`) |
| 500K | **exact frozen baseline fallback** |
| 1M | contract-fast aligned Workgroup 3, default QoS, 8-yield spin (`wg3_default_s8`) |

## Final exact-map 12-sample medians

For 82K and 500K the selected route is literally the frozen baseline route, so they are structurally equal. Separate invocations show ordinary hosted-runner timing noise and are not treated as a code regression or improvement.

| n | frozen wall ns/el | selected wall ns/el | wall improvement | frozen CPU ns/el | selected CPU ns/el | CPU saving | Apple wall ns/el | Apple CPU ns/el | Apple/selected wall | selected/Apple CPU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 78K | 0.761441 | 0.513734 | 48.22% | 1.816880 | 1.605513 | 11.63% | 1.282381 | 1.282521 | 2.496x | 1.252x |
| 80K | 0.772917 | 0.699854 | 10.44% | 1.823208 | 1.418792 | 22.18% | 1.302137 | 1.300792 | 1.861x | 1.091x |
| 82K | exact fallback | exact fallback | 0% | exact fallback | exact fallback | 0% | 1.301588 | 1.299585 | baseline-dependent | baseline-dependent |
| 100K | 0.622184 | 0.567014 | 9.73% | 1.908958 | 1.618042 | 15.24% | 1.290542 | 1.290083 | 2.276x | 1.254x |
| 200K | 0.591444 | 0.477056 | 23.98% | 2.226667 | 1.489625 | 33.10% | 1.300458 | 1.300500 | 2.726x | 1.145x |
| 500K | exact fallback | exact fallback | 0% | exact fallback | exact fallback | 0% | 1.339682 | 1.339208 | baseline-dependent | baseline-dependent |
| 1M | 0.722457 | 0.510958 | 41.39% | 1.948708 | 1.498917 | 23.08% | 1.319061 | 1.314042 | 2.582x | 1.141x |

Paired same-slot/same-round median selected/baseline ratios at modified points:

| n | paired wall ratio | paired CPU ratio | wall wins | CPU wins | both wins |
|---:|---:|---:|---:|---:|---:|
| 78K | 0.6886 | 0.9207 | 9/12 | 8/12 | 8/12 |
| 80K | 0.9462 | 0.8090 | 9/12 | 11/12 | 9/12 |
| 100K | 0.9020 | 0.9050 | 9/12 | 10/12 | 8/12 |
| 200K | 0.8161 | 0.7674 | 8/12 | 11/12 | 8/12 |
| 1M | 0.7315 | 0.7581 | 10/12 | 12/12 | 10/12 |

Across the seven >50K checkpoints, treating 82K and 500K structural fallbacks as exactly equal to the frozen baseline, geometric means are:

- frozen/selected wall = **1.1780x** (~17.8% faster);
- selected/frozen CPU = **0.8420x** (~15.8% lower CPU);
- Apple/selected wall = **2.1887x**;
- selected/Apple CPU = **1.2407x**.

This is an experimental successor candidate only. The frozen baseline `b47b2fff...` remains unchanged and active until explicitly promoted.
