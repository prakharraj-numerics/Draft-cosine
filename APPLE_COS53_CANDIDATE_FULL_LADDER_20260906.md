# Apple COS53 candidate full ladder vs Apple M1

Candidate commit: `2daf7bd0a5a08df06efd169b1c38b55569de32f7` on `exp/apple-cos53-cpu-pockets-v1-20260906`.

Current frozen baseline ancestor: `b47b2fff19ad2c323de8ded704cb6ddc6573541c`.

Candidate routing changes vs current frozen baseline:

- 78K -> contract-fast existing Apple Workgroup 3, utility QoS (`wg3_utility`)
- 80K -> contract-fast aligned Workgroup 2, utility QoS, 8-yield spin (`wg2_utility_s8`)
- 82K -> contract-fast existing Apple Workgroup 2, default QoS (`wg2_default`)
- 100K -> original `ffp-contract=off` hot-loop native pool3
- 200K -> contract-fast existing Apple Workgroup 3, utility QoS (`wg3_utility`)
- 500K -> exact current frozen baseline fallback
- 1M -> contract-fast aligned Workgroup 3, default QoS, 8-yield spin (`wg3_default_s8`)
- all other standard benchmark points -> current frozen baseline route unchanged.

Workflow run: `34023323060`, 4 Apple M1 jobs x 3 rounds = 12 samples per stack per batch. All four jobs passed. Slot 1 accuracy validation passed for both fast/off hot kernels with unchanged results: 9600 max 2 ULP and genuine 1M stress max 2 ULP.

Ratios below use the median of paired same-slot/same-round measurements. Absolute ns/el columns are pooled medians across the 12 samples.

| n | candidate wall ns/el | Apple wall ns/el | Apple/candidate wall | candidate CPU ns/el | Apple CPU ns/el | candidate/Apple CPU |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 1.3571 | 1.3431 | **0.990x** | 1.3552 | 1.3414 | **1.010x** |
| 400 | 0.9604 | 1.2354 | **1.329x** | 1.8168 | 1.2349 | **1.433x** |
| 700 | 0.8298 | 1.2562 | **1.498x** | 1.5713 | 1.2563 | **1.239x** |
| 1,200 | 0.8166 | 1.2136 | **1.569x** | 1.5293 | 1.2122 | **1.179x** |
| 3,000 | 0.9264 | 1.2413 | **1.331x** | 1.8277 | 1.2369 | **1.432x** |
| 5,000 | 0.7704 | 1.2276 | **1.618x** | 1.4215 | 1.2269 | **1.163x** |
| 7,500 | 0.7096 | 1.2195 | **1.734x** | 1.3654 | 1.2195 | **1.122x** |
| 15,000 | 0.6924 | 1.2404 | **1.764x** | 1.3394 | 1.2397 | **1.105x** |
| 29,999 | 0.8520 | 1.2086 | **1.435x** | 1.5983 | 1.2086 | **1.313x** |
| 30,000 | 0.5986 | 1.2056 | **2.000x** | 1.8126 | 1.2055 | **1.516x** |
| 40,000 | 0.4649 | 1.2167 | **2.595x** | 1.4608 | 1.2162 | **1.148x** |
| 40,001 | 0.8658 | 1.2041 | **1.415x** | 1.7239 | 1.2033 | **1.404x** |
| 50,000 | 0.6732 | 1.2100 | **1.805x** | 1.3311 | 1.2091 | **1.103x** |
| 78,000 | 0.4694 | 1.2274 | **2.594x** | 1.5985 | 1.2273 | **1.282x** |
| 80,000 | 0.7012 | 1.2276 | **1.773x** | 1.3218 | 1.2276 | **1.064x** |
| 82,000 | 0.6864 | 1.2555 | **1.819x** | 1.3211 | 1.2557 | **1.032x** |
| 100,000 | 0.5523 | 1.2547 | **2.239x** | 1.5087 | 1.2543 | **1.130x** |
| 200,000 | 0.5954 | 1.2457 | **2.193x** | 1.7862 | 1.2457 | **1.429x** |
| 500,000 | 0.7339 | 1.2887 | **1.748x** | 1.9350 | 1.2877 | **1.559x** |
| 1,000,000 | 0.5417 | 1.2648 | **2.489x** | 1.4683 | 1.2581 | **1.175x** |

Geometric mean across all 20 standard points:

- Apple/candidate wall = **1.7458x**.
- candidate/Apple CPU = **1.2315x**.

Across the 8 points at 50K and above:

- Apple/candidate wall = **2.0586x**.
- candidate/Apple CPU = **1.2100x**.

Across the 7 points above 50K (78K through 1M):

- Apple/candidate wall = **2.0977x**.
- candidate/Apple CPU = **1.2261x**.

Observation: 200K is scheduler-sensitive. This full-ladder run measured candidate/Apple CPU at ~1.429x, materially worse than the earlier focused pocket run, despite wall remaining ~2.193x faster than Apple. 500K is intentionally unchanged from the current baseline.

This candidate is not promoted to a new baseline yet.
