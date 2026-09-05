# Apple COS53 hybrid speed+CPU vs Accelerate vvcos — all batches

Date: 2026-09-06
Workflow run: `33996024995`
Branch: `exp/apple-cos53-k1280-cpu-eff-all-attacks-20260906`
Workflow commit: `14c74c60c3d2507033c49fff10382ccc11c308e1`
Hardware: GitHub macOS-15 arm64 Apple M1 runners
Sampling: 4 independent slots × 3 rounds = 12 paired samples per batch/stack; table uses medians.

Hybrid route uses the winner map from the previous all-attacks run. `frozen` means the frozen K1280 fast-reduction implementation; other modes use the optimized split-pi/AoS/unrolled kernel.

Accuracy revalidated in this run:
- 9,600 MPFR256 cases: max 2 ULP, 0 cases >2 ULP.
- 1,000,000 stressed MPFR256 cases: max 2 ULP, 0 cases >2 ULP.

Positive speed advantage means hybrid is faster than `vvcos`. CPU ratio is hybrid process CPU ns/el divided by `vvcos` process CPU ns/el; values above 1 mean hybrid consumes more process CPU.

| n | selected | hybrid wall ns/el | vvcos wall ns/el | wall speedup | speed advantage | hybrid CPU ns/el | vvcos CPU ns/el | CPU ratio H/A | CPU advantage |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | single | 1.8441 | 1.4078 | 0.763× | -31.0% | 1.8300 | 1.4038 | 1.304× | -30.4% |
| 400 | frozen | 1.2503 | 1.3542 | 1.083× | +7.7% | 2.5271 | 1.3298 | 1.900× | -90.0% |
| 700 | frozen | 1.1765 | 1.3493 | 1.147× | +12.8% | 2.2941 | 1.3486 | 1.701× | -70.1% |
| 1,200 | frozen | 1.1005 | 1.3057 | 1.187× | +15.7% | 2.1290 | 1.3057 | 1.631× | -63.1% |
| 3,000 | auto | 1.0741 | 1.2950 | 1.206× | +17.1% | 2.0477 | 1.2951 | 1.581× | -58.1% |
| 5,000 | adapt2 | 1.1005 | 1.3061 | 1.187× | +15.7% | 2.0024 | 1.3053 | 1.534× | -53.4% |
| 7,500 | auto | 0.9569 | 1.3043 | 1.363× | +26.6% | 1.8537 | 1.3036 | 1.422× | -42.2% |
| 15,000 | auto | 0.9379 | 1.3146 | 1.402× | +28.7% | 1.8494 | 1.3138 | 1.408× | -40.8% |
| 29,999 | adapt2 | 1.0612 | 1.3175 | 1.242× | +19.5% | 1.9278 | 1.3159 | 1.465× | -46.5% |
| 30,000 | pool3 | 1.5078 | 1.3455 | 0.892× | -12.1% | 4.2285 | 1.3451 | 3.144× | -214.4% |
| 40,000 | pool3 | 0.8857 | 1.4026 | 1.584× | +36.9% | 2.6286 | 1.3962 | 1.883× | -88.3% |
| 40,001 | adapt2 | 1.1671 | 1.3930 | 1.194× | +16.2% | 1.9880 | 1.3915 | 1.429× | -42.9% |
| 50,000 | auto | 0.9766 | 1.3417 | 1.374× | +27.2% | 1.9035 | 1.3413 | 1.419× | -41.9% |
| 100,000 | auto | 0.9060 | 1.3595 | 1.500× | +33.4% | 2.1697 | 1.3525 | 1.604× | -60.4% |
| 200,000 | pool3 | 0.8765 | 1.3192 | 1.505× | +33.6% | 2.3603 | 1.3179 | 1.791× | -79.1% |
| 500,000 | frozen | 0.8204 | 1.3402 | 1.634× | +38.8% | 1.9607 | 1.3374 | 1.466× | -46.6% |
| 1,000,000 | frozen | 0.7614 | 1.3450 | 1.767× | +43.4% | 1.9368 | 1.3419 | 1.443× | -44.3% |

## Summary

- Hybrid is faster than `vvcos` at 15 of 17 tested batch sizes.
- Exceptions: n=100 and n=30,000.
- Largest median wall wins: 1M (+43.4%), 500K (+38.8%), 40K (+36.9%), 200K (+33.6%), 100K (+33.4%).
- However, hybrid still consumes more total process CPU than `vvcos` at every tested batch size. The CPU gap is typically ~1.4×–1.9×, with a severe pool3 instability at n=30K in this run.
- Therefore the previous CPU-efficiency attack improved CPU use relative to our own frozen implementation, but has not yet reached Apple `vvcos` CPU efficiency.
