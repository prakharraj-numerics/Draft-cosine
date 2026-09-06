# Apple COS53 dispatch piece sweep, 40K–100K

Baseline: `baseline/apple-cos53-current-vvcos-20260906` at exact SHA `70eea3cc4395d2643e9e4fea3824fee18d755c4f`.

Experiment branch: `exp/apple-cos53-p-sweep-40-100k-20260906`.

Initial sweep run: `34018106421`, 4 M1 slots x 3 rounds = 12 samples per stack/size.
Focused confirmation run: `34018280521`, 4 M1 slots x 5 rounds = 20 samples per stack/size.

The challenger uses the current hot kernel with Apple `dispatch_apply_f(..., DISPATCH_APPLY_AUTO, ...)` and a tunable number of contiguous pieces `p`.

Baseline routes used for comparison:
- 40K: current robust `wg3_user`
- 40,001: canonical `adapt2`
- 50K: current robust `wg2_ui`
- 60K, 80K, 100K: canonical `auto` continuation for the 50K–100K zone

## Initial broad sweep

Piece counts: p8, p12, p16, p20, p24, p28, p32, p40, p48, p64.

| n | baseline wall ns/el | best p | best p wall ns/el | speed change vs baseline | best p CPU ns/el | baseline CPU ns/el | CPU change |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 40,000 | 0.7758 | p40 | 1.2368 | 37.3% slower | 2.7918 | 2.1850 | +27.8% |
| 40,001 | 1.0913 | p48 | 1.2705 | 14.1% slower | 2.8870 | 2.1680 | +33.2% |
| 50,000 | 1.2561 | p32 | 1.1562 | 8.6% faster | 2.6399 | 2.1351 | +23.6% |
| 60,000 | 1.1100 | p64 | 1.1999 | 7.5% slower | 2.8229 | 2.0271 | +39.3% |
| 80,000 | 1.3028 | p24 | 0.9380 | 38.9% faster | 2.1781 | 2.2712 | -4.1% |
| 100,000 | 1.2191 | p64 | 0.9430 | 29.3% faster | 2.2845 | 3.9117 | -41.6% |

Because scheduler timing was visibly noisy, the apparent 50K/80K/100K wins were not promoted from this sweep alone.

## Focused confirmation

### 50K

| stack | wall ns/el | CPU ns/el |
|---|---:|---:|
| baseline | 0.8627 | 1.7128 |
| p24 | 0.8839 | 2.1295 |
| p32 | 0.8963 | 2.1361 |
| p12 | 1.0303 | 2.3086 |
| Apple vvcos | 1.2919 | 1.2923 |

Conclusion: reject dispatch-p replacement at 50K. Best confirmed p24 is 2.4% slower and 24.3% more CPU than baseline.

### 80K

| stack | wall ns/el | CPU ns/el |
|---|---:|---:|
| p64 | 0.8047 | 1.9348 |
| p40 | 0.8307 | 1.9460 |
| p24 | 0.8422 | 1.9858 |
| p28 | 0.8648 | 1.9541 |
| baseline | 0.9122 | 1.9076 |
| Apple vvcos | 1.2963 | 1.2966 |

Conclusion: 80K is a real dispatch-piece opportunity. p64 is 13.36% faster than baseline, while process CPU rises only 1.43%. Apple/ours wall speedup improves from about 1.421x for baseline to 1.611x for p64. CPU/Apple rises slightly from about 1.471x to 1.492x.

### 100K

| stack | wall ns/el | CPU ns/el |
|---|---:|---:|
| baseline | 0.6086 | 1.7641 |
| p48 | 0.7683 | 1.8978 |
| p64 | 0.7729 | 1.9022 |
| p40 | 0.8251 | 1.9103 |
| p16 | 0.8405 | 1.9385 |
| p32 | 0.8587 | 1.9013 |
| Apple vvcos | 1.2912 | 1.2916 |

Conclusion: reject dispatch-p replacement at 100K. The best confirmed candidate p48 is 20.8% slower than baseline and uses 7.6% more CPU.

## Current decision

- 40K: keep baseline.
- 40,001: keep baseline.
- 50K: keep baseline.
- 60K: keep baseline.
- 80K: p64 is a confirmed speed candidate; do not change baseline unless explicitly promoted/frozen.
- 100K: keep baseline.

The broad sweep's apparent 50K and 100K wins did not reproduce. The 80K improvement did reproduce across the larger confirmation run and is supported by several nearby p values, not just a single lucky p.
