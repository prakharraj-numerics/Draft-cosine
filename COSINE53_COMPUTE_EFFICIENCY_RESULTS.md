# COS53 compute-efficiency benchmark vs Intel Xeon

Saved benchmark state from corrected run **33684682062** at commit **9c041f39c434cf70026e8da178ccf9ffca0e0973**.

Hardware / comparator:
- Intel Xeon 6973P-C, AVX-512
- Intel oneMKL `vmdCos(..., VML_HA)`, sequential
- COS53 frozen production dispatcher: current evaluator for `n < 1600`, permanent custom2 scheduler for `n >= 1600`
- Six deterministic cases per batch: ±unit, ±mid, ±far
- Batch sizes: 100, 700, 3.5K, 15K, 50K, 1M, 2M
- Accuracy gate: observed max difference <= 2 ULP

Corrected benchmark isolation: Intel and no-op measurements do **not** construct the COS53 permanent helper scheduler.

Exact-Xeon artifacts used for the central result: shards **12, 13, 37, 59, 64, 84**. Values below are medians across those six shards.

| n | COS53 speedup vs Intel | CPU-efficiency ratio (Intel/COS53) | COS53 instr/el | Intel instr/el | Instruction-efficiency ratio (Intel/COS53) | COS53 logical B/el | Intel logical B/el |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 1.294x | 0.633x | 1342.42 | 12.43 | 0.0093x | 2697.9 | 130.6 |
| 700 | 1.293x | 0.667x | 220.81 | 8.74 | 0.0396x | 453.65 | 108.35 |
| 3.5K | 1.930x | 1.035x | 48.22 | 8.25 | 0.171x | 108.49 | 108.68 |
| 15K | 2.421x | 1.271x | 14.55 | 8.15 | 0.560x | 41.05 | 108.27 |
| 50K | 2.668x | 1.382x | 7.63 | 8.13 | 1.065x | 27.19 | 108.27 |
| 1M | 2.503x | 1.384x | 4.36 | 8.13 | 1.863x | 20.64 | 108.25 |
| 2M | 2.429x | 1.259x | 4.27 | 8.13 | 1.902x | 20.46 | 108.25 |

Interpretation:
- Ratios above 1 mean COS53 is more efficient than Intel for that metric.
- COS53 is faster at every tested batch size.
- CPU-time efficiency reaches approximate parity by 3.5K and is better than Intel from 15K onward.
- Dynamic-instruction efficiency crosses Intel around 50K; by 1M-2M COS53 uses about 1.86x-1.90x fewer dynamic instructions per element.
- Logical memory traffic is approximately 5.24x lower at 1M and 5.29x lower at 2M.
- Small batches are resource-expensive because constructing the frozen production dispatcher also creates its permanent helper thread even when `n < 1600` routes the actual evaluation through the current single-core evaluator.

Approximate peak RSS medians from the same corrected benchmark:

| n | COS53 RSS | Intel RSS |
|---:|---:|---:|
| 50K | ~21.7 MiB | ~20.1 MiB |
| 1M | ~108.2 MiB | ~77.6 MiB |
| 2M | ~199.7 MiB | ~138.7 MiB |

Accuracy observed in this benchmark:
- <=1 ULP at 100, 700, 3.5K, 15K, and 1M
- <=2 ULP at 50K and 2M

This file is evidence only. It does not change production routing or implementation.