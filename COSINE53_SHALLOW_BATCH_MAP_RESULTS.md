# COSINE53 shallow batch map vs Intel oneMKL VML_HA

## Run

- GitHub Actions run: `33561282465`
- Exact Xeon shard: `1`
- Commit: `ee213b293b103a6ed331f40896e912a088e5d76d`
- CPU: Intel(R) Xeon(R) 6973P-C
- Compiler: Intel oneAPI DPC++/C++ Compiler 2026.1.1 (2026.1.1.20260724)
- Intel comparator: sequential oneMKL `vmdCos(..., VML_HA)`
- CPU affinity: logical CPU 0
- MKL/OMP threads: 1
- Production split used: X50 for `0<|x|<1`; X67 for `1<|x|<500` and `1000<|x|<10000`

## Shallow input design

At each batch size, exactly six deterministic random batch realizations were timed:

1. `unit_pos`: `0<x<1`
2. `unit_neg`: `-1<x<0`
3. `mid_pos`: `1<x<500`
4. `mid_neg`: `-500<x<-1`
5. `far_pos`: `1000<x<10000`
6. `far_neg`: `-10000<x<-1000`

This is intentionally a shallow performance map, not an exhaustive correctness or random-input campaign. Timing repetitions reuse the same six arrays.

`speedup = Intel ns/value / COS53 ns/value`, so values above 1 mean COS53 is faster.

## Results

| Batch n | `|x|<1` COS53 / Intel ns | Speedup | `1<|x|<500` COS53 / Intel ns | Speedup | `1000<|x|<10000` COS53 / Intel ns | Speedup | All-six speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.621121 / 1.048182 | 1.6876x | 1.047865 / 1.068005 | 1.0192x | 1.044235 / 1.065959 | 1.0208x | 1.1728x |
| 250 | 0.557400 / 0.733840 | 1.3165x | 0.612138 / 0.728541 | 1.1902x | 0.618003 / 0.730387 | 1.1819x | 1.2267x |
| 1,200 | 0.538548 / 0.685834 | 1.2735x | 0.470638 / 0.669783 | 1.4231x | 0.470817 / 0.668338 | 1.4195x | 1.3675x |
| 5,000 | 0.538715 / 0.667670 | 1.2394x | 0.458195 / 0.665204 | 1.4518x | 0.457278 / 0.665505 | 1.4554x | 1.3742x |
| 10,000 | 0.537157 / 0.662893 | 1.2341x | 0.457872 / 0.665673 | 1.4538x | 0.460320 / 0.664312 | 1.4432x | 1.3693x |
| 30,000 | 0.536273 / 0.662340 | 1.2351x | 0.455177 / 0.662014 | 1.4544x | 0.454234 / 0.664231 | 1.4623x | 1.3755x |
| 50,000 | 0.536107 / 0.664434 | 1.2394x | 0.456481 / 0.664835 | 1.4564x | 0.458388 / 0.664495 | 1.4496x | 1.3741x |
| 100,000 | 0.538388 / 0.676551 | 1.2566x | 0.461158 / 0.669248 | 1.4512x | 0.463487 / 0.668170 | 1.4416x | 1.3766x |
| 500,000 | 0.549654 / 0.694099 | 1.2628x | 0.541156 / 0.694852 | 1.2840x | 0.546308 / 0.716481 | 1.3115x | 1.2861x |
| 1,000,000 | 0.555664 / 0.685275 | 1.2333x | 0.544496 / 0.695644 | 1.2776x | 0.545861 / 0.714541 | 1.3090x | 1.2730x |
| 2,000,000 | 0.547014 / 0.679452 | 1.2421x | 0.553231 / 0.697103 | 1.2601x | 0.545681 / 0.705031 | 1.2920x | 1.2647x |
| 4,000,000 | 0.561470 / 0.710734 | 1.2658x | 0.574826 / 0.715335 | 1.2444x | 0.639097 / 0.886348 | 1.3869x | 1.3025x |

## Cell result

- COS53 wins: **72 / 72** individual sign/range cells.
- Intel wins: **0 / 72**.
- Every one of the 12 tested batch sizes is a six-for-six COS53 win in this shallow run.

## Interpretation

The current serial/vector COS53 baseline does not show the mid-size loss seen in the EXP53 batch map. In this shallow sample it is already ahead of Intel oneMKL VML_HA at every tested size from 50 through 4,000,000 and in all three magnitude bands.

The wide-domain advantage is strongest across much of the 1.2K-100K region, around 1.42x-1.46x in this run. The all-six average speedup peaks at about 1.38x around 30K-100K, while the 50-element result is a smaller all-domain 1.17x because the two wide bands are nearly tied there.

Because this test intentionally uses only six deterministic random batch realizations per size, it should be treated as a crossover/performance map, not as evidence that every possible input distribution or batch size will produce the same ratios.
