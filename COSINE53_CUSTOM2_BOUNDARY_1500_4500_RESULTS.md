# COSINE53 custom2 boundary sweep: 1500-4500

## Run

- GitHub Actions run: `33563237026`
- Exact Xeon shard: `5`
- Artifact: `cosine53-custom2-boundary-5`
- Commit: `c8831661cabe8fddc238ef7957efecd70acb6bb4`
- CPU: Intel(R) Xeon(R) 6973P-C
- Compiler: Intel oneAPI DPC++/C++ Compiler 2026.1.1
- Current COS53: X50 for `0<|x|<1`, X67 for the two wide bands
- custom2: permanent CPU0+CPU2 scheduler, 32-double aligned split
- Intel: sequential oneMKL `vmdCos(..., VML_HA)` on CPU0
- current/custom2/Intel timed in separate processes
- 3 rotated outer repetitions; medians reported

## Input grid

Batch sizes: `1500, 2000, 2500, 3000, 3500, 4000, 4500`.

At each size six deterministic random cases were used:

- `unit_pos`, `unit_neg`: `0<|x|<1`
- `mid_pos`, `mid_neg`: `1<|x|<500`
- `far_pos`, `far_neg`: `1000<|x|<10000`

custom2 was bit-identical to current COS53 over the full boundary grid.

## All-six averages

| n | Current ns/value | custom2 ns/value | Intel ns/value | Current/custom2 | Intel/custom2 | Intel/current | Best |
|---:|---:|---:|---:|---:|---:|---:|:---|
| 1,500 | 0.508425 | 0.511069 | 0.684855 | 0.9948x | 1.3400x | 1.3470x | Current |
| 2,000 | 0.497784 | 0.432024 | 0.680230 | 1.1522x | 1.5745x | 1.3665x | custom2 |
| 2,500 | 0.498510 | 0.391368 | 0.674552 | 1.2738x | 1.7236x | 1.3531x | custom2 |
| 3,000 | 0.496828 | 0.373012 | 0.667296 | 1.3319x | 1.7889x | 1.3431x | custom2 |
| 3,500 | 0.495679 | 0.349974 | 0.677249 | 1.4163x | 1.9351x | 1.3663x | custom2 |
| 4,000 | 0.492056 | 0.333569 | 0.677961 | 1.4751x | 2.0324x | 1.3778x | custom2 |
| 4,500 | 0.494753 | 0.330553 | 0.673721 | 1.4967x | 2.0382x | 1.3617x | custom2 |

## Cell result

Across 42 sign/range cells:

- custom2 beats current: **38/42**
- custom2 beats Intel: **42/42**
- current beats Intel: **42/42**

At `n=1500`, custom2 wins only the two unit-domain cells; current wins the four wide-domain cells, and current is about 0.5% faster on the all-six average.

At every tested point from `n=2000` through `n=4500`, custom2 wins **all six cells** against current and Intel.

## Production promotion

Following this boundary run, the production threshold was explicitly promoted and frozen as:

- `n < 2000`: current COS53 evaluator
- `n >= 2000`: custom2-core scheduler

The production dispatcher is `cosine53_batch_production.hpp`, using `cosine53_custom_2core_2000_frozen.hpp`.
