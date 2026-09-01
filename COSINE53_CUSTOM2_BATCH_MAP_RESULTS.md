# COSINE53 custom2-core batch map vs current and Intel VML_HA

## Run

- GitHub Actions run: `33562041646`
- Exact Xeon shard: `41`
- Artifact: `cosine53-custom2-map-41`
- Commit: `ced6353f6f375b7d8ad8c4c9879c7e19fbdcf5cc`
- CPU: Intel(R) Xeon(R) 6973P-C
- Compiler: Intel oneAPI DPC++/C++ Compiler 2026.1.1
- Current COS53: X50 for `0<|x|<1`, X67 for the two wide bands
- custom2: permanent caller CPU0 + helper CPU2, 32-double-aligned split, release/acquire handoff, no queue/work stealing/per-call thread creation
- Intel comparator: sequential oneMKL `vmdCos(..., VML_HA)` on CPU0
- current, custom2, and Intel timings ran in separate processes
- three outer repetitions with rotated mode order; reported cells are medians of the three per-process medians

## Synchronization

custom2 was checked bit-for-bit against the current COS53 evaluator before timing.

- X50 unit domain: total bit differences = 0
- X67 wide domains: total bit differences = 0
- Combined requested map: **0 differences across all 72 size/sign/range cells**

Thus the custom2 experiment changes scheduling only, not the computed binary64 results.

## Input map

At every batch size the same six deterministic random cases were used:

- `unit_pos`, `unit_neg`: `0<|x|<1`
- `mid_pos`, `mid_neg`: `1<|x|<500`
- `far_pos`, `far_neg`: `1000<|x|<10000`

Batch sizes: `50, 250, 1200, 5000, 10000, 30000, 50000, 100000, 500000, 1000000, 2000000, 4000000`.

Ratios below are throughput speedups in the natural direction:

- `current/custom2 > 1`: custom2 faster than current
- `Intel/custom2 > 1`: custom2 faster than Intel
- `Intel/current > 1`: current faster than Intel

## All-six average result

| Batch n | Current ns/value | custom2 ns/value | Intel ns/value | Current/custom2 | Intel/custom2 | Intel/current | Best |
|---:|---:|---:|---:|---:|---:|---:|:---|
| 50 | 0.992182 | 1.001711 | 1.155271 | 0.990x | 1.153x | 1.164x | Current |
| 250 | 0.634038 | 2.024151 | 0.827995 | 0.313x | 0.409x | 1.306x | Current |
| 1,200 | 0.522887 | 0.612137 | 0.763132 | 0.854x | 1.247x | 1.459x | Current |
| 5,000 | 0.515561 | 0.334353 | 0.753729 | **1.542x** | **2.254x** | 1.462x | custom2 |
| 10,000 | 0.516440 | 0.300082 | 0.756961 | **1.721x** | **2.523x** | 1.466x | custom2 |
| 30,000 | 0.511775 | 0.279555 | 0.767268 | **1.831x** | **2.745x** | 1.499x | custom2 |
| 50,000 | 0.516048 | 0.271752 | 0.767509 | **1.899x** | **2.824x** | 1.487x | custom2 |
| 100,000 | 0.525760 | 0.268233 | 0.770341 | **1.960x** | **2.872x** | 1.465x | custom2 |
| 500,000 | 0.575674 | 0.297317 | 0.778582 | **1.936x** | **2.619x** | 1.352x | custom2 |
| 1,000,000 | 0.554647 | 0.300175 | 0.767772 | **1.848x** | **2.558x** | 1.384x | custom2 |
| 2,000,000 | 0.552772 | 0.283580 | 0.768931 | **1.949x** | **2.712x** | 1.391x | custom2 |
| 4,000,000 | 0.551650 | 0.290006 | 0.768111 | **1.902x** | **2.649x** | 1.392x | custom2 |

## Domain-specific custom2 speedup

| Batch n | Unit current/custom2 | Mid current/custom2 | Far current/custom2 | Unit Intel/custom2 | Mid Intel/custom2 | Far Intel/custom2 |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.980x | 0.993x | 0.994x | 1.592x | 1.014x | 1.013x |
| 250 | 0.288x | 0.328x | 0.323x | 0.415x | 0.414x | 0.398x |
| 1,200 | 0.848x | 0.852x | 0.863x | 1.186x | 1.279x | 1.282x |
| 5,000 | 1.506x | 1.572x | 1.553x | 2.088x | 2.371x | 2.330x |
| 10,000 | 1.688x | 1.734x | 1.745x | 2.334x | 2.634x | 2.628x |
| 30,000 | 1.802x | 1.863x | 1.831x | 2.490x | 2.889x | 2.894x |
| 50,000 | 1.895x | 1.910x | 1.893x | 2.583x | 2.968x | 2.957x |
| 100,000 | 1.971x | 1.966x | 1.942x | 2.645x | 3.017x | 2.985x |
| 500,000 | 1.937x | 2.007x | 1.867x | 2.427x | 2.777x | 2.679x |
| 1,000,000 | 1.805x | 1.858x | 1.884x | 2.398x | 2.643x | 2.648x |
| 2,000,000 | 1.958x | 1.936x | 1.954x | 2.609x | 2.739x | 2.792x |
| 4,000,000 | 1.983x | 1.791x | 1.938x | 2.680x | 2.549x | 2.724x |

## Cell wins

Across all 72 individual sign/range cells:

- custom2 beats current: **54 / 72**
- custom2 beats Intel: **66 / 72**
- current beats Intel: **72 / 72**

The 54 custom2-vs-current wins are exactly all six cells at every requested batch size from **5,000 through 4,000,000**. custom2 loses all 18 current comparisons at `50`, `250`, and `1200`.

Against Intel, custom2 loses only the six `n=250` cells. At `n=50` the custom2 implementation falls back to a single caller because there are fewer than two full 32-value blocks, and still remains ahead of Intel in all six cells.

## Routing implication from this requested grid

For the tested points, the evidence-backed routing is:

- `n <= 1200`: retain current COS53
- `n >= 5000`: custom2 is the clear winner

The exact crossover between 1200 and 5000 was not measured in this run and should not be invented. A focused boundary sweep is required before choosing a production threshold.
