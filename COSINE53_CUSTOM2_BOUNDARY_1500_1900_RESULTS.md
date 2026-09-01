# COSINE53 custom2 boundary sweep: 1500-1900

## Run

- GitHub Actions run: `33564357475`
- Commit: `444f160c05c0615f2663cf6e0ec4419bdff4565e`
- Exact Intel Xeon 6973P-C artifacts: shards `23` and `38`
- Compiler: Intel oneAPI DPC++/C++ Compiler 2026.1.1
- Current COS53: X50 for `0<|x|<1`, X67 for the two wide bands
- custom2: permanent CPU0+CPU2 scheduler, 32-double aligned split
- Intel: sequential oneMKL `vmdCos(..., VML_HA)` on CPU0
- current/custom2/Intel timed in separate processes
- 3 rotated outer repetitions per shard; medians reported

## Input grid

Batch sizes: `1500, 1600, 1700, 1800, 1900`.

At every size, six deterministic random cases were used:

- `unit_pos`, `unit_neg`: `0<|x|<1`
- `mid_pos`, `mid_neg`: `1<|x|<500`
- `far_pos`, `far_neg`: `1000<|x|<10000`

custom2 was bit-identical to current COS53 over the full requested grid on both exact-Xeon artifacts.

## All-six averages — shard 23

| n | Current ns/value | custom2 ns/value | Intel ns/value | Current/custom2 | Intel/custom2 | Best |
|---:|---:|---:|---:|---:|---:|:---|
| 1,500 | 0.495573 | 0.523333 | 0.673752 | 0.9470x | 1.2874x | Current |
| 1,600 | 0.480822 | 0.462827 | 0.666971 | 1.0389x | 1.4411x | custom2 |
| 1,700 | 0.484312 | 0.470161 | 0.670592 | 1.0301x | 1.4263x | custom2 |
| 1,800 | 0.484112 | 0.454690 | 0.667800 | 1.0647x | 1.4687x | custom2 |
| 1,900 | 0.485635 | 0.456574 | 0.668732 | 1.0637x | 1.4647x | custom2 |

Cell-level custom2/current results on shard 23:

- 1,500: custom2 `0/6`
- 1,600: custom2 `6/6`
- 1,700: custom2 `6/6`
- 1,800: custom2 `5/6`
- 1,900: custom2 `6/6`

Overall: custom2 beats current `23/30`; custom2 beats Intel `30/30`; current beats Intel `30/30`.

## All-six averages — shard 38

| n | Current ns/value | custom2 ns/value | Intel ns/value | Current/custom2 | Intel/custom2 | Best |
|---:|---:|---:|---:|---:|---:|:---|
| 1,500 | 0.496444 | 0.511330 | 0.673566 | 0.9709x | 1.3173x | Current |
| 1,600 | 0.480595 | 0.475642 | 0.666805 | 1.0104x | 1.4019x | custom2 |
| 1,700 | 0.488511 | 0.463383 | 0.668030 | 1.0542x | 1.4416x | custom2 |
| 1,800 | 0.483684 | 0.444277 | 0.667303 | 1.0887x | 1.5020x | custom2 |
| 1,900 | 0.486650 | 0.443762 | 0.666815 | 1.0966x | 1.5026x | custom2 |

Cell-level custom2/current results on shard 38:

- 1,500: custom2 `1/6`
- 1,600: custom2 `4/6`
- 1,700: custom2 `6/6`
- 1,800: custom2 `6/6`
- 1,900: custom2 `6/6`

Overall: custom2 beats current `23/30`; custom2 beats Intel `30/30`; current beats Intel `30/30`.

## Replicated interpretation

Both independent exact-Xeon artifacts agree on the all-six winner at every requested size:

- `n=1500`: current COS53 wins
- `n=1600`: custom2 wins, but the margin is small and some wide-domain cells are mixed on shard 38
- `n=1700`: custom2 wins all six individual cells on both shards
- `n=1800`: custom2 wins the all-six average on both shards; one far-negative cell narrowly favors current on shard 23
- `n=1900`: custom2 wins all six individual cells on both shards

Therefore the strongest replicated per-cell boundary from this grid is `n>=1700`. The six-case-average crossover appears by `n=1600`, but `1600` is not a clean 6/6 win on both exact-Xeon artifacts.

The frozen production threshold remains `n>=2000` unless explicitly promoted.
