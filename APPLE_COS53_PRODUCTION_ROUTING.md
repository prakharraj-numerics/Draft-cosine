# Apple COS53 production routing

Frozen Apple routing baseline, 2026-09-06.

The math kernel is unchanged: frozen Google Highway Apple COS53 K=2048, degree=3, terms=1 path with the established MPFR256 validation state unchanged. This update changes orchestration only.

Production orchestration:

| Batch size | Route |
|---:|:---|
| n < 5,000 | existing frozen `AppleTwoCoreHighway` path |
| 5,000 <= n < 30,000 | `pthreadpool`, 2 threads / exactly 2 balanced contiguous tasks |
| 30,000 <= n <= 40,000 | Halide runtime scheduler, 32 pieces, 3 threads |
| 40,001 <= n < 50,000 | `dispatch_apply_f` + `DISPATCH_APPLY_AUTO`, 32 pieces |
| 50,000 <= n < 100,000 | `dispatch_apply_f` + `DISPATCH_APPLY_AUTO`, 16 pieces |
| 100,000 <= n < 1,000,000 | `dispatch_apply_f` + `DISPATCH_APPLY_AUTO`, 12 pieces |
| n >= 1,000,000 | `dispatch_apply_f` + `DISPATCH_APPLY_AUTO`, 24 pieces |

Implementation: `apple_cos53_production_routing.hpp`.

## Frozen pthreadpool contract for 5K-30K

The baseline is the same scheduler variant that won the Apple M1 comparison run `33989147927`:

- Maratyszcza `pthreadpool`.
- `pthreadpool_create(2)`.
- Apple build uses `PTHREADPOOL_SYNC_PRIMITIVE=condvar`; do not substitute pthreadpool's default Apple GCD backend when reproducing this baseline.
- Exactly two contiguous COS53 chunks per call.
- Split point is `(n/2) & ~1`, preserving the 2-double Highway vector boundary.
- Flags are `0`; workers are not explicitly yielded after each call.
- The COS53 evaluator and coefficients are unchanged.

The 12-sample M1 medians from that experiment favored pthreadpool over the prior helper at 5K, 7.5K, 10K, 20K, 25K and nominally 29,999. The user selected the whole 5K <= n < 30K region as the frozen working baseline for now; no claim is made that every integer batch size inside the interval was individually optimized.

The stable dispatch callback above 40K uses balanced integer partitions `a=(n*j)/pieces`, `b=(n*(j+1))/pieces`, and invokes the frozen Highway evaluator on each slice. The routing layer does not replace or alter the COS53 formula.

Intel/Xeon production is outside this routing file and remains untouched.
