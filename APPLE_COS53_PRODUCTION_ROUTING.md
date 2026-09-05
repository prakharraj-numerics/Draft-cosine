# Apple COS53 production routing

Frozen Apple routing state, 2026-09-05.

The math kernel is unchanged: frozen Google Highway Apple COS53 K=2048, degree=3, terms=1 path with the established 9600-case MPFR256 validation state at max <=2 ULP.

Production orchestration:

| Batch size | Route |
|---:|:---|
| n < 30,000 | existing frozen `AppleTwoCoreHighway` path |
| 30,000 <= n < 50,000 | `dispatch_apply_f` + `DISPATCH_APPLY_AUTO`, 32 pieces |
| 50,000 <= n < 100,000 | `dispatch_apply_f` + `DISPATCH_APPLY_AUTO`, 16 pieces |
| 100,000 <= n < 1,000,000 | `dispatch_apply_f` + `DISPATCH_APPLY_AUTO`, 12 pieces |
| n >= 1,000,000 | `dispatch_apply_f` + `DISPATCH_APPLY_AUTO`, 24 pieces |

Implementation: `apple_cos53_production_routing.hpp`.

The stable dispatch callback uses balanced integer partitions `a=(n*j)/pieces`, `b=(n*(j+1))/pieces`, and invokes the frozen Highway evaluator on each slice. The routing layer changes orchestration only; it does not replace or alter the COS53 formula.

Benchmark evidence used for this routing comes from the stabilized dispatch sweep and final 1M confirmation. The 1M confirmation used four independent Apple M1 GitHub-hosted slots and 10 outer repetitions per partition count; p24 was the final aggregate winner.

Intel/Xeon production is outside this routing file and remains untouched.
