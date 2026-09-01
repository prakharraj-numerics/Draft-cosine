# COSINE53 frozen batch routing

Current production routing is frozen as:

- `n < 2000`: current COS53 evaluator
- `n >= 2000`: frozen custom permanent 2-core scheduler

The routing is backed by two exact Intel Xeon 6973P-C benchmarks:

- broad three-way run `33562041646`, shard `41`
- focused boundary run `33563237026`, shard `5`

In the boundary run, custom2 was bit-identical to the current COS53 evaluator over the tested grid. At every tested point from 2,000 through 4,500, custom2 beat current COS53 in all six requested sign/range cases. At 1,500, current COS53 remained slightly faster on the all-six average.

The broader run independently showed custom2 beating current COS53 in all six cases at every tested size from 5,000 through 4,000,000.

Therefore the frozen production rule is a conservative evidence-backed threshold at 2,000. No lower crossover is inferred from these measurements.

Frozen production files:

- `cosine53_batch_production.hpp`
- `cosine53_custom_2core_2000_frozen.hpp`

The previous `cosine53_custom_2core_5000_frozen.hpp` remains only as historical freeze evidence and is no longer used by the production dispatcher.

Do not change the 2,000 threshold or frozen scheduler without a new benchmark and explicit production promotion.
