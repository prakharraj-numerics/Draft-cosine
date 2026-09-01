# COSINE53 frozen batch routing

Current production routing is frozen as:

- `n < 1600`: current COS53 evaluator
- `n >= 1600`: frozen custom permanent 2-core scheduler

The routing is backed by exact Intel Xeon 6973P-C benchmarks:

- broad three-way run `33562041646`, shard `41`
- focused 1500-4500 boundary run `33563237026`, shard `5`
- replicated 1500-1900 boundary run `33564357475`, exact Xeon shards `23` and `38`

custom2 was bit-identical to the current COS53 evaluator over the tested maps.

The replicated 1500-1900 run established the final production boundary used here:

- at `n=1500`, current COS53 won the all-six average on both exact-Xeon shards
- at `n=1600`, custom2 won the all-six average on both exact-Xeon shards; the margin was small and some individual cells were mixed
- at `n=1700`, custom2 won all six individual cells on both shards
- at `n=1800`, custom2 won the all-six average on both shards, with one narrow individual-cell loss on shard 23
- at `n=1900`, custom2 won all six individual cells on both shards

The earlier focused and broad runs showed custom2 ahead throughout the tested range from 2,000 through 4,000,000.

The user explicitly promoted `1600` as the final production boundary. It is therefore frozen as a measured average crossover boundary, not as a claim that every individual input case is faster exactly at `n=1600`.

Frozen production files:

- `cosine53_batch_production.hpp`
- `cosine53_custom_2core_1600_frozen.hpp`

The older `cosine53_custom_2core_2000_frozen.hpp` and `cosine53_custom_2core_5000_frozen.hpp` remain only as historical freeze evidence and are no longer used by the production dispatcher.

Do not change the 1,600 threshold or frozen scheduler without a new benchmark and explicit production promotion.
