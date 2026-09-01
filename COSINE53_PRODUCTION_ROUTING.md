# COSINE53 frozen batch routing

Current production routing is frozen as:

- `n < 5000`: current COS53 evaluator
- `n >= 5000`: frozen custom permanent 2-core scheduler

The rule is backed by GitHub Actions run `33562041646` on exact Intel Xeon 6973P-C, shard `41`.

In that run, custom2 output was bit-identical to the current COS53 evaluator across all 72 requested size/sign/range cells. At every tested size from 5,000 through 4,000,000, custom2 beat current COS53 in all six requested input cases.

The exact crossover between 1,200 and 5,000 was not measured. The 5,000 threshold is therefore a conservative tested production boundary, not a claim that 5,000 is the mathematical or hardware-optimal crossover.

Frozen production files:

- `cosine53_batch_production.hpp`
- `cosine53_custom_2core_5000_frozen.hpp`

Do not change this threshold or the frozen scheduler without a new benchmark and explicit production promotion.
