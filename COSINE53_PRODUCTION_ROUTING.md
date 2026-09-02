# COSINE53 frozen batch routing

Current production routing is frozen as:

- `n < 4500`: current COS53 evaluator; custom2 helper is not constructed
- `n >= 4500`: construct/use the frozen permanent 2-core custom2 scheduler
- after first construction, the helper remains alive for subsequent large batches

This is a scheduling/lifecycle change only. The COS53 mathematics and the frozen custom2 scheduler are unchanged.

## Promotion evidence

The 4.5K lifecycle boundary was explicitly promoted after the full exact-Xeon benchmark:

- GitHub Actions run `33690497495`
- exact Intel Xeon 6973P-C artifacts: shards `3` and `20`
- tested sizes: `100, 700, 3500, 4500, 5000, 8000, 15000, 50000, 1000000, 2000000`
- native metrics: wall time, process CPU time, effective cores, RSS, and accuracy
- SDE metrics: retired instructions per element and logical memory bytes per element

The resource-elastic region was especially strong at `n=100`, `700`, and `3500`, where the helper stayed absent. At `n=4500` the helper/custom2 path activates exactly as benchmarked.

Observed maximum comparator difference versus Intel `vmdCos(..., VML_HA)` was no more than 2 ULP over the tested map.

## Historical boundary

The prior production boundary was `1600`, backed by runs `33562041646`, `33563237026`, and `33564357475`. That boundary remains useful historical speed-crossover evidence, but it was superseded for production by the later resource-proportionality work: eagerly creating the permanent helper caused severe small-load CPU/instruction/memory overhead even when the current evaluator handled the batch.

## Frozen production files

- `cosine53_batch_production.hpp`
- `cosine53_custom_2core_1600_frozen.hpp`
- `cosine53_x50_unit_production.c`
- `cosine53_x67_wide_production.c`

The custom2 implementation filename still contains `1600` because that scheduler body itself remains the previously frozen implementation. The production dispatcher now controls when it is created and used.

Do not change the `<4500` lazy region, the `4500` activation boundary, the frozen scheduler, or the production COS53 kernels without a new benchmark and explicit promotion.
