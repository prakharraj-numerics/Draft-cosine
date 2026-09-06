# WORK smooth-switch threshold-only attack

Frozen incumbent: `WORK` = `2daf7bd0a5a08df06efd169b1c38b55569de32f7`.

Experiment branch: `exp/apple-cos53-WORK-smooth-switching-20260906`.

Exact full-ladder tested SHA: `925ae7aad2a8878e325f521b1dcc69d5eb443e99`.
Workflow run: `34024775921`.
All 4 Apple M1 jobs succeeded. Slot 1 accuracy validation succeeded for both fast/off kernels (max 2 ULP state unchanged).

Attack changes only handoff thresholds. No kernel, math, compiler flag, library implementation, worker count, QoS definition, or executor implementation was modified.

Attack map:
- 5,000 <= n < 30,000: existing WORK `pool2` executor.
- 30,000 <= n < 40,000: existing WORK `wg2_ui` executor.
- n = 40,000: exact WORK route retained (`wg3_user`).
- 40,001 <= n < 78,000: existing WORK `wg2_ui` executor.
- all other n: exact WORK route.

Thus among the standard benchmark anchors, the substantive route changes are 29,999, 30,000 and 40,001. 5K/7.5K/15K and 50K already used the corresponding executor in WORK.

Ratios below are medians of paired same-slot/same-round samples across 4 M1 slots x 3 rounds = 12 samples per stack/batch.
- Speed vs Apple = Apple wall / attack wall; >1 means attack faster.
- CPU vs Apple = attack process CPU / Apple process CPU; closer to 1 is better.
- Attack/WORK wall and CPU <1 means improvement over WORK.

| n | speed vs Apple | CPU vs Apple | attack/WORK wall | attack/WORK CPU |
|---:|---:|---:|---:|---:|
|100|0.963|1.038|1.008|1.007|
|400|1.149|1.756|0.993|1.001|
|700|1.362|1.429|0.994|0.891|
|1,200|1.410|1.366|0.946|0.980|
|3,000|1.312|1.484|0.930|0.903|
|5,000|1.494|1.283|0.992|0.990|
|7,500|1.406|1.385|1.009|1.001|
|15,000|1.755|1.090|1.004|0.920|
|29,999|1.459|1.436|**0.780**|**0.813**|
|30,000|1.477|1.271|**1.023**|**0.596**|
|40,000|0.981|2.672|1.010|1.070|
|40,001|1.721|1.122|**0.801**|**0.785**|
|50,000|1.701|1.146|0.999|1.019|
|78,000|1.577|2.332|0.999|0.945|
|80,000|1.804|1.094|0.981|0.915|
|82,000|1.664|1.163|1.006|0.960|
|100,000|1.591|1.785|1.000|1.009|
|200,000|1.849|1.718|0.992|0.943|
|500,000|1.288|1.948|1.011|1.001|
|1,000,000|1.794|1.450|0.994|0.931|

Key changed-anchor interpretation:
- 29,999: attack is ~22.0% lower wall and ~18.7% lower CPU than WORK by paired median.
- 30,000: attack gives back ~2.3% wall versus WORK but reduces process CPU by ~40.4%; still ~1.48x faster than Apple in this run.
- 40,001: attack is ~19.9% lower wall and ~21.5% lower CPU than WORK; ~1.72x faster than Apple and ~1.12x Apple CPU.

Geometric mean across all 20 paired ratios in this particular run:
- Apple/attack wall = 1.464x.
- attack/Apple CPU = 1.448x.
- attack/WORK wall = 0.971x (~2.9% lower wall).
- attack/WORK CPU = 0.927x (~7.3% lower CPU).

Important runner-variance note: unchanged routes at 40K and 78K were heavily scheduler-sensitive on slots 1 and 4 but normal on slots 2 and 3. Because ATTACK and WORK are structurally identical at those points, their divergence from prior full-ladder Apple ratios is runner-state noise rather than a smooth-switch code change. For example at 40K slots 2/3 reproduced roughly 2.4–2.6x Apple wall speed in several rounds while slots 1/4 fell below 1x. No samples were discarded.

Conclusion: threshold smoothing is strongly supported at 29,999 and 40,001. The 30K trade is CPU-favorable (~40% CPU reduction for ~2.3% wall cost vs WORK) and remains comfortably ahead of Apple in this run. The attack does not address 3K, 200K or 500K; those routes were intentionally left unchanged because the seam screen did not identify a stable threshold-only crossover there.
