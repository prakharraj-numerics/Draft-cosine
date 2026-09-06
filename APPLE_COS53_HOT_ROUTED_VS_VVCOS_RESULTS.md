# Apple COS53 corrected hot-routed vs Accelerate vvcos

Workflow run: `34015834744`
Benchmark commit: `989b616eb9ead562950622e4cdb179b4b60e52c7`
Hardware: GitHub macOS-15 arm64 Apple M1 runners
Sampling: 4 independent slots x 3 rounds = 12 samples per stack/batch; medians below.

Candidate: max-2-ULP hot-loop kernel under the previously accepted Apple-specific/hybrid route map. This is the routed comparison, not the single-core diagnostic.

Accuracy on slot 1:
- 9,600 MPFR256 cases: max 2 ULP, 0 >2 ULP.
- 1,000,000 stressed MPFR256 cases: max 2 ULP, 0 >2 ULP.

| n | route | hot wall ns/el | vvcos wall | speed advantage | hot CPU ns/el | vvcos CPU | CPU excess | hot peak RSS MiB | vvcos RSS MiB | RSS delta |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|100|single_utility|1.689|1.428|-15.4%|1.688|1.429|+18.2%|2.53|2.52|+0.6%|
|400|wg2_ui|1.165|1.377|+18.1%|2.245|1.376|+63.2%|2.66|2.53|+4.9%|
|700|wg2_utility|1.168|1.361|+16.5%|2.260|1.336|+69.2%|2.68|2.56|+4.6%|
|1200|wg2_ui|1.161|1.364|+17.5%|1.992|1.357|+46.8%|2.66|2.61|+1.8%|
|3000|wg2_default|1.236|1.378|+11.5%|1.905|1.353|+40.8%|2.89|2.77|+4.5%|
|5000|wg3_utility|2.075|1.369|-34.0%|4.040|1.369|+195.1%|3.25|3.09|+5.1%|
|7500|auto|0.867|1.354|+56.3%|1.553|1.353|+14.7%|3.45|3.28|+5.2%|
|15000|wg3_user|0.608|1.357|+123.0%|1.785|1.339|+33.3%|4.18|4.03|+3.7%|
|29999|wg3_utility|0.616|1.359|+120.7%|1.884|1.360|+38.5%|5.48|5.34|+2.6%|
|30000|pool3|0.593|1.409|+137.5%|1.810|1.401|+29.2%|5.41|5.34|+1.2%|
|40000|wg3_user|2.268|1.374|-39.4%|2.445|1.374|+77.9%|6.40|6.28|+1.9%|
|40001|wg2_default|0.923|1.331|+44.3%|1.848|1.332|+38.8%|6.41|6.28|+2.0%|
|50000|wg2_user|0.900|1.340|+48.9%|1.786|1.341|+33.2%|7.34|7.22|+1.7%|
|100000|auto|0.572|1.310|+129.1%|1.730|1.309|+32.1%|11.91|11.72|+1.6%|
|200000|pool3|0.576|1.319|+129.1%|1.836|1.319|+39.1%|21.00|20.91|+0.4%|
|500000|wg3_default|0.890|1.344|+51.0%|2.207|1.341|+64.6%|48.61|48.47|+0.3%|
|1000000|wg3_ui|1.124|1.382|+23.0%|2.437|1.378|+76.8%|94.36|94.30|+0.1%|

Summary for this exact route map:
- Faster than `vvcos` at 14/17 sizes.
- Losses: 100, 5K and 40K.
- 5K and 40K are Workgroup scheduler regressions and must not replace the older frozen hybrid winners. In the prior frozen-hybrid benchmark, 5K was 15.7% faster than `vvcos` and 40K was 36.9% faster.
- Geometric-mean wall speedup `vvcos / hot-routed`: 1.378x, despite the two Workgroup regressions.
- Geometric-mean process-CPU ratio hot-routed / `vvcos`: 1.498x. Route-level CPU is still materially worse because multicore parallelism buys wall speed.
- Geometric-mean peak-RSS ratio: 1.025x; memory footprint is close to parity.
- Hot common kernel logical bytes: 32 B/el (8 input + 16 coefficient loads + 8 output); rare root repair adds marginal extra traffic.
- Retired instructions/cycles cannot be measured on the GitHub-hosted M1: Instruments CPU Counters reports hardware counters unsupported and powermetrics returns zero Instr/s/Cycles/s.

Preservation policy: new kernels/schedulers should replace a frozen route only at batch sizes where they beat the existing frozen winner under the required accuracy and resource constraints.
