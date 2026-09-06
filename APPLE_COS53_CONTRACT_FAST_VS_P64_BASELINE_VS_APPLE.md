# Apple COS53 `ffp-contract=fast` attack vs current p64 baseline vs Apple

Benchmark run: `34020555973`

Attack branch derives from current baseline SHA `9ef125b86cbc996686daab5d3c7ebaa1b629e80d`, which includes the p64 route for `78,000 <= n <= 82,000`.

Method: 4 independent GitHub `macos-15` Apple M1 jobs x 3 rounds = 12 samples per stack/batch. Order rotated among attack, baseline and Apple `vvcos`.

Compiler attack: replace `-ffp-contract=off` with `-ffp-contract=fast`; keep `-O3 -DNDEBUG -std=c++20 -mcpu=native -fno-fast-math`.

Accuracy on slot 1 was bit-for-bit identical between baseline and attack:

- 9600 MPFR256: exact 4046, <=1 ULP 9241, <=2 ULP 9600, max 2 ULP.
- genuine 1M stress: exact 441685, <=1 ULP 962384, <=2 ULP 1000000, max 2 ULP.

`-ffast-math` and `-Ofast` were rejected in the preceding sweep because both catastrophically failed the accuracy gate.

## 12-sample medians

| n | baseline wall ns/el | attack wall ns/el | attack speed gain vs baseline | Apple wall ns/el | Apple/attack wall | baseline CPU ns/el | attack CPU ns/el | attack CPU change vs baseline | Apple CPU ns/el | attack/Apple CPU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|100|1.6459|1.3892|+18.47%|1.3724|0.988x|1.6454|1.3894|-15.56%|1.3721|1.013x|
|400|1.1134|0.9532|+16.81%|1.2887|1.352x|2.0903|1.7876|-14.48%|1.2888|1.387x|
|700|0.9805|0.8481|+15.60%|1.2901|1.521x|1.8155|1.5929|-12.26%|1.2887|1.236x|
|1200|0.9500|0.8051|+18.00%|1.2509|1.554x|1.7873|1.4613|-18.24%|1.2510|1.168x|
|3000|0.9654|0.9348|+3.28%|1.2338|1.320x|1.8733|1.8255|-2.55%|1.2339|1.479x|
|5000|0.8821|0.7515|+17.38%|1.2194|1.623x|1.6346|1.4156|-13.40%|1.2195|1.161x|
|7500|0.8292|0.7230|+14.69%|1.2303|1.702x|1.5008|1.3621|-9.24%|1.2303|1.107x|
|15000|0.8197|0.6900|+18.80%|1.2200|1.768x|1.4725|1.3366|-9.23%|1.2201|1.096x|
|29999|0.8936|0.8700|+2.71%|1.2272|1.411x|1.7420|1.6110|-7.52%|1.2273|1.313x|
|30000|0.6256|0.5928|+5.53%|1.2272|2.070x|1.8556|1.7613|-5.08%|1.2263|1.436x|
|40000|0.5749|0.4674|+23.01%|1.2710|2.720x|1.8030|1.3643|-24.33%|1.2710|1.073x|
|40001|0.9213|0.8859|+4.00%|1.2618|1.424x|1.8462|1.7520|-5.10%|1.2606|1.390x|
|50000|0.8449|0.7013|+20.47%|1.2992|1.853x|1.6863|1.3742|-18.51%|1.2940|1.062x|
|78000|0.8328|0.7773|+7.14%|1.2762|1.642x|2.0572|1.7987|-12.57%|1.2762|1.409x|
|80000|0.8393|0.7251|+15.76%|1.2562|1.733x|2.0835|1.7759|-14.76%|1.2562|1.414x|
|82000|0.8116|0.7225|+12.33%|1.2593|1.743x|2.0443|1.8728|-8.39%|1.2571|1.490x|
|100000|0.6105|0.5819|+4.93%|1.2772|2.195x|1.5211|1.8412|+21.04%|1.2770|1.442x|
|200000|0.6133|0.5851|+4.83%|1.2427|2.124x|1.9421|1.8970|-2.32%|1.2428|1.526x|
|500000|0.7188|0.7198|-0.13%|1.2959|1.800x|1.9932|1.9597|-1.68%|1.2931|1.516x|
|1000000|0.6982|0.6956|+0.37%|1.2924|1.858x|1.9768|1.9332|-2.21%|1.2880|1.501x|

Across all 20 measured points, applying `contract_fast` globally gives geometric-mean baseline/attack wall speedup `1.1096x`, baseline/attack CPU ratio `1.1019x`, Apple/attack wall speedup `1.6823x`, and attack/Apple CPU ratio `1.2991x`.

The attack wins wall at 19/20 points, CPU at 19/20, and both simultaneously at 18/20.

For a no-speed-regression/no-CPU-regression production candidate, keep the exact baseline routes at 100K (attack CPU regresses) and 500K (attack wall regresses slightly), and use `contract_fast` elsewhere. That conservative hybrid has estimated geometric means from the same measured medians: baseline/hybrid wall `1.1070x`, baseline/hybrid CPU `1.1115x`, Apple/hybrid wall `1.6783x`, hybrid/Apple CPU `1.2879x`.
