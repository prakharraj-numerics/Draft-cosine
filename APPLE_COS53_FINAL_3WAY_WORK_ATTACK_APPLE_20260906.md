# Apple COS53 final 3-way benchmark: WORK vs ATTACK vs Apple M1

Exact benchmarked workflow SHA: `d3d65b8c77653bd6ca04cb92c2b82cc47ae58e88`.
Workflow run: `34026337381`.
Frozen WORK SHA: `2daf7bd0a5a08df06efd169b1c38b55569de32f7`.
Exact ATTACK wrapper lineage: `925ae7aad2a8878e325f521b1dcc69d5eb443e99`; wrapper blob `f33a3187b191c076397f433c56111b2f041dfa4a`.

All 4 Apple M1 jobs succeeded. The workflow proved that relative to frozen WORK, the only implementation addition was the exact smooth-switch attack wrapper (plus the workflow YAML), and asserted both WORK and ATTACK routes before timing. Slot 1 accuracy validation passed unchanged for fast/off kernels: 9,600 cases max 2 ULP and 1,000,000 stress cases max 2 ULP.

Each runner measured WORK, ATTACK, and Apple vvcos interleaved for 3 rounds at all 20 batch sizes. Ratios are paired same-slot/same-round ratios, then medians within the stated population. Speed/Apple = Apple wall / candidate wall (>1 is faster than Apple). CPU/Apple = candidate process CPU / Apple process CPU (closer to 1 is better).

## Runner-state classification

| slot | WORK speed GM vs Apple | WORK CPU GM vs Apple | ATTACK speed GM vs Apple | ATTACK CPU GM vs Apple | classification |
|---:|---:|---:|---:|---:|---|
| 1 | 1.724x | 1.216x | 1.799x | 1.160x | good / normal |
| 2 | 1.353x | 1.519x | 1.259x | 1.560x | degraded / mild outlier |
| 3 | 0.935x | 2.087x | 0.973x | 1.953x | severe outlier |
| 4 | 0.938x | 2.009x | 0.941x | 2.019x | severe outlier |

Slot 1 is the only normal-state runner in this run and matches the known WORK envelope. Slots 2-4 are grouped as degraded/outlier for segregation; slot 2 is materially less severe than slots 3-4.

## GOOD population: slot 1, 3 paired samples per batch

| n | WORK speed/Apple | ATTACK speed/Apple | WORK CPU/Apple | ATTACK CPU/Apple |
|---:|---:|---:|---:|---:|
| 100 | 0.985x | 0.997x | 1.015x | 1.006x |
| 400 | 1.350x | 1.337x | 1.396x | 1.363x |
| 700 | 1.495x | 1.520x | 1.313x | 1.283x |
| 1,200 | 1.649x | 1.576x | 1.175x | 1.172x |
| 3,000 | 1.374x | 1.371x | 1.363x | 1.469x |
| 5,000 | 1.634x | 1.648x | 1.158x | 1.126x |
| 7,500 | 1.741x | 1.756x | 1.086x | 1.081x |
| 15,000 | 1.824x | 1.796x | 1.098x | 1.047x |
| 29,999 | 1.454x | 1.829x | 1.389x | 1.147x |
| 30,000 | 2.184x | 1.825x | 1.257x | 1.133x |
| 40,000 | 2.749x | 2.823x | 1.097x | 1.206x |
| 40,001 | 1.430x | 1.841x | 1.404x | 1.108x |
| 50,000 | 1.852x | 1.821x | 1.020x | 1.069x |
| 78,000 | 2.390x | 2.753x | 1.448x | 1.236x |
| 80,000 | 1.874x | 1.861x | 0.995x | 0.982x |
| 82,000 | 1.860x | 1.845x | 1.079x | 1.061x |
| 100,000 | 2.297x | 2.365x | 1.305x | 1.198x |
| 200,000 | 2.221x | 2.749x | 1.268x | 1.223x |
| 500,000 | 1.877x | 1.868x | 1.473x | 1.411x |
| 1,000,000 | 2.353x | 2.697x | 1.140x | 1.016x |

Good-population geometric means across all observations:
- WORK: 1.724x Apple speed; 1.216x Apple CPU.
- ATTACK: 1.799x Apple speed; 1.160x Apple CPU.

The ATTACK changes routes at standard anchors 29,999, 30,000, and 40,001 only. On the good runner, paired median ATTACK-vs-WORK at those points:
- 29,999: ATTACK speed multiplier vs WORK 1.258x; ATTACK/WORK CPU 0.815x.
- 30,000: ATTACK speed multiplier vs WORK 0.835x; ATTACK/WORK CPU 0.901x.
- 40,001: ATTACK speed multiplier vs WORK 1.287x; ATTACK/WORK CPU 0.808x.

At unchanged-route anchors, WORK and ATTACK differences are scheduler/timing variation because they invoke the same underlying WORK route.

## DEGRADED / OUTLIER population: slots 2-4, 9 paired samples per batch

| n | WORK speed/Apple | ATTACK speed/Apple | WORK CPU/Apple | ATTACK CPU/Apple |
|---:|---:|---:|---:|---:|
| 100 | 0.924x | 1.021x | 1.087x | 0.979x |
| 400 | 0.898x | 1.331x | 1.857x | 1.462x |
| 700 | 0.862x | 1.102x | 2.025x | 1.812x |
| 1,200 | 1.141x | 0.830x | 1.378x | 2.134x |
| 3,000 | 1.094x | 0.909x | 1.879x | 2.215x |
| 5,000 | 1.448x | 1.324x | 1.407x | 1.284x |
| 7,500 | 1.706x | 1.549x | 1.131x | 1.145x |
| 15,000 | 1.101x | 1.708x | 1.927x | 1.258x |
| 29,999 | 1.057x | 0.830x | 1.641x | 2.057x |
| 30,000 | 0.713x | 1.116x | 4.390x | 1.614x |
| 40,000 | 0.767x | 0.811x | 2.083x | 3.217x |
| 40,001 | 0.799x | 1.343x | 2.067x | 1.286x |
| 50,000 | 1.689x | 1.168x | 1.132x | 1.562x |
| 78,000 | 0.835x | 0.943x | 2.821x | 2.390x |
| 80,000 | 1.437x | 1.069x | 1.289x | 1.643x |
| 82,000 | 1.513x | 0.981x | 1.219x | 1.654x |
| 100,000 | 1.288x | 1.072x | 2.288x | 2.840x |
| 200,000 | 1.216x | 0.756x | 1.920x | 2.910x |
| 500,000 | 1.037x | 1.046x | 2.433x | 2.331x |
| 1,000,000 | 1.256x | 1.159x | 1.766x | 1.810x |

Outlier-population geometric means across all observations:
- WORK: 1.059x Apple speed; 1.854x Apple CPU.
- ATTACK: 1.048x Apple speed; 1.833x Apple CPU.

## Literal all-runner aggregate (12 paired samples per batch)

| n | WORK speed/Apple | ATTACK speed/Apple | WORK CPU/Apple | ATTACK CPU/Apple |
|---:|---:|---:|---:|---:|
| 100 | 0.930x | 1.008x | 1.085x | 0.992x |
| 400 | 0.983x | 1.334x | 1.709x | 1.429x |
| 700 | 1.072x | 1.136x | 1.860x | 1.702x |
| 1,200 | 1.268x | 1.000x | 1.358x | 1.846x |
| 3,000 | 1.164x | 0.988x | 1.795x | 1.939x |
| 5,000 | 1.538x | 1.510x | 1.288x | 1.231x |
| 7,500 | 1.722x | 1.731x | 1.087x | 1.104x |
| 15,000 | 1.132x | 1.763x | 1.802x | 1.072x |
| 29,999 | 1.276x | 1.479x | 1.582x | 1.308x |
| 30,000 | 1.126x | 1.169x | 3.027x | 1.602x |
| 40,000 | 1.089x | 0.940x | 1.819x | 2.743x |
| 40,001 | 1.078x | 1.503x | 1.732x | 1.230x |
| 50,000 | 1.758x | 1.478x | 1.119x | 1.332x |
| 78,000 | 0.880x | 0.995x | 1.814x | 2.098x |
| 80,000 | 1.486x | 1.305x | 1.236x | 1.409x |
| 82,000 | 1.690x | 1.104x | 1.178x | 1.636x |
| 100,000 | 1.388x | 1.367x | 2.083x | 2.093x |
| 200,000 | 1.309x | 0.787x | 1.841x | 2.807x |
| 500,000 | 1.052x | 1.078x | 2.147x | 2.319x |
| 1,000,000 | 1.450x | 1.246x | 1.496x | 1.741x |

All-runner geometric means across all observations:
- WORK: 1.196x Apple speed; 1.668x Apple CPU.
- ATTACK: 1.200x Apple speed; 1.634x Apple CPU.

Because 9/12 paired samples come from degraded/outlier runner states, the all-runner aggregate is not representative of the normal WORK performance envelope.
