# Exact WORK full-ladder verification vs Apple M1

Frozen WORK SHA: `2daf7bd0a5a08df06efd169b1c38b55569de32f7`.
Verification workflow commit: `98447234bd7f324d8033705f197c2508d22bebcb`.
Workflow run: `34025563931`.

The verification branch was created from exact WORK. Before build, the workflow runs `git diff --exit-code WORK HEAD -- .` excluding only the workflow YAML. This passed on all four jobs. Frozen route assertions also passed on all four jobs. Slot 1 accuracy passed unchanged: 9600 cases max 2 ULP and 1,000,000 stress cases max 2 ULP.

Fresh run: 4 Apple M1 jobs x 3 rounds = 12 paired samples per batch. Two runners (slots 3 and 4, and parts of slot 2) entered a broad scheduler-degraded state for the multithreaded WORK routes while Apple vvcos remained stable. Therefore the 12-sample aggregate is included exactly as measured, but it is not representative of the normal WORK performance envelope. Slot 1 remained in the normal scheduler state and independently reproduced the prior WORK shape.

| n | 12-sample Apple/WORK wall | 12-sample WORK/Apple CPU | slot-1 Apple/WORK wall | slot-1 WORK/Apple CPU |
|---:|---:|---:|---:|---:|
| 100 | 0.996x | 1.005x | 0.995x | 1.006x |
| 400 | 0.948x | 2.062x | 1.334x | 1.458x |
| 700 | 1.033x | 1.859x | 1.484x | 1.248x |
| 1,200 | 0.978x | 2.011x | 1.607x | 1.138x |
| 3,000 | 0.916x | 2.072x | 1.304x | 1.484x |
| 5,000 | 1.080x | 1.851x | 1.624x | 1.208x |
| 7,500 | 1.188x | 1.655x | 1.736x | 1.117x |
| 15,000 | 1.098x | 1.856x | 1.791x | 1.109x |
| 29,999 | 0.727x | 2.449x | 1.444x | 1.346x |
| 30,000 | 1.381x | 2.148x | 2.061x | 1.464x |
| 40,000 | 1.198x | 2.549x | 2.626x | 1.005x |
| 40,001 | 1.197x | 1.669x | 1.469x | 1.380x |
| 50,000 | 1.207x | 1.563x | 1.842x | 1.110x |
| 78,000 | 1.265x | 2.660x | 2.631x | 1.137x |
| 80,000 | 1.495x | 1.318x | 1.799x | 1.109x |
| 82,000 | 1.589x | 1.159x | 1.846x | 1.072x |
| 100,000 | 2.241x | 1.423x | 2.221x | 1.174x |
| 200,000 | 0.961x | 2.407x | 2.091x | 1.407x |
| 500,000 | 1.031x | 2.538x | 1.768x | 1.566x |
| 1,000,000 | 1.139x | 1.974x | 2.545x | 1.164x |

Geometric mean, all 20 points: 12-sample aggregate speed 1.150x Apple and CPU 1.851x Apple; slot-1 normal-state speed 1.759x Apple and CPU 1.224x Apple.

No WORK source or routing was changed by this verification. The rejected smooth-switch experiment is not in WORK.
