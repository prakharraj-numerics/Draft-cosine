# Apple COS53 Apple-specific efficiency results — 2026-09-06

Workflow run: `33996683203`
Branch: `exp/apple-cos53-apple-specific-eff-20260906`
Benchmark commit: `60fc3dc27d4d7447f693b97a19f7fe3ecc14ebc3`
Baseline: `freeze/apple-cos53-hybrid-speed-cpu-vvcos-20260906`
Hardware: 4 independent GitHub macOS-15 arm64 Apple M1 runners; 3 rounds each = 12 samples per mode/batch; medians below.

Acceptance rule: candidate wall time <= 1.01 × frozen-hybrid median wall time AND candidate process CPU ns/el < frozen-hybrid CPU ns/el. Among eligible candidates, choose the minimum CPU ns/el. Otherwise retain the frozen hybrid.

Accuracy revalidation: 9,600 MPFR256 cases max 2 ULP, 0 >2 ULP; 1,000,000 stressed MPFR256 cases max 2 ULP, 0 >2 ULP.

Tested Apple-specific variants: persistent `os_workgroup_parallel` runners with 2 or 3 total cooperating threads, C++20 atomic wait/notify event parking with a 32-yield completion spin, and QoS sweep across user-interactive, user-initiated, default, and utility. Lower-QoS single-thread controls were also tested.

| n | selected | hybrid wall | selected wall | wall vs hybrid | hybrid CPU | selected CPU | CPU reduction | vvcos wall | vvcos CPU | selected CPU / vvcos |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | single_utility | 1.7984 | 1.7230 | -4.2% | 1.7973 | 1.7214 | 4.2% | 1.3937 | 1.3914 | 1.237× |
| 400 | wg2_ui | 1.1802 | 1.1842 | +0.3% | 2.3302 | 2.2793 | 2.2% | 1.2972 | 1.2963 | 1.758× |
| 700 | wg2_utility | 1.1035 | 1.0478 | -5.0% | 2.1106 | 1.9766 | 6.3% | 1.2473 | 1.2464 | 1.586× |
| 1,200 | wg2_ui | 1.0354 | 1.0019 | -3.2% | 2.0263 | 1.8921 | 6.6% | 1.2526 | 1.2526 | 1.510× |
| 3,000 | wg2_default | 0.9944 | 0.9626 | -3.2% | 1.9331 | 1.8134 | 6.2% | 1.2425 | 1.2419 | 1.460× |
| 5,000 | wg3_utility | 0.9793 | 0.6706 | -31.5% | 1.8071 | 1.6670 | 7.8% | 1.2349 | 1.2348 | 1.350× |
| 7,500 | hybrid | 0.9318 | 0.9318 | +0.0% | 1.6128 | 1.6128 | 0.0% | 1.2702 | 1.2701 | 1.270× |
| 15,000 | wg3_user | 0.9271 | 0.6265 | -32.4% | 1.8350 | 1.6603 | 9.5% | 1.3023 | 1.3009 | 1.276× |
| 29,999 | wg3_utility | 0.9071 | 0.6283 | -30.7% | 1.7829 | 1.6981 | 4.8% | 1.2395 | 1.2396 | 1.370× |
| 30,000 | hybrid | 0.6240 | 0.6240 | +0.0% | 1.7849 | 1.7849 | 0.0% | 1.2269 | 1.2266 | 1.455× |
| 40,000 | wg3_user | 0.6178 | 0.6166 | -0.2% | 1.8683 | 1.7925 | 4.1% | 1.2316 | 1.2314 | 1.456× |
| 40,001 | wg2_default | 0.9029 | 0.8979 | -0.6% | 1.8045 | 1.7286 | 4.2% | 1.2220 | 1.2222 | 1.414× |
| 50,000 | wg2_user | 0.8960 | 0.8855 | -1.2% | 1.7725 | 1.6954 | 4.4% | 1.2218 | 1.2199 | 1.390× |
| 100,000 | hybrid | 0.6067 | 0.6067 | +0.0% | 1.6435 | 1.6435 | 0.0% | 1.2530 | 1.2532 | 1.312× |
| 200,000 | hybrid | 0.6083 | 0.6083 | +0.0% | 1.8387 | 1.8387 | 0.0% | 1.2569 | 1.2519 | 1.469× |
| 500,000 | wg3_default | 0.7006 | 0.6504 | -7.2% | 1.9363 | 1.8468 | 4.6% | 1.2543 | 1.2489 | 1.479× |
| 1,000,000 | wg3_ui | 0.6674 | 0.6529 | -2.2% | 1.9035 | 1.8970 | 0.3% | 1.3283 | 1.3265 | 1.430× |

Accepted Apple-specific improvement at 13 of 17 batch sizes.

Strongest additional CPU reductions vs frozen hybrid: 15K 9.5%, 5K 7.8%, 1.2K 6.6%, 700 6.3%, 3K 6.2%, 29,999 4.8%, 500K 4.6%, 50K 4.4%, 100 4.2%, 40,001 4.2%, 40K 4.1%, 400 2.2%, 1M 0.3%.

No accepted Apple-specific replacement at 7.5K, 30K, 100K, or 200K; the frozen hybrid remains the winner there under the strict speed-and-CPU rule.

The workgroup/QoS layer still does not match `vvcos` process-CPU efficiency. After selection, CPU ratios vs `vvcos` remain roughly 1.24×–1.59× on most points, though wall time remains much faster than `vvcos` across the meaningful medium/large ranges.
