# REST 79K–81K hypothesis vs Apple M1

Exact benchmarked SHA: `6008e871ccf5b3ef8edca3a956b6cc3b556f9b8f`
Workflow run: `34027849023`
Frozen REST parent: `925ae7aad2a8878e325f521b1dcc69d5eb443e99`

The experiment branch differs from frozen REST only in the workflow YAML. All four jobs passed exact-REST identity and route assertions. Slot 1 accuracy remained unchanged: 9,600 cases max 2 ULP and 1,000,000 stress cases max 2 ULP.

Each of 79K, 80K and 81K was measured for 5 rounds on four Apple M1 runners, REST and Apple interleaved. Slot 2 was a clear degraded runner: Apple itself ran around 1.9–2.2 ns/el versus about 1.30 ns/el on slots 1,3,4. Therefore slots 1,3,4 are the good population (15 paired samples per size), and slot 2 is segregated as outlier.

## Good population: slots 1,3,4

Paired-ratio medians:

| n | REST wall ns/el | Apple wall ns/el | Apple/REST speed | REST CPU ns/el | Apple CPU ns/el | REST/Apple CPU |
|---:|---:|---:|---:|---:|---:|---:|
| 79,000 | 0.729244 | 1.305707 | 1.764x | 1.872996 | 1.306160 | 1.435x |
| 80,000 | 0.726229 | 1.325816 | 1.798x | 1.433250 | 1.325917 | 1.105x |
| 81,000 | 0.742752 | 1.304509 | 1.752x | 1.871399 | 1.304698 | 1.431x |

The speed and CPU ratios above are medians of paired same-slot/same-round ratios; the ns/el columns are population medians, so ratios need not equal ratios of those displayed medians exactly.

At 80K, only 1 of 15 good-population paired samples had REST/Apple CPU < 1. The good-population geometric-mean CPU ratio at 80K was 1.098x. Therefore REST does not reliably beat Apple on CPU efficiency at 80K in this confirmation run, although 80K is a very clear CPU-efficiency pocket relative to 79K and 81K.

Routes:
- 79K: REST falls through WORK -> frozen baseline -> contract-fast p64.
- 80K: REST falls through WORK -> `wg2_utility_s8` special route.
- 81K: REST falls through WORK -> frozen baseline -> contract-fast p64.

Conclusion: the hypothesis of a broad 79–81K CPU-efficiency region is rejected. The 80K point is a narrow route-specific efficiency pocket, but the prior sub-1.0 CPU ratio was not reproduced robustly.
