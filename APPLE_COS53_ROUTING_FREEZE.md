# Apple COS53 routing freeze

This commit is the complete frozen Apple COS53 routing state.

- `< 30,000`: existing frozen AppleTwoCoreHighway
- `30,000 .. < 50,000`: p32
- `50,000 .. < 100,000`: p16
- `100,000 .. < 1,000,000`: p12
- `>= 1,000,000`: p24

Production routing implementation: `apple_cos53_production_routing.hpp`.

Frozen kernel/orchestration assumptions:
- Highway Apple COS53 K=2048, degree=3, terms=1
- stabilized `dispatch_apply_f` + `DISPATCH_APPLY_AUTO` above 30K
- balanced fixed partitions
- 1M route confirmed separately with 4 M1 slots x 10 repetitions

No Intel/Xeon production files are changed by this Apple routing freeze.
