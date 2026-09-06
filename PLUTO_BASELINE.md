# PLUTO baseline

PLUTO is the frozen single-thread Apple COS53 baseline selected from the integrated kernel-attack shootout on 2026-09-06.

- Parent baseline: DESTINY (`f3288a1faf54d7b352bd564ae8760ddb35ce1892`).
- Kernel delta: remove only the `PI_P3` low-part correction from the DESTINY K1280 hot kernel.
- Benchmark contract: PLUTO and successors are evaluated single-threaded; Apple `vvcos` comparator is constrained with `VECLIB_MAXIMUM_THREADS=1`.
- Selection evidence at n=1,000,000: approximately 3.18% faster and 3.03% lower process CPU than DESTINY single-thread, while retaining max 2 ULP in the accuracy gate.
- Frozen DESTINY remains unchanged.

Canonical runner: `benchmark_support/run_apple_cos53_PLUTO.sh`.
