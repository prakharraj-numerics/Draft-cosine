#!/usr/bin/env bash
set -euo pipefail

# PLUTO: frozen single-thread Apple COS53 baseline.
# Selected from the integrated KERNEL ATTACK shootout.
# Mathematical change vs DESTINY hot kernel: omit only the PI_P3 correction
#     rl = fma(rl, qd, -PI_P3)
# while retaining all other DESTINY K1280 hot-kernel machinery.
# Benchmark contract: PLUTO and all successors are evaluated single-threaded.

[[ "$(uname -m)" == arm64 ]]
[[ "$(sysctl -n machdep.cpu.brand_string)" == *"Apple M1"* ]]

MODE="${1:-}"

if [[ "$MODE" == build ]]; then
  bash benchmark_support/run_apple_cos53_KERNEL_ATTACK.sh build
  test -x /tmp/kernel_attack_no_p3
  test -x /tmp/apple_cos53_hotloop_bench
  test -x /tmp/apple_cos53_off_frozen
  exit 0
fi

if [[ "$MODE" == validate ]]; then
  exec /tmp/kernel_attack_no_p3 validate
fi

if [[ "$MODE" == one ]]; then
  [[ $# -eq 2 ]]
  exec /tmp/kernel_attack_no_p3 single "$2"
fi

if [[ "$MODE" == destiny ]]; then
  [[ $# -eq 2 ]]
  exec /tmp/apple_cos53_hotloop_bench single "$2"
fi

if [[ "$MODE" == apple ]]; then
  [[ $# -eq 2 ]]
  exec env VECLIB_MAXIMUM_THREADS=1 /tmp/apple_cos53_off_frozen apple "$2"
fi

echo "usage: $0 build | validate | one N | destiny N | apple N" >&2
exit 2
