#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"

if [[ "$MODE" == build ]]; then
  bash benchmark_support/run_apple_cos53_contract_fast_baseline.sh build
  bash benchmark_support/run_apple_cos53_cpu_pocket_attack.sh build
  clang++ -O3 -DNDEBUG -std=c++20 -mcpu=native -fno-fast-math -ffp-contract=off \
    -I/tmp -I/tmp/pthreadpool-install/include /tmp/apple_cos53_cpu_pocket_attack.cpp \
    /tmp/pthreadpool-install/lib/libpthreadpool.a -framework Accelerate -pthread \
    -o /tmp/apple_cos53_cpu_pocket_attack_off
  exit 0
fi

if [[ "$MODE" == route ]]; then
  [[ $# -eq 2 ]]
  case "$2" in
    78000) echo old_wg3_utility ;;
    80000) echo wg2_utility_s8 ;;
    82000) echo frozen_baseline ;;
    100000) echo native_pool3_off ;;
    200000) echo old_wg3_utility ;;
    500000) echo frozen_baseline ;;
    1000000) echo wg3_default_s8 ;;
    *) echo frozen_baseline ;;
  esac
  exit 0
fi

if [[ "$MODE" == one ]]; then
  [[ $# -eq 2 ]]
  n="$2"
  case "$n" in
    78000)
      bash benchmark_support/run_apple_cos53_contract_fast_baseline.sh activate fast
      exec bash benchmark_support/run_apple_cos53_hotloop_eff.sh apple_one wg3_utility "$n" ;;
    80000)
      exec /tmp/apple_cos53_cpu_pocket_attack wg2_utility_s8 "$n" ;;
    82000)
      exec bash benchmark_support/run_apple_cos53_contract_fast_baseline.sh one "$n" ;;
    100000)
      bash benchmark_support/run_apple_cos53_contract_fast_baseline.sh activate off
      exec bash benchmark_support/run_apple_cos53_hotloop_eff.sh one pool3 "$n" ;;
    200000)
      bash benchmark_support/run_apple_cos53_contract_fast_baseline.sh activate fast
      exec bash benchmark_support/run_apple_cos53_hotloop_eff.sh apple_one wg3_utility "$n" ;;
    500000)
      exec bash benchmark_support/run_apple_cos53_contract_fast_baseline.sh one "$n" ;;
    1000000)
      exec /tmp/apple_cos53_cpu_pocket_attack wg3_default_s8 "$n" ;;
    *)
      exec bash benchmark_support/run_apple_cos53_contract_fast_baseline.sh one "$n" ;;
  esac
fi

echo "usage: $0 build | route N | one N" >&2
exit 2
