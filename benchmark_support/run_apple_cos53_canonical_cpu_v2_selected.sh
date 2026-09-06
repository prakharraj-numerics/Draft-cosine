#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"

if [[ "$MODE" == build ]]; then
  bash benchmark_support/run_apple_cos53_cpu_eff_all_attacks.sh build
  bash benchmark_support/run_apple_cos53_apple_specific_eff.sh build
  bash benchmark_support/run_apple_cos53_hotloop_eff.sh build
  exit 0
fi

if [[ "$MODE" == validate ]]; then
  exec bash benchmark_support/run_apple_cos53_hotloop_eff.sh validate
fi

if [[ "$MODE" == one ]]; then
  [[ $# -eq 2 ]]
  n="$2"
  case "$n" in
    100)     exec bash benchmark_support/run_apple_cos53_hotloop_eff.sh apple_one single_utility "$n" ;;
    400)     exec bash benchmark_support/run_apple_cos53_hotloop_eff.sh apple_one wg2_utility "$n" ;;
    700)     exec bash benchmark_support/run_apple_cos53_hotloop_eff.sh apple_one wg2_ui "$n" ;;
    1200)    exec bash benchmark_support/run_apple_cos53_hotloop_eff.sh apple_one wg2_utility "$n" ;;
    3000)    exec bash benchmark_support/run_apple_cos53_hotloop_eff.sh apple_one wg3_ui "$n" ;;
    5000)    exec bash benchmark_support/run_apple_cos53_hotloop_eff.sh one pool2 "$n" ;;
    7500)    exec bash benchmark_support/run_apple_cos53_hotloop_eff.sh one pool2 "$n" ;;
    15000)   exec bash benchmark_support/run_apple_cos53_hotloop_eff.sh one pool2 "$n" ;;
    29999)   exec bash benchmark_support/run_apple_cos53_hotloop_eff.sh one auto "$n" ;;
    30000)   exec bash benchmark_support/run_apple_cos53_hotloop_eff.sh apple_one wg3_default "$n" ;;
    40000)   exec bash benchmark_support/run_apple_cos53_hotloop_eff.sh apple_one wg3_user "$n" ;;
    40001)   exec bash benchmark_support/run_apple_cos53_hotloop_eff.sh apple_one wg3_ui "$n" ;;
    50000)   exec bash benchmark_support/run_apple_cos53_hotloop_eff.sh apple_one wg2_ui "$n" ;;
    100000)  exec bash benchmark_support/run_apple_cos53_hotloop_eff.sh apple_one wg3_utility "$n" ;;
    200000)  exec bash benchmark_support/run_apple_cos53_hotloop_eff.sh apple_one wg3_default "$n" ;;
    500000)  exec bash benchmark_support/run_apple_cos53_hotloop_eff.sh apple_one wg3_ui "$n" ;;
    1000000) exec bash benchmark_support/run_apple_cos53_hotloop_eff.sh apple_one wg3_user "$n" ;;
    *) echo "unsupported selected batch $n" >&2; exit 3 ;;
  esac
fi

echo "usage: $0 build | validate | one N" >&2
exit 2
