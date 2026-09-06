#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"

if [[ "$MODE" == build ]]; then
  test -x /tmp/apple_cos53_canonical_frozen_bench
  test -x /tmp/apple_cos53_canonical_optimized_bench
  bash benchmark_support/run_apple_cos53_apple_specific_eff.sh build
  bash benchmark_support/run_apple_cos53_hotloop_eff.sh build
  bash benchmark_support/run_apple_cos53_current_p64.sh build
  exit 0
fi

if [[ "$MODE" == validate ]]; then
  exec bash benchmark_support/run_apple_cos53_hotloop_eff.sh validate
fi

run_canonical() {
  local n="$1"
  case "$n" in
    100) exec /tmp/apple_cos53_canonical_optimized_bench single "$n" ;;
    400|700|1200|500000|1000000) exec /tmp/apple_cos53_canonical_frozen_bench candidate "$n" ;;
    3000|7500|15000|50000|100000) exec /tmp/apple_cos53_canonical_optimized_bench auto "$n" ;;
    5000|29999|40001) exec /tmp/apple_cos53_canonical_optimized_bench adapt2 "$n" ;;
    30000|40000|200000) exec /tmp/apple_cos53_canonical_optimized_bench pool3 "$n" ;;
    *) echo "unsupported batch $n" >&2; exit 3 ;;
  esac
}

if [[ "$MODE" == route ]]; then
  [[ $# -eq 2 ]]
  n="$2"
  if (( n >= 78000 && n <= 82000 )); then
    echo p64
  else
    echo canonical
  fi
  exit 0
fi

if [[ "$MODE" == one ]]; then
  [[ $# -eq 2 ]]
  n="$2"

  # Current baseline delta: the paired M1 boundary sweep established a narrow
  # p64 production zone.  Do this check before all historical point routes so
  # the 78K-82K interval is unambiguous.
  if (( n >= 78000 && n <= 82000 )); then
    exec bash benchmark_support/run_apple_cos53_current_p64.sh one "$n"
  fi

  # Only routes that reproduced as BOTH faster and lower-CPU than canonical
  # in pooled final 12-sample verification are allowed to replace canonical.
  case "$n" in
    100)   exec bash benchmark_support/run_apple_cos53_hotloop_eff.sh apple_one single_utility "$n" ;;
    400)   exec bash benchmark_support/run_apple_cos53_hotloop_eff.sh apple_one wg2_utility "$n" ;;
    700)   exec bash benchmark_support/run_apple_cos53_hotloop_eff.sh apple_one wg2_ui "$n" ;;
    1200)  exec bash benchmark_support/run_apple_cos53_hotloop_eff.sh apple_one wg2_utility "$n" ;;
    5000)  exec bash benchmark_support/run_apple_cos53_hotloop_eff.sh one pool2 "$n" ;;
    7500)  exec bash benchmark_support/run_apple_cos53_hotloop_eff.sh one pool2 "$n" ;;
    15000) exec bash benchmark_support/run_apple_cos53_hotloop_eff.sh one pool2 "$n" ;;
    40000) exec bash benchmark_support/run_apple_cos53_hotloop_eff.sh apple_one wg3_user "$n" ;;
    50000) exec bash benchmark_support/run_apple_cos53_hotloop_eff.sh apple_one wg2_ui "$n" ;;
    *) run_canonical "$n" ;;
  esac
fi

echo "usage: $0 build | validate | route N | one N" >&2
exit 2
