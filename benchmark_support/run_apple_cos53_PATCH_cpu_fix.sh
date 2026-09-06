#!/usr/bin/env bash
set -euo pipefail

# Experimental CPU-efficiency repair built strictly on frozen PATCH:
#   5f13c87218873fd9353ef5ab549b8170d8bf3e2a
# Frozen PATCH itself is never modified.
#
# Selection rule: replace a PATCH route only when the 4-slot x 3-round paired
# M1 sweeps showed the candidate both faster AND lower-process-CPU than PATCH,
# with a material CPU improvement.  Otherwise delegate to exact PATCH.

MODE="${1:-}"

if [[ "$MODE" == build ]]; then
  [[ "$(uname -m)" == arm64 ]]
  [[ "$(sysctl -n machdep.cpu.brand_string)" == *"Apple M1"* ]]
  bash benchmark_support/run_apple_cos53_REST_fixed_transfer_all.sh build
  bash benchmark_support/run_apple_cos53_cpu_pocket_attack.sh build
  test -x /tmp/apple_cos53_fast_hotapple
  test -x /tmp/apple_cos53_cpu_pocket_attack
  exit 0
fi

if [[ "$MODE" == validate ]]; then
  exec bash benchmark_support/run_apple_cos53_REST_fixed_transfer_all.sh validate
fi

route_for() {
  case "$1" in
    1200)   echo wg2_ui ;;
    30000)  echo wg2_default ;;
    79000)  echo wg3_ui_s8 ;;
    100000) echo wg3_utility_s0 ;;
    200000) echo wg3_user_s0 ;;
    500000) echo wg3_default_s0 ;;
    *)      echo PATCH ;;
  esac
}

if [[ "$MODE" == route ]]; then
  [[ $# -eq 2 ]]
  route_for "$2"
  exit 0
fi

if [[ "$MODE" == one ]]; then
  [[ $# -eq 2 ]]
  n="$2"
  case "$(route_for "$n")" in
    wg2_ui)
      exec bash benchmark_support/run_apple_cos53_hotloop_eff.sh apple_one wg2_ui "$n" ;;
    wg2_default)
      exec bash benchmark_support/run_apple_cos53_hotloop_eff.sh apple_one wg2_default "$n" ;;
    wg3_ui_s8)
      exec /tmp/apple_cos53_cpu_pocket_attack wg3_ui_s8 "$n" ;;
    wg3_utility_s0)
      exec /tmp/apple_cos53_cpu_pocket_attack wg3_utility_s0 "$n" ;;
    wg3_user_s0)
      exec /tmp/apple_cos53_cpu_pocket_attack wg3_user_s0 "$n" ;;
    wg3_default_s0)
      exec /tmp/apple_cos53_cpu_pocket_attack wg3_default_s0 "$n" ;;
    PATCH)
      exec bash benchmark_support/run_apple_cos53_REST_fixed_transfer_all.sh one "$n" ;;
  esac
fi

echo "usage: $0 build | validate | route N | one N" >&2
exit 2
