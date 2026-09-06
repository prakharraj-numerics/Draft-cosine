#!/usr/bin/env bash
set -euo pipefail

# Experimental CPU-efficiency repair built strictly on frozen PATCH:
#   5f13c87218873fd9353ef5ab549b8170d8bf3e2a
# Frozen PATCH itself is never modified.
#
# Final selection rule after the full 4-slot x 3-round ladder verification:
# replace PATCH only where the candidate remained BOTH faster and lower-CPU
# than PATCH in the verification run.  All other sizes delegate to PATCH.

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
    wg3_default_s0)
      exec /tmp/apple_cos53_cpu_pocket_attack wg3_default_s0 "$n" ;;
    PATCH)
      exec bash benchmark_support/run_apple_cos53_REST_fixed_transfer_all.sh one "$n" ;;
  esac
fi

echo "usage: $0 build | validate | route N | one N" >&2
exit 2
