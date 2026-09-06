#!/usr/bin/env bash
set -euo pipefail

# Experimental FIXED copy. Current REST remains frozen at:
# 925ae7aad2a8878e325f521b1dcc69d5eb443e99
#
# Transfer strategy from the REST waste audit:
#   * preserve REST routes that are already locally efficient;
#   * replace only demonstrated routing/kernel holes with the proven
#     conversion-free K1280 hot kernel + Apple wg3_default;
#   * never edit tracked REST implementation files.

MODE="${1:-}"

if [[ "$MODE" == build ]]; then
  [[ "$(uname -m)" == arm64 ]]
  [[ "$(sysctl -n machdep.cpu.brand_string)" == *"Apple M1"* ]]

  # Builds exact REST off/fast families and the conversion-free hot-loop/workgroup binaries.
  bash benchmark_support/run_apple_cos53_contract_fast_baseline.sh build
  # Required by exact REST's special-pocket routes.
  bash benchmark_support/run_apple_cos53_cpu_pocket_attack.sh build

  test -x /tmp/apple_cos53_fast_hotapple
  test -x /tmp/apple_cos53_fast_validate
  cp /tmp/apple_cos53_fast_hotapple /tmp/apple_cos53_REST_fixed_transfer_hotapple
  cp /tmp/apple_cos53_fast_validate /tmp/apple_cos53_REST_fixed_transfer_validate
  exit 0
fi

if [[ "$MODE" == validate ]]; then
  exec /tmp/apple_cos53_REST_fixed_transfer_validate validate
fi

route_for() {
  case "$1" in
    # Strong dual-win holes from the prior 4-slot x 3-round audit.
    3000|79000|81000|100000|500000) echo hot_wg3_default ;;
    # All other requested checkpoints retain exact REST because its specialized
    # route was already as CPU-efficient or better.
    *) echo REST ;;
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
    hot_wg3_default)
      exec /tmp/apple_cos53_REST_fixed_transfer_hotapple wg3_default "$n" ;;
    REST)
      exec bash benchmark_support/run_apple_cos53_WORK_smooth_switch.sh one "$n" ;;
  esac
fi

echo "usage: $0 build | validate | route N | one N" >&2
exit 2
