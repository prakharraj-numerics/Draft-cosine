#!/usr/bin/env bash
set -euo pipefail

# Experimental copy only.  The frozen/current REST implementation is never edited.
# Base REST SHA: 925ae7aad2a8878e325f521b1dcc69d5eb443e99
# Fixed copy = conversion-free K1280 hot kernel + existing Apple WG3/default QoS.

MODE="${1:-}"

if [[ "$MODE" == "build" ]]; then
  [[ "$(uname -m)" == arm64 ]]
  [[ "$(sysctl -n machdep.cpu.brand_string)" == *"Apple M1"* ]]

  # Build REST's exact off/fast families.  This generates the proven hot-loop
  # derivative without changing any tracked REST source.
  bash benchmark_support/run_apple_cos53_contract_fast_baseline.sh build

  # Freeze independent copies so later REST route activation cannot replace them.
  test -x /tmp/apple_cos53_fast_hotapple
  test -x /tmp/apple_cos53_fast_validate
  cp /tmp/apple_cos53_fast_hotapple /tmp/apple_cos53_REST_wg3_default_fixed_copy
  cp /tmp/apple_cos53_fast_validate /tmp/apple_cos53_REST_wg3_default_fixed_validate
  exit 0
fi

if [[ "$MODE" == "validate" ]]; then
  exec /tmp/apple_cos53_REST_wg3_default_fixed_validate validate
fi

if [[ "$MODE" == "route" ]]; then
  [[ $# -eq 2 ]]
  echo "hot_k1280_conversionfree_wg3_default"
  exit 0
fi

if [[ "$MODE" == "one" ]]; then
  [[ $# -eq 2 ]]
  exec /tmp/apple_cos53_REST_wg3_default_fixed_copy wg3_default "$2"
fi

echo "usage: $0 build | validate | route N | one N" >&2
exit 2
