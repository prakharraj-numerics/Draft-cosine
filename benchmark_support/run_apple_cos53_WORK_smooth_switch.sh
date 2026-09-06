#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-}"
if [[ "$MODE" == build ]]; then
  bash benchmark_support/run_apple_cos53_cpu_pockets_selected.sh build
  exit 0
fi
if [[ "$MODE" == route ]]; then
  n="$2"
  if (( n >= 5000 && n < 30000 )); then echo pool2;
  elif (( n >= 30000 && n < 40000 )); then echo wg2_ui;
  elif (( n == 40000 )); then echo WORK_40K_wg3_user;
  elif (( n > 40000 && n < 78000 )); then echo wg2_ui;
  else echo WORK;
  fi
  exit 0
fi
if [[ "$MODE" == one ]]; then
  n="$2"
  if (( n >= 5000 && n < 30000 )); then
    bash benchmark_support/run_apple_cos53_contract_fast_baseline.sh activate fast >/dev/null
    exec bash benchmark_support/run_apple_cos53_hotloop_eff.sh one pool2 "$n"
  fi
  if (( n >= 30000 && n < 40000 )); then
    bash benchmark_support/run_apple_cos53_contract_fast_baseline.sh activate fast >/dev/null
    exec bash benchmark_support/run_apple_cos53_hotloop_eff.sh apple_one wg2_ui "$n"
  fi
  if (( n == 40000 )); then
    exec bash benchmark_support/run_apple_cos53_cpu_pockets_selected.sh one "$n"
  fi
  if (( n > 40000 && n < 78000 )); then
    bash benchmark_support/run_apple_cos53_contract_fast_baseline.sh activate fast >/dev/null
    exec bash benchmark_support/run_apple_cos53_hotloop_eff.sh apple_one wg2_ui "$n"
  fi
  exec bash benchmark_support/run_apple_cos53_cpu_pockets_selected.sh one "$n"
fi
echo "usage: $0 build | route N | one N" >&2
exit 2
