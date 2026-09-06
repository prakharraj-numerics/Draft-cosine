#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
FILES=(
  benchmark_support/run_apple_cos53_k1280_fastreduce_vs_vvcos.sh
  benchmark_support/run_apple_cos53_cpu_eff_all_attacks.sh
  benchmark_support/run_apple_cos53_apple_specific_eff.sh
  benchmark_support/run_apple_cos53_hotloop_eff.sh
  benchmark_support/run_apple_cos53_current_p64.sh
)

prepare_k1280() {
  python3 - <<'PY'
from pathlib import Path
p=Path('benchmark_support/run_apple_cos53_k1280_fastreduce_vs_vvcos.sh')
s=p.read_text()
a=s.index("a=s.index('static int validate_mode()')")
b=s.index("print('K1280_RETUNE'", a)
s=s[:a]+"Path('/tmp/candidate_stress.cpp').write_text(s)\n"+s[b:]
s=s.replace('  echo "=== stressed 1M MPFR256 diagnostic ==="\n  /tmp/apple_cos53_k1280_fastreduce_stress validate\n','')
p.write_text(s)
PY
}

build_current_stack() {
  prepare_k1280
  bash benchmark_support/run_apple_cos53_k1280_fastreduce_vs_vvcos.sh build
  cp /tmp/apple_cos53_k1280_fastreduce_bench /tmp/apple_cos53_canonical_frozen_bench
  bash benchmark_support/run_apple_cos53_cpu_eff_all_attacks.sh build
  cp /tmp/apple_cos53_cpu_eff_bench /tmp/apple_cos53_canonical_optimized_bench
  bash benchmark_support/run_apple_cos53_canonical_cpu_v2_robust.sh build
}

save_stack() {
  local tag="$1"
  cp /tmp/apple_cos53_canonical_frozen_bench "/tmp/apple_cos53_${tag}_frozen"
  cp /tmp/apple_cos53_canonical_optimized_bench "/tmp/apple_cos53_${tag}_optimized"
  cp /tmp/apple_cos53_hotloop_bench "/tmp/apple_cos53_${tag}_hotloop"
  cp /tmp/apple_cos53_hotloop_apple_specific_bench "/tmp/apple_cos53_${tag}_hotapple"
  cp /tmp/apple_cos53_current_p64_bench "/tmp/apple_cos53_${tag}_p64"
  cp /tmp/apple_cos53_hotloop_validate "/tmp/apple_cos53_${tag}_validate"
}

activate_stack() {
  local tag="$1"
  cp "/tmp/apple_cos53_${tag}_frozen" /tmp/apple_cos53_canonical_frozen_bench
  cp "/tmp/apple_cos53_${tag}_optimized" /tmp/apple_cos53_canonical_optimized_bench
  cp "/tmp/apple_cos53_${tag}_hotloop" /tmp/apple_cos53_hotloop_bench
  cp "/tmp/apple_cos53_${tag}_hotapple" /tmp/apple_cos53_hotloop_apple_specific_bench
  cp "/tmp/apple_cos53_${tag}_p64" /tmp/apple_cos53_current_p64_bench
  cp "/tmp/apple_cos53_${tag}_validate" /tmp/apple_cos53_hotloop_validate
}

if [[ "$MODE" == build ]]; then
  [[ "$(uname -m)" == arm64 ]]
  [[ "$(sysctl -n machdep.cpu.brand_string)" == *"Apple M1"* ]]
  rm -rf /tmp/contract_fast_script_orig
  mkdir -p /tmp/contract_fast_script_orig
  for f in "${FILES[@]}"; do cp "$f" "/tmp/contract_fast_script_orig/$(basename "$f")"; done

  restore_scripts() {
    for f in "${FILES[@]}"; do cp "/tmp/contract_fast_script_orig/$(basename "$f")" "$f"; done
  }
  trap restore_scripts EXIT

  # Exact prior p64 baseline (-ffp-contract=off).
  build_current_stack
  save_stack off

  # Same source/routing, only permit compiler FP contraction.
  restore_scripts
  python3 - <<'PY'
from pathlib import Path
paths=[
 'benchmark_support/run_apple_cos53_k1280_fastreduce_vs_vvcos.sh',
 'benchmark_support/run_apple_cos53_cpu_eff_all_attacks.sh',
 'benchmark_support/run_apple_cos53_apple_specific_eff.sh',
 'benchmark_support/run_apple_cos53_hotloop_eff.sh',
 'benchmark_support/run_apple_cos53_current_p64.sh',
]
for q in paths:
    p=Path(q); s=p.read_text()
    if '-ffp-contract=off' not in s:
        raise SystemExit(f'missing contract flag in {q}')
    p.write_text(s.replace('-ffp-contract=off','-ffp-contract=fast'))
PY
  build_current_stack
  save_stack fast
  activate_stack fast
  restore_scripts
  trap - EXIT
  exit 0
fi

if [[ "$MODE" == activate ]]; then
  [[ $# -eq 2 ]]
  activate_stack "$2"
  exit 0
fi

if [[ "$MODE" == validate ]]; then
  activate_stack fast
  exec bash benchmark_support/run_apple_cos53_canonical_cpu_v2_robust.sh validate
fi

if [[ "$MODE" == route ]]; then
  [[ $# -eq 2 ]]
  n="$2"
  if (( n == 100000 || n == 500000 )); then
    echo prior_baseline
  elif (( n >= 78000 && n <= 82000 )); then
    echo contract_fast_p64
  else
    echo contract_fast
  fi
  exit 0
fi

if [[ "$MODE" == one ]]; then
  [[ $# -eq 2 ]]
  n="$2"
  if (( n == 100000 )); then
    exec /tmp/apple_cos53_off_optimized auto "$n"
  fi
  if (( n == 500000 )); then
    exec /tmp/apple_cos53_off_frozen candidate "$n"
  fi
  activate_stack fast
  exec bash benchmark_support/run_apple_cos53_canonical_cpu_v2_robust.sh one "$n"
fi

echo "usage: $0 build | activate {off|fast} | validate | route N | one N" >&2
exit 2
