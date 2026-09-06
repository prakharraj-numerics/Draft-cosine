#!/usr/bin/env bash
set -euo pipefail

[[ "$(uname -m)" == arm64 ]]
[[ "$(sysctl -n machdep.cpu.brand_string)" == *"Apple M1"* ]]
MODE="${1:-}"

if [[ "$MODE" == build ]]; then
  bash benchmark_support/run_apple_cos53_PATCH_cpu_fix.sh build
  test -f /tmp/apple_cos53_hotloop.cpp
  test -x /tmp/apple_cos53_hotloop_bench
  test -x /tmp/apple_cos53_off_frozen
  python3 - <<'PY'
from pathlib import Path
p=Path('/tmp/apple_cos53_hotloop.cpp')
s=p.read_text()
old='    rl=vfmaq_n_f64(rl,qd,-PI_P3);\n'
assert old in s
s=s.replace(old,'',1)
Path('/tmp/apple_cos53_no_p3.cpp').write_text(s)
PY
  MP="$(brew --prefix mpfr)"; GP="$(brew --prefix gmp)"
  clang++ -O3 -DNDEBUG -std=c++20 -mcpu=native -fno-fast-math -ffp-contract=fast \
    -DOPT_VALIDATE_MPFR -I/tmp -I/tmp/pthreadpool-install/include -I"$MP/include" -I"$GP/include" \
    /tmp/apple_cos53_no_p3.cpp /tmp/pthreadpool-install/lib/libpthreadpool.a \
    -framework Accelerate -pthread -L"$MP/lib" -L"$GP/lib" -lmpfr -lgmp \
    -o /tmp/apple_cos53_no_p3
  exit 0
fi

if [[ "$MODE" == validate ]]; then exec /tmp/apple_cos53_no_p3 validate; fi
if [[ "$MODE" == candidate ]]; then [[ $# -eq 2 ]]; exec /tmp/apple_cos53_no_p3 single "$2"; fi
if [[ "$MODE" == destiny ]]; then [[ $# -eq 2 ]]; exec /tmp/apple_cos53_hotloop_bench single "$2"; fi
if [[ "$MODE" == apple ]]; then [[ $# -eq 2 ]]; exec env VECLIB_MAXIMUM_THREADS=1 /tmp/apple_cos53_off_frozen apple "$2"; fi

echo "usage: $0 build | validate | candidate N | destiny N | apple N" >&2
exit 2
