#!/usr/bin/env bash
set -euo pipefail

# PLUTO-CLEAN: successor experiment branched from frozen PLUTO.
# Math must remain bit-for-bit source-equivalent to PLUTO no_p3 kernel.
# This build removes all multicore-only source and link baggage from the
# candidate executable: pthreadpool, helper threads, QoS, workgroups, pools,
# adaptive routing, auto routing, and Apple vvcos code.

[[ "$(uname -m)" == arm64 ]]
[[ "$(sysctl -n machdep.cpu.brand_string)" == *"Apple M1"* ]]
MODE="${1:-}"

if [[ "$MODE" == build ]]; then
  # Use frozen PLUTO only as a reproducible source/LUT generator. Nothing from
  # its threading stack is linked into PLUTO-CLEAN.
  bash benchmark_support/run_apple_cos53_PLUTO.sh build
  test -f /tmp/kernel_attack_no_p3.cpp
  test -f /tmp/apple_cos53_coeff_aos.h
  test -x /tmp/kernel_attack_no_p3
  test -x /tmp/apple_cos53_off_frozen

  python3 - <<'PY'
from pathlib import Path
s=Path('/tmp/kernel_attack_no_p3.cpp').read_text()

# Remove headers used only by multicore or Apple-vForce benchmark plumbing.
for inc in [
    '#include <Accelerate/Accelerate.h>\n',
    '#include <pthreadpool.h>\n',
    '#include <pthread.h>\n',
    '#include <atomic>\n',
    '#include <thread>\n',
]:
    assert inc in s, inc
    s=s.replace(inc,'',1)

# Remove every helper/pool/auto runner. Keep fill_case onward.
a=s.index('class Adaptive2 {')
b=s.index('static void fill_case',a)
s=s[:a]+s[b:]

# Replace generic runner + Apple runner with a direct single-thread benchmark.
a=s.index('template<class R> static int bench_runner')
b=s.index('#ifdef OPT_VALIDATE_MPFR',a)
bench=r'''static int bench_single(size_t n) {
    Buffers b(n); size_t reps=reps_for(n);
    for(int w=0;w<12;w++) for(int c=0;c<6;c++) opt_cos53_eval(b.x[c],b.y[c],n);
    double c0=process_cpu_ns(); uint64_t w0=now_ticks();
    for(size_t rr=0;rr<reps;rr++) for(int c=0;c<6;c++) opt_cos53_eval(b.x[c],b.y[c],n);
    uint64_t w1=now_ticks(); double c1=process_cpu_ns();
    double den=(double)reps*(double)n*6.0;
    double wall=ticks_to_ns(w1-w0)/den, cpu=(c1-c0)/den;
    g_sink+=b.y[0][(n*7/11)%n];
    std::printf("APPLE_COS53_CPU_EFF_RESULT stack=PLUTO_CLEAN n=%zu wall_ns_el=%.9f cpu_ns_el=%.9f effective_cores=%.6f reps=%zu sink=%.17g\n",
                n,wall,cpu,cpu/wall,reps,(double)g_sink);
    return 0;
}

'''
s=s[:a]+bench+s[b:]

# Replace main: no QoS manipulation, no routes, no other runner objects.
a=s.index('int main(int argc,char**argv) {')
main=r'''int main(int argc,char**argv) {
#ifdef OPT_VALIDATE_MPFR
    if(argc==2 && std::string(argv[1])=="validate") return validate();
#endif
    if(argc!=2)return 2;
    size_t n=(size_t)std::strtoull(argv[1],nullptr,10);
    return bench_single(n);
}
'''
s=s[:a]+main

# Hard source-level liability gate.
for bad in ['pthreadpool','pthread_set_qos','QOS_CLASS_','Adaptive2','PoolRunner','AutoRunner',
            'os_workgroup','std::thread','vvcos','APPLE_SINGLE','pool_piece']:
    if bad in s:
        raise SystemExit(f'forbidden residue in PLUTO-CLEAN source: {bad}')

# The PLUTO mathematical change must still be exactly no-P3.
assert 'rl=vfmaq_n_f64(rl,qd,-PI_P3);' not in s
assert 'float64x2_t rl=vfmaq_n_f64(d,qd,-PI_P2);' in s
assert 'const float64x2_t magic=vdupq_n_f64(0x1p52);' in s
assert 'for (; i+8<=n; i+=8)' in s

Path('/tmp/apple_cos53_pluto_clean.cpp').write_text(s)
PY

  MP="$(brew --prefix mpfr)"; GP="$(brew --prefix gmp)"
  COMMON="-O3 -DNDEBUG -std=c++20 -mcpu=native -fno-fast-math -ffp-contract=fast"
  clang++ $COMMON -I/tmp /tmp/apple_cos53_pluto_clean.cpp -o /tmp/apple_cos53_pluto_clean_bench
  clang++ $COMMON -DOPT_VALIDATE_MPFR -I/tmp -I"$MP/include" -I"$GP/include" \
    /tmp/apple_cos53_pluto_clean.cpp -L"$MP/lib" -L"$GP/lib" -lmpfr -lgmp \
    -o /tmp/apple_cos53_pluto_clean_validate

  # Compile assembly for code-layout comparison and prove no direct multicore refs.
  clang++ $COMMON -I/tmp -S /tmp/apple_cos53_pluto_clean.cpp -o /tmp/apple_cos53_pluto_clean.s
  if nm -u /tmp/apple_cos53_pluto_clean_bench | grep -E 'pthreadpool|pthread_set_qos|os_workgroup' ; then
    echo 'unexpected multicore symbol in PLUTO-CLEAN' >&2; exit 1
  fi
  exit 0
fi

if [[ "$MODE" == validate ]]; then exec /tmp/apple_cos53_pluto_clean_validate validate; fi
if [[ "$MODE" == one ]]; then [[ $# -eq 2 ]]; exec /tmp/apple_cos53_pluto_clean_bench "$2"; fi
if [[ "$MODE" == pluto ]]; then [[ $# -eq 2 ]]; exec /tmp/kernel_attack_no_p3 single "$2"; fi
if [[ "$MODE" == apple ]]; then [[ $# -eq 2 ]]; exec env VECLIB_MAXIMUM_THREADS=1 /tmp/apple_cos53_off_frozen apple "$2"; fi

echo "usage: $0 build | validate | one N | pluto N | apple N" >&2
exit 2
