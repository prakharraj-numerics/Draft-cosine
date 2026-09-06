#!/usr/bin/env bash
set -euo pipefail

# PLUTO-HIGHWAY: single-thread Google Highway translation of frozen PLUTO.
# Baseline source: freeze/apple-cos53-PLUTO-20260906 @ d4110c3d...
# Math is intentionally unchanged from PLUTO no-P3. Only the SIMD expression
# layer is translated from direct arm_neon intrinsics to Highway 1.4.0 static
# Full128<double> operations. No helper threads / pools / QoS / workgroups are
# linked into the candidate executable.

[[ "$(uname -m)" == arm64 ]]
[[ "$(sysctl -n machdep.cpu.brand_string)" == *"Apple M1"* ]]
MODE="${1:-}"

if [[ "$MODE" == build ]]; then
  # Build exact frozen PLUTO first. Besides the controls, this reproducibly
  # generates the exact no-P3 source and pinned Highway 1.4.0 checkout.
  bash benchmark_support/run_apple_cos53_PLUTO.sh build
  test -f /tmp/kernel_attack_no_p3.cpp
  test -f /tmp/apple_cos53_coeff_aos.h
  test -d /tmp/highway/hwy
  test -x /tmp/kernel_attack_no_p3
  test -x /tmp/apple_cos53_off_frozen

  python3 - <<'PY'
from pathlib import Path
s=Path('/tmp/kernel_attack_no_p3.cpp').read_text()

# Remove headers used only by PLUTO's direct NEON/multicore/Apple benchmark
# plumbing. Highway is compiled static-only for the native M1 target.
for inc in [
    '#include <Accelerate/Accelerate.h>\n',
    '#include <arm_neon.h>\n',
    '#include <pthreadpool.h>\n',
    '#include <pthread.h>\n',
    '#include <atomic>\n',
    '#include <thread>\n',
]:
    assert inc in s, inc
    s=s.replace(inc,'',1)

marker='#include "apple_cos53_coeff_aos.h"\n'
assert marker in s
s=s.replace(marker,
'''#define HWY_COMPILE_ONLY_STATIC\n#include <hwy/highway.h>\n#include "apple_cos53_coeff_aos.h"\n\nnamespace hn = hwy::HWY_NAMESPACE;\n''',1)

# Replace only PLUTO's two-lane SIMD pair kernel. The reduction, magic-bias
# q/j rounding, LUT layout, polynomial, parity sign, root repair, constants and
# operation ordering are preserved. Highway Full128<double> is two lanes on M1.
a=s.index('__attribute__((always_inline)) static inline void opt_pair(const double* x,double* y)')
b=s.index('static inline void opt_cos53_eval',a)
hwy_pair=r'''HWY_ATTR __attribute__((always_inline)) static inline void opt_pair(const double* x,double* y)
{
    hn::Full128<double> d;
    hn::RebindToUnsigned<decltype(d)> du;

    const auto xv=hn::LoadU(d,x);
    const auto ax=hn::Abs(xv);

    // q = round(|x|/pi) via the exact same 2^52 magic-bias scheme as PLUTO.
    const auto magic=hn::Set(d,0x1p52);
    const auto qscaled=hn::Mul(ax,hn::Set(d,INVPI));
    const auto qmagic=hn::Add(qscaled,magic);
    const auto qd=hn::Sub(qmagic,magic);
    const auto qbits=hn::BitCast(du,qmagic);

    // Same split-pi Cody-Waite reduction; PLUTO deliberately omits PI_P3.
    const auto qp1=hn::Mul(qd,hn::Set(d,PI_P1));
    const auto t=hn::Sub(ax,qp1);
    const auto rh=hn::MulAdd(qd,hn::Set(d,-PI_P2),t);
    const auto dd=hn::Sub(t,rh);
    const auto rl=hn::MulAdd(qd,hn::Set(d,-PI_P2),dd);

    const auto signmask=hn::Set(du,UINT64_C(0x8000000000000000));
    const auto rhbits=hn::BitCast(du,rh);
    const auto rsign=hn::And(rhbits,signmask);
    const auto ah=hn::BitCast(d,hn::AndNot(signmask,rhbits));
    const auto al=hn::BitCast(d,hn::Xor(hn::BitCast(du,rl),rsign));

    // Same K=1280 magic-bias grid index and LUT-address extraction.
    const auto jscaled=hn::Mul(ah,hn::Set(d,KGRID));
    const auto jmagic=hn::Add(jscaled,magic);
    const auto jd=hn::Sub(jmagic,magic);
    const auto jbits=hn::BitCast(du,jmagic);
    constexpr uint64_t JMASK=(UINT64_C(1)<<52)-1;
    const uint64_t j0=hn::GetLane(jbits)&JMASK;
    const uint64_t j1=hn::ExtractLane(jbits,1)&JMASK;

    // Same split reciprocal correction.
    auto delta=hn::MulAdd(jd,hn::Set(d,NINVK_HI),ah);
    delta=hn::MulAdd(jd,hn::Set(d,NINVK_LO),delta);
    delta=hn::Add(delta,al);

    // Same AoS table and the same native two-load deinterleave shape.
    const auto aa=hn::LoadU(d,opt_cos53_coeff_aos+2*j0);
    const auto bb=hn::LoadU(d,opt_cos53_coeff_aos+2*j1);
    const auto c0=hn::InterleaveLower(aa,bb);
    const auto c1=hn::InterleaveUpper(d,aa,bb);

    // Frozen PLUTO degree-3 polynomial, same FMA dependency order.
    const auto c2=hn::Mul(c0,hn::Set(d,MH));
    const auto c3=hn::Mul(c1,hn::Set(d,M6));
    auto p=hn::MulAdd(c3,delta,c2);
    p=hn::MulAdd(p,delta,c1);
    p=hn::MulAdd(p,delta,c0);

    // Same (-1)^q parity sign reconstruction.
    const auto parity=hn::And(qbits,hn::Set(du,UINT64_C(1)));
    const auto outsign=hn::ShiftLeft<63>(parity);
    p=hn::BitCast(d,hn::Xor(hn::BitCast(du,p),outsign));
    hn::StoreU(p,d,y);

    // Same rare root-only scalar repair.
    if (__builtin_expect(j0 >= 2009, 0)) y[0]=std::cos(x[0]);
    if (__builtin_expect(j1 >= 2009, 0)) y[1]=std::cos(x[1]);
}

'''
s=s[:a]+hwy_pair+s[b:]

# Remove all multicore runner classes. Keep the exact PLUTO unroll-8
# opt_cos53_eval and the benchmark input/accuracy machinery from fill_case on.
a=s.index('class Adaptive2 {')
b=s.index('static void fill_case',a)
s=s[:a]+s[b:]

# Replace the generic/multicore benchmark runner with a direct single-thread
# benchmark, exactly analogous to PLUTO-CLEAN.
a=s.index('template<class R> static int bench_runner')
b=s.index('#ifdef OPT_VALIDATE_MPFR',a)
bench=r'''static int bench_highway(size_t n) {
    Buffers b(n); size_t reps=reps_for(n);
    for(int w=0;w<12;w++) for(int c=0;c<6;c++) opt_cos53_eval(b.x[c],b.y[c],n);
    double c0=process_cpu_ns(); uint64_t w0=now_ticks();
    for(size_t rr=0;rr<reps;rr++) for(int c=0;c<6;c++) opt_cos53_eval(b.x[c],b.y[c],n);
    uint64_t w1=now_ticks(); double c1=process_cpu_ns();
    double den=(double)reps*(double)n*6.0;
    double wall=ticks_to_ns(w1-w0)/den, cpu=(c1-c0)/den;
    g_sink+=b.y[0][(n*7/11)%n];
    std::printf("APPLE_COS53_CPU_EFF_RESULT stack=PLUTO_HIGHWAY n=%zu wall_ns_el=%.9f cpu_ns_el=%.9f effective_cores=%.6f reps=%zu sink=%.17g\n",
                n,wall,cpu,cpu/wall,reps,(double)g_sink);
    return 0;
}

'''
s=s[:a]+bench+s[b:]

# Direct candidate main only; validation remains the exact inherited MPFR gate.
a=s.index('int main(int argc,char**argv) {')
main=r'''int main(int argc,char**argv) {
#ifdef OPT_VALIDATE_MPFR
    if(argc==2 && std::string(argv[1])=="validate") return validate();
#endif
    if(argc!=2)return 2;
    size_t n=(size_t)std::strtoull(argv[1],nullptr,10);
    return bench_highway(n);
}
'''
s=s[:a]+main

# Hard structural and mathematical gates.
for bad in ['arm_neon','float64x2_t','uint64x2_t','int64x2_t','vld1q_','vfmaq_',
            'pthreadpool','pthread_set_qos','QOS_CLASS_','Adaptive2','PoolRunner',
            'AutoRunner','os_workgroup','std::thread','vvcos','APPLE_SINGLE','pool_piece']:
    if bad in s:
        raise SystemExit(f'forbidden residue in PLUTO-HIGHWAY source: {bad}')
assert '#include <hwy/highway.h>' in s
assert 'hn::Full128<double> d;' in s
assert 'hn::MulAdd' in s
assert 'PI_P3' in s  # constant declaration is retained for provenance
assert 'hn::Set(d,-PI_P3)' not in s
assert 'rl=vfmaq_n_f64(rl,qd,-PI_P3);' not in s
assert 'for (; i+8<=n; i+=8)' in s

Path('/tmp/apple_cos53_pluto_highway.cpp').write_text(s)
PY

  MP="$(brew --prefix mpfr)"; GP="$(brew --prefix gmp)"
  COMMON="-O3 -DNDEBUG -std=c++20 -mcpu=native -fno-fast-math -ffp-contract=fast -DHWY_COMPILE_ONLY_STATIC"
  clang++ $COMMON -I/tmp -I/tmp/highway /tmp/apple_cos53_pluto_highway.cpp \
    -o /tmp/apple_cos53_pluto_highway_bench
  clang++ $COMMON -DOPT_VALIDATE_MPFR -I/tmp -I/tmp/highway -I"$MP/include" -I"$GP/include" \
    /tmp/apple_cos53_pluto_highway.cpp -L"$MP/lib" -L"$GP/lib" -lmpfr -lgmp \
    -o /tmp/apple_cos53_pluto_highway_validate
  clang++ $COMMON -I/tmp -I/tmp/highway -S /tmp/apple_cos53_pluto_highway.cpp \
    -o /tmp/apple_cos53_pluto_highway.s

  if nm -u /tmp/apple_cos53_pluto_highway_bench | grep -E 'pthreadpool|pthread_set_qos|os_workgroup|vvcos' ; then
    echo 'unexpected threading/Apple symbol in PLUTO-HIGHWAY' >&2; exit 1
  fi
  exit 0
fi

if [[ "$MODE" == validate ]]; then exec /tmp/apple_cos53_pluto_highway_validate validate; fi
if [[ "$MODE" == one ]]; then [[ $# -eq 2 ]]; exec /tmp/apple_cos53_pluto_highway_bench "$2"; fi
if [[ "$MODE" == pluto ]]; then [[ $# -eq 2 ]]; exec /tmp/kernel_attack_no_p3 single "$2"; fi
if [[ "$MODE" == apple ]]; then [[ $# -eq 2 ]]; exec env VECLIB_MAXIMUM_THREADS=1 /tmp/apple_cos53_off_frozen apple "$2"; fi

echo "usage: $0 build | validate | one N | pluto N | apple N" >&2
exit 2
