#!/usr/bin/env bash
set -euo pipefail

[[ "$(uname -m)" == arm64 ]]
[[ "$(sysctl -n machdep.cpu.brand_string)" == *"Apple M1"* ]]

MODE="${1:-}"

if [[ "$MODE" == build ]]; then
  test -f /tmp/apple_cos53_cpu_eff.cpp
  test -f /tmp/apple_cos53_apple_specific.cpp

  python3 - <<'PY'
from pathlib import Path
p=Path('/tmp/apple_cos53_cpu_eff.cpp')
s=p.read_text()

old='''    uint64x2_t valid=vcleq_f64(ax,vdupq_n_f64(XMAX));
    if ((vgetq_lane_u64(valid,0)&vgetq_lane_u64(valid,1)) != UINT64_MAX) {
        y[0]=std::cos(x[0]); y[1]=std::cos(x[1]); return;
    }

'''
assert old in s
s=s.replace(old,'''    // Hot-path contract: caller has already selected the supported |x|<=10000 domain.
    // Do not repeat a vector mask extraction/branch for every two elements.

''',1)

old='''    // q = nearest integer to |x|/pi, entirely vectorized.
    int64x2_t qi=vcvtnq_s64_f64(vmulq_n_f64(ax,INVPI));
    float64x2_t qd=vcvtq_f64_s64(qi);
'''
assert old in s
s=s.replace(old,'''    // q = nearest integer to |x|/pi.  Since q<=3183, the 2^52 bias gives
    // round-to-nearest-even while retaining both q-as-double and q parity bits,
    // avoiding FP<->integer conversion instructions in the common path.
    const float64x2_t magic=vdupq_n_f64(0x1p52);
    float64x2_t qscaled=vmulq_n_f64(ax,INVPI);
    float64x2_t qmagic=vaddq_f64(qscaled,magic);
    float64x2_t qd=vsubq_f64(qmagic,magic);
    uint64x2_t qbits=vreinterpretq_u64_f64(qmagic);
''',1)

old='''    // K=1280 and |r|<=pi/2 imply j is provably in [0,2011], so no clamps.
    int64x2_t ji=vcvtnq_s64_f64(vmulq_n_f64(ah,KGRID));
    const int64_t j0=vgetq_lane_s64(ji,0);
    const int64_t j1=vgetq_lane_s64(ji,1);
    float64x2_t jd=vcvtq_f64_s64(ji);
'''
assert old in s
s=s.replace(old,'''    // K=1280 and |r|<=pi/2 imply j is in [0,2011].  Use the same 2^52
    // bias so one rounded value supplies jd and the integer LUT address bits.
    float64x2_t jscaled=vmulq_n_f64(ah,KGRID);
    float64x2_t jmagic=vaddq_f64(jscaled,magic);
    float64x2_t jd=vsubq_f64(jmagic,magic);
    uint64x2_t jbits=vreinterpretq_u64_f64(jmagic);
    constexpr uint64_t JMASK=(UINT64_C(1)<<52)-1;
    const uint64_t j0=vgetq_lane_u64(jbits,0)&JMASK;
    const uint64_t j1=vgetq_lane_u64(jbits,1)&JMASK;
''',1)

old='''    // (-1)^q sign without scalar q extraction.
    uint64x2_t parity=vandq_u64(vreinterpretq_u64_s64(qi),vdupq_n_u64(1));
'''
assert old in s
s=s.replace(old,'''    // (-1)^q directly from the low bit of the magic-biased representation.
    uint64x2_t parity=vandq_u64(qbits,vdupq_n_u64(1));
''',1)

old='''static inline void opt_cos53_eval(const double* x,double* y,size_t n)
{
    // Two independent Full128-equivalent vectors per iteration: lets the M1
    // overlap range-reduction, coefficient-load, and FMA dependency chains.
    size_t i=0;
    for (; i+4<=n; i+=4) {
        opt_pair(x+i,y+i);
        opt_pair(x+i+2,y+i+2);
    }
    if (i+2<=n) { opt_pair(x+i,y+i); i+=2; }
    if (i<n) {
        double xx[2]={x[i],x[i]}, yy[2];
        opt_pair(xx,yy); y[i]=yy[0];
    }
}
'''
assert old in s
s=s.replace(old,'''static inline void opt_cos53_eval(const double* __restrict x,double* __restrict y,size_t n)
{
    // Four independent 128-bit vectors per iteration.  This exposes more
    // range-reduction/LUT/FMA work to the M1 scheduler and reduces loop overhead.
    size_t i=0;
    for (; i+8<=n; i+=8) {
        opt_pair(x+i,y+i);
        opt_pair(x+i+2,y+i+2);
        opt_pair(x+i+4,y+i+4);
        opt_pair(x+i+6,y+i+6);
    }
    for (; i+2<=n; i+=2) opt_pair(x+i,y+i);
    if (i<n) {
        double xx[2]={x[i],x[i]}, yy[2];
        opt_pair(xx,yy); y[i]=yy[0];
    }
}
''',1)

Path('/tmp/apple_cos53_hotloop.cpp').write_text(s)

w=Path('/tmp/apple_cos53_apple_specific.cpp').read_text()
old='#include "/tmp/apple_cos53_cpu_eff.cpp"'
assert old in w
w=w.replace(old,'#include "/tmp/apple_cos53_hotloop.cpp"',1)
Path('/tmp/apple_cos53_hotloop_apple_specific.cpp').write_text(w)
PY

  MP="$(brew --prefix mpfr)"; GP="$(brew --prefix gmp)"
  COMMON="-O3 -DNDEBUG -std=c++20 -mcpu=native -fno-fast-math -ffp-contract=off"

  clang++ $COMMON -I/tmp -I/tmp/pthreadpool-install/include \
    /tmp/apple_cos53_hotloop.cpp /tmp/pthreadpool-install/lib/libpthreadpool.a \
    -framework Accelerate -pthread -o /tmp/apple_cos53_hotloop_bench

  clang++ $COMMON -DOPT_VALIDATE_MPFR -I/tmp -I/tmp/pthreadpool-install/include \
    -I"$MP/include" -I"$GP/include" /tmp/apple_cos53_hotloop.cpp \
    /tmp/pthreadpool-install/lib/libpthreadpool.a -framework Accelerate -pthread \
    -L"$MP/lib" -L"$GP/lib" -lmpfr -lgmp -o /tmp/apple_cos53_hotloop_validate

  clang++ $COMMON -I/tmp -I/tmp/pthreadpool-install/include \
    /tmp/apple_cos53_hotloop_apple_specific.cpp /tmp/pthreadpool-install/lib/libpthreadpool.a \
    -framework Accelerate -pthread -o /tmp/apple_cos53_hotloop_apple_specific_bench
  exit 0
fi

if [[ "$MODE" == validate ]]; then
  exec /tmp/apple_cos53_hotloop_validate validate
fi

if [[ "$MODE" == one ]]; then
  [[ $# -eq 3 ]]
  exec /tmp/apple_cos53_hotloop_bench "$2" "$3"
fi

if [[ "$MODE" == apple_one ]]; then
  [[ $# -eq 3 ]]
  exec /tmp/apple_cos53_hotloop_apple_specific_bench "$2" "$3"
fi

echo "usage: $0 build | validate | one {single|adapt2|pool2|pool3|auto|apple} N | apple_one {single|wg2|wg3}_{ui|user|default|utility} N" >&2
exit 2
