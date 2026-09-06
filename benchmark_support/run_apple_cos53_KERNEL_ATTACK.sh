#!/usr/bin/env bash
set -euo pipefail

[[ "$(uname -m)" == arm64 ]]
[[ "$(sysctl -n machdep.cpu.brand_string)" == *"Apple M1"* ]]

MODE="${1:-}"
VARIANTS=(early_lut early_coeff early_sign fma_p1 root_onebranch no_p3 no_invlo estrin soa j_int fma_p1_no_p3 early_coeff_no_p3 sched_safe lean_safe lean_estrin simple_reduce)

if [[ "$MODE" == build ]]; then
  # Build exact frozen DESTINY stack first. This creates the exact conversion-free
  # K1280 hot source at /tmp/apple_cos53_hotloop.cpp and the Apple vvcos comparator.
  bash benchmark_support/run_apple_cos53_PATCH_cpu_fix.sh build
  test -f /tmp/apple_cos53_hotloop.cpp
  test -x /tmp/apple_cos53_hotloop_bench
  test -x /tmp/apple_cos53_off_frozen

  python3 - <<'PY'
from pathlib import Path
import re
base=Path('/tmp/apple_cos53_hotloop.cpp').read_text()

# SoA copy of the same frozen LUT values. This changes layout only, never coefficients.
h=Path('/tmp/apple_cos53_coeff_aos.h').read_text()
m=re.search(r'opt_cos53_coeff_aos\[OPT_COS53_LUTN\*2\] = \{(.*?)\};',h,re.S); assert m
vals=re.findall(r'-?0x[0-9a-fA-F]+(?:\.[0-9a-fA-F]*)?p[+-]?\d+',m.group(1)); assert len(vals)==4024
c0=vals[0::2]; c1=vals[1::2]
out=['#pragma once','#include <cstddef>','alignas(16) static const double opt_cos53_c0_soa[OPT_COS53_LUTN] = {']
out += ['  '+v+',' for v in c0]; out += ['};','alignas(16) static const double opt_cos53_c1_soa[OPT_COS53_LUTN] = {']
out += ['  '+v+',' for v in c1]; out += ['};']
Path('/tmp/apple_cos53_coeff_soa.h').write_text('\n'.join(out)+'\n')

QTAIL='''    uint64x2_t qbits=vreinterpretq_u64_f64(qmagic);\n'''
SIGN='''    // (-1)^q directly from the low bit of the magic-biased representation.\n    uint64x2_t parity=vandq_u64(qbits,vdupq_n_u64(1));\n    uint64x2_t outsign=vshlq_n_u64(parity,63);\n'''
P1='''    float64x2_t qp1=vmulq_n_f64(qd,PI_P1);\n    float64x2_t t=vsubq_f64(ax,qp1);\n'''
LOW='''    float64x2_t d=vsubq_f64(t,rh);\n    float64x2_t rl=vfmaq_n_f64(d,qd,-PI_P2);\n    rl=vfmaq_n_f64(rl,qd,-PI_P3);\n'''
J_MAGIC='''    float64x2_t jscaled=vmulq_n_f64(ah,KGRID);\n    float64x2_t jmagic=vaddq_f64(jscaled,magic);\n    float64x2_t jd=vsubq_f64(jmagic,magic);\n    uint64x2_t jbits=vreinterpretq_u64_f64(jmagic);\n    constexpr uint64_t JMASK=(UINT64_C(1)<<52)-1;\n    const uint64_t j0=vgetq_lane_u64(jbits,0)&JMASK;\n    const uint64_t j1=vgetq_lane_u64(jbits,1)&JMASK;\n'''
DELTA='''    // Split reciprocal subtraction: delta = |r| - j/1280 + low.\n    float64x2_t delta=vfmaq_n_f64(ah,jd,NINVK_HI);\n    delta=vfmaq_n_f64(delta,jd,NINVK_LO);\n    delta=vaddq_f64(delta,al);\n'''
LOAD='''    // AoS coefficient layout: two 128-bit loads, then native zip/deinterleave.\n    float64x2_t a=vld1q_f64(opt_cos53_coeff_aos+2*j0);\n    float64x2_t b=vld1q_f64(opt_cos53_coeff_aos+2*j1);\n    float64x2_t c0=vzip1q_f64(a,b);\n    float64x2_t c1=vzip2q_f64(a,b);\n'''
C23='''    float64x2_t c2=vmulq_n_f64(c0,MH);\n    float64x2_t c3=vmulq_n_f64(c1,M6);\n'''
HORNER='''    float64x2_t p=vfmaq_f64(c2,c3,delta);\n    p=vfmaq_f64(c1,p,delta);\n    p=vfmaq_f64(c0,p,delta);\n'''
ROOT='''    if (__builtin_expect(j0 >= 2009, 0)) y[0]=std::cos(x[0]);\n    if (__builtin_expect(j1 >= 2009, 0)) y[1]=std::cos(x[1]);\n'''

for needle in [QTAIL,SIGN,P1,LOW,J_MAGIC,DELTA,LOAD,C23,HORNER,ROOT]:
    assert needle in base, needle[:60]

def early_sign(s):
    assert s.count(SIGN)==1
    s=s.replace(SIGN,'',1)
    return s.replace(QTAIL,QTAIL+'\n'+SIGN,1)

def fma_p1(s):
    return s.replace(P1,'    float64x2_t t=vfmaq_n_f64(ax,qd,-PI_P1);\n',1)

def no_p3(s):
    old='    rl=vfmaq_n_f64(rl,qd,-PI_P3);\n'
    assert old in s
    return s.replace(old,'',1)

def no_invlo(s):
    old='    delta=vfmaq_n_f64(delta,jd,NINVK_LO);\n'
    assert old in s
    return s.replace(old,'',1)

def root_one(s):
    new='''    if (__builtin_expect((j0 >= 2009) || (j1 >= 2009), 0)) {\n        if (j0 >= 2009) y[0]=std::cos(x[0]);\n        if (j1 >= 2009) y[1]=std::cos(x[1]);\n    }\n'''
    return s.replace(ROOT,new,1)

def move_loads(s, coeffmul=False):
    assert LOAD in s and DELTA in s
    s=s.replace(LOAD,'',1)
    ins=LOAD
    if coeffmul:
        assert C23 in s
        s=s.replace(C23,'',1)
        ins += C23
    return s.replace(DELTA,ins+'\n'+DELTA,1)

def estrin(s):
    assert HORNER in s
    new='''    float64x2_t d2=vmulq_f64(delta,delta);\n    float64x2_t e0=vfmaq_f64(c0,c1,delta);\n    float64x2_t e1=vfmaq_f64(c2,c3,delta);\n    float64x2_t p=vfmaq_f64(e0,e1,d2);\n'''
    return s.replace(HORNER,new,1)

def jint(s):
    assert J_MAGIC in s
    new='''    int64x2_t ji=vcvtnq_s64_f64(vmulq_n_f64(ah,KGRID));\n    const int64_t j0=vgetq_lane_s64(ji,0);\n    const int64_t j1=vgetq_lane_s64(ji,1);\n    float64x2_t jd=vcvtq_f64_s64(ji);\n'''
    return s.replace(J_MAGIC,new,1)

def soa(s):
    inc='#include "apple_cos53_coeff_aos.h"\n'
    assert inc in s and LOAD in s
    s=s.replace(inc,inc+'#include "apple_cos53_coeff_soa.h"\n',1)
    new='''    // SoA experiment: no zip/deinterleave, four lane loads from two arrays.\n    float64x2_t c0=vdupq_n_f64(opt_cos53_c0_soa[j0]);\n    c0=vsetq_lane_f64(opt_cos53_c0_soa[j1],c0,1);\n    float64x2_t c1=vdupq_n_f64(opt_cos53_c1_soa[j0]);\n    c1=vsetq_lane_f64(opt_cos53_c1_soa[j1],c1,1);\n'''
    return s.replace(LOAD,new,1)

def simple_reduce(s):
    assert LOW in s
    return s.replace(LOW,'    float64x2_t rl=vdupq_n_f64(0.0);\n',1)

def make(name, funcs):
    s=base
    for f in funcs: s=f(s)
    Path('/tmp/kernel_attack_'+name+'.cpp').write_text(s)

make('early_lut',[lambda s:move_loads(s,False)])
make('early_coeff',[lambda s:move_loads(s,True)])
make('early_sign',[early_sign])
make('fma_p1',[fma_p1])
make('root_onebranch',[root_one])
make('no_p3',[no_p3])
make('no_invlo',[no_invlo])
make('estrin',[estrin])
make('soa',[soa])
make('j_int',[jint])
make('fma_p1_no_p3',[fma_p1,no_p3])
make('early_coeff_no_p3',[lambda s:move_loads(s,True),no_p3])
make('sched_safe',[fma_p1,lambda s:move_loads(s,True),early_sign,root_one])
make('lean_safe',[fma_p1,lambda s:move_loads(s,True),early_sign,root_one,no_p3])
make('lean_estrin',[fma_p1,lambda s:move_loads(s,True),early_sign,root_one,no_p3,estrin])
make('simple_reduce',[fma_p1,lambda s:move_loads(s,True),early_sign,root_one,simple_reduce])
PY

  MP="$(brew --prefix mpfr)"; GP="$(brew --prefix gmp)"
  COMMON="-O3 -DNDEBUG -std=c++20 -mcpu=native -fno-fast-math -ffp-contract=fast -DOPT_VALIDATE_MPFR"
  for v in "${VARIANTS[@]}"; do
    clang++ $COMMON -I/tmp -I/tmp/pthreadpool-install/include -I"$MP/include" -I"$GP/include" \
      "/tmp/kernel_attack_${v}.cpp" /tmp/pthreadpool-install/lib/libpthreadpool.a \
      -framework Accelerate -pthread -L"$MP/lib" -L"$GP/lib" -lmpfr -lgmp \
      -o "/tmp/kernel_attack_${v}"
  done
  # Emit assembly for the exact base and the strongest planned composite for inspection.
  clang++ -O3 -DNDEBUG -std=c++20 -mcpu=native -fno-fast-math -ffp-contract=fast \
    -I/tmp -I/tmp/pthreadpool-install/include -S /tmp/apple_cos53_hotloop.cpp -o /tmp/kernel_attack_destiny.s
  clang++ -O3 -DNDEBUG -std=c++20 -mcpu=native -fno-fast-math -ffp-contract=fast \
    -I/tmp -I/tmp/pthreadpool-install/include -S /tmp/kernel_attack_lean_safe.cpp -o /tmp/kernel_attack_lean_safe.s
  exit 0
fi

if [[ "$MODE" == variants ]]; then printf '%s\n' "${VARIANTS[@]}"; exit 0; fi
if [[ "$MODE" == destiny ]]; then [[ $# -eq 2 ]]; exec /tmp/apple_cos53_hotloop_bench single "$2"; fi
if [[ "$MODE" == apple ]]; then [[ $# -eq 2 ]]; exec env VECLIB_MAXIMUM_THREADS=1 /tmp/apple_cos53_off_frozen apple "$2"; fi
if [[ "$MODE" == one ]]; then [[ $# -eq 3 ]]; exec "/tmp/kernel_attack_$2" single "$3"; fi
if [[ "$MODE" == validate ]]; then [[ $# -eq 2 ]]; exec "/tmp/kernel_attack_$2" validate; fi

echo "usage: $0 build | variants | destiny N | apple N | one VARIANT N | validate VARIANT" >&2
exit 2
