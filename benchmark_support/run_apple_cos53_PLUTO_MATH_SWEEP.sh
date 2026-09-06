#!/usr/bin/env bash
set -euo pipefail

# PLUTO-MATH-SWEEP: mathematics-only successors of frozen PLUTO.
# Contract: single-thread only, same Mode-5/secant-spine family, <=2 ULP gate.
# Frozen PLUTO itself is never edited.

[[ "$(uname -m)" == arm64 ]]
[[ "$(sysctl -n machdep.cpu.brand_string)" == *"Apple M1"* ]]
MODE="${1:-}"
VARIANTS=(
  direct1_pi1280_d3
  pio2_k1280_d3 pio2_k640_d3 pio2_k512_d3 pio2_k320_d3 pio2_k256_d3
  pio2_k512_d5 pio2_k256_d5
  hybrid_k512_d3 hybrid_k256_d5
  nolut_t9 nolut_t10
  halfoct_t5 halfoct_t6 halfoct_t7
)

if [[ "$MODE" == build ]]; then
  # Exact frozen controls and exact generated PLUTO source/harness.
  bash benchmark_support/run_apple_cos53_PLUTO.sh build
  test -f /tmp/kernel_attack_no_p3.cpp
  test -x /tmp/kernel_attack_no_p3
  test -x /tmp/apple_cos53_off_frozen
  brew list flint >/dev/null 2>&1 || brew install flint
  brew list mpfr >/dev/null 2>&1 || brew install mpfr
  brew list gmp >/dev/null 2>&1 || brew install gmp

  rm -rf /tmp/pluto_math
  mkdir -p /tmp/pluto_math

  cat >/tmp/pluto_math/dump_coeff.c <<'C'
#include <flint/arf.h>
#include <flint/flint.h>
#include <flint/fmpz.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "SRCFILE"
static double cv(const mp_limb_t q[2], int neg) {
    fmpz_t z; arf_t a; fmpz_init(z); arf_init(a);
    fmpz_set_ui_array(z,q,2); arf_set_fmpz(a,z); arf_mul_2exp_si(a,a,-103);
    double d=arf_get_d(a,ARF_RND_NEAR); arf_clear(a); fmpz_clear(z); return neg?-d:d;
}
int main(int argc,char**argv){
    if(argc!=2)return 2; int terms=atoi(argv[1]);
    sine_fixed_ctx*c=s53_coeff_create_terms(terms); if(!c)return 3;
    printf("LUTN %lu DEG %d\n",(unsigned long)SF_LUT_N,c->poly_deg);
    for(size_t a=0;a<(size_t)SF_LUT_N;a++){
        size_t off=a*(size_t)(c->poly_deg+1); printf("A %zu",a);
        for(int k=0;k<=c->poly_deg;k++) printf(" %a",cv(c->coef+2*(off+(size_t)k),c->coef_sign[off+(size_t)k]!=0));
        putchar('\n');
    }
    s53_coeff_destroy(c); return 0;
}
C

  python3 - <<'PY'
from pathlib import Path
import math, re, shutil, subprocess, sys
sys.path.insert(0,'.')
import cosine53_apply_formula_conversion as conv
root=Path('/tmp/pluto_math')
base=Path('benchmark_support/sine_53_coeff_source.c').read_text()

def source_for(K,LUTN,cosine,name,maxdeg=24,maxterms=10):
    s=base
    s=s.replace('#define SF_LUT_N ((1UL << SF_K) + 1UL)',f'#define SF_LUT_N {LUTN}UL')
    s=s.replace('#define SF_MAX_DEG 8',f'#define SF_MAX_DEG {maxdeg}')
    s=s.replace('if (terms < 1 || terms > 3) return NULL;',f'if (terms < 1 || terms > {maxterms}) return NULL;')
    old='st |= nfloat_mul_2exp_si(delta, delta, -SF_K, b->nctx);'
    assert old in s
    s=s.replace(old,f'st |= nfloat_div_ui(delta, delta, {K}UL, b->nctx);')
    p=root/f'{name}.c'; p.write_text(s)
    if cosine: conv.patch_coeff(p)
    return p

FP=subprocess.check_output(['brew','--prefix','flint'],text=True).strip()
MP=subprocess.check_output(['brew','--prefix','mpfr'],text=True).strip()
GP=subprocess.check_output(['brew','--prefix','gmp'],text=True).strip()

def compile_dump(src,name):
    t=(root/'dump_coeff.c').read_text().replace('"SRCFILE"',f'"{src}"')
    c=root/f'dump_{name}.c'; c.write_text(t)
    exe=root/f'dump_{name}'
    subprocess.run(['clang','-O2','-DNDEBUG',f'-I{FP}/include',f'-I{MP}/include',f'-I{GP}/include',str(c),f'-L{FP}/lib',f'-L{MP}/lib',f'-L{GP}/lib','-lflint','-lmpfr','-lgmp','-lm','-o',str(exe)],check=True)
    return exe

def dump(exe,terms):
    txt=subprocess.check_output([str(exe),str(terms)],text=True).splitlines()
    m=re.match(r'LUTN (\d+) DEG (\d+)',txt[0]); assert m
    lutn,deg=map(int,m.groups()); rows=[]
    for line in txt[1:]:
        p=line.split(); assert p[0]=='A'; rows.append([float.fromhex(x) for x in p[2:]])
    assert len(rows)==lutn and all(len(r)==deg+1 for r in rows)
    return rows,deg

def emit_table(K,terms):
    # PIO2 residual is <=pi/4, so this LUT only needs that interval.
    lutn=int(math.floor((math.pi/4)*K+0.5))+2
    ss=source_for(K,lutn,False,f'sine_k{K}_t{terms}')
    cs=source_for(K,lutn,True,f'cos_k{K}_t{terms}')
    se=compile_dump(ss,f'sine_k{K}_t{terms}'); ce=compile_dump(cs,f'cos_k{K}_t{terms}')
    S,ds=dump(se,terms); C,dc=dump(ce,terms); assert ds==dc==2*terms+1
    deg=dc
    # Exact formula structure: even coefficients scale from cosine c0;
    # odd coefficients scale from sine c1.  Verify ratios across anchors.
    ratios={}
    for k in range(2,deg+1):
        if k%2==0:
            ratios[k]=C[0][k]/C[0][0]
        else:
            ratios[k]=S[0][k]/S[0][1]
    for a in range(min(lutn,64)):
        if abs(C[a][0])>1e-30:
            for k in range(2,deg+1,2):
                q=C[a][k]/C[a][0]
                assert abs(q-ratios[k]) <= 2e-13*max(1.0,abs(ratios[k]))
        if abs(S[a][1])>1e-30:
            for k in range(3,deg+1,2):
                q=S[a][k]/S[a][1]
                assert abs(q-ratios[k]) <= 2e-13*max(1.0,abs(ratios[k]))
    # Pack per anchor: cosine(c0,c1), sine(c0,c1). Two doubles are loaded per lane.
    h=['#pragma once','#include <cstddef>','#include <cstdint>',f'#define MATH_K {K}',f'#define MATH_LUTN {lutn}',f'#define MATH_DEG {deg}']
    h.append('alignas(16) static const double math_coeff[MATH_LUTN*4] = {')
    for a in range(lutn):
        h.append('  '+', '.join(v.hex() for v in (C[a][0],C[a][1],S[a][0],S[a][1]))+',')
    h.append('};')
    for k,v in ratios.items(): h.append(f'static constexpr double MATH_R{k} = {v.hex()};')
    # exact split of -1/K: hi is binary64, lo is rounded residual.
    from fractions import Fraction
    exact=-Fraction(1,K); hi=float(exact); lo=float(exact-Fraction.from_float(hi))
    h += [f'static constexpr double MATH_NIK_HI = {hi.hex()};',f'static constexpr double MATH_NIK_LO = {lo.hex()};']
    (root/f'table_k{K}_t{terms}.h').write_text('\n'.join(h)+'\n')
    return lutn,deg

for K,t in [(1280,1),(640,1),(512,1),(320,1),(256,1),(512,2),(256,2)]: emit_table(K,t)

# One-anchor high-order coefficients from the exact same sine/cosine Mode-5 spine.
ss=source_for(256,1,False,'sine_nolut'); cs=source_for(256,1,True,'cos_nolut')
se=compile_dump(ss,'sine_nolut'); ce=compile_dump(cs,'cos_nolut')
for t in [5,6,7,9,10]:
    S,ds=dump(se,t); C,dc=dump(ce,t); assert ds==dc==2*t+1
    p=root/f'nolut_t{t}.h'
    h=['#pragma once',f'#define NL_TERMS {t}',f'#define NL_DEG {dc}']
    for k in range(0,dc+1,2): h.append(f'static constexpr double NL_C{k} = {C[0][k].hex()};')
    for k in range(1,dc+1,2): h.append(f'static constexpr double NL_S{k} = {S[0][k].hex()};')
    p.write_text('\n'.join(h)+'\n')
PY

  # Generate the 15 translation units by replacing only PLUTO's opt_pair.
  python3 - <<'PY'
from pathlib import Path
import re
root=Path('/tmp/pluto_math')
base=Path('/tmp/kernel_attack_no_p3.cpp').read_text()
start=base.index('__attribute__((always_inline)) static inline void opt_pair')
end=base.index('\nstatic inline void opt_cos53_eval',start)

COMMON_HEAD=r'''
static constexpr double MATH_INVPI = 0x1.45f306dc9c883p-2;
static constexpr double MATH_INVPIO2 = 0x1.45f306dc9c883p-1;
static constexpr double MATH_FOUROPI = 0x1.45f306dc9c883p+0;
static constexpr double MATH_PI_P1 = 0x1.921fb54442000p+1;
static constexpr double MATH_PI_P2 = 0x1.a308d313198a3p-40;
static constexpr double MATH_PIO2_P1 = 0x1.921fb54442000p+0;
static constexpr double MATH_PIO2_P2 = 0x1.a308d313198a3p-41;
static constexpr double MATH_PIO4_P1 = 0x1.921fb54442000p-1;
static constexpr double MATH_PIO4_P2 = 0x1.a308d313198a3p-42;
'''

def local_poly(deg):
    s='''    float64x2_t aa=vld1q_f64(math_coeff+4*j0+2*use_sin0);\n    float64x2_t bb=vld1q_f64(math_coeff+4*j1+2*use_sin1);\n    float64x2_t c0=vzip1q_f64(aa,bb);\n    float64x2_t c1=vzip2q_f64(aa,bb);\n'''
    coeff=['c0','c1']
    for k in range(2,deg+1):
        basec='c0' if k%2==0 else 'c1'
        s+=f'    float64x2_t c{k}=vmulq_n_f64({basec},MATH_R{k});\n'; coeff.append(f'c{k}')
    s+=f'    float64x2_t p=c{deg};\n'
    for k in range(deg-1,-1,-1): s+=f'    p=vfmaq_f64(c{k},p,delta);\n'
    return s

def pio2_kernel(K,terms,direct=False):
    deg=2*terms+1; rootj=max(1,int(K*0.0025+0.999999))
    direct_block='''    uint64x2_t unit=vcltq_f64(ax,vdupq_n_f64(1.0));\n    bool allunit=((vgetq_lane_u64(unit,0)&vgetq_lane_u64(unit,1))==UINT64_MAX);\n''' if direct else '    bool allunit=false;\n'
    return COMMON_HEAD+f'''\n__attribute__((always_inline)) static inline void opt_pair(const double* x,double* y)\n{{\n    float64x2_t xv=vld1q_f64(x);\n    float64x2_t ax=vabsq_f64(xv);\n    const float64x2_t magic=vdupq_n_f64(0x1p52);\n{direct_block}    float64x2_t qd; uint64x2_t qbits; float64x2_t rh,rl;\n    if (__builtin_expect(allunit,0)) {{\n        qd=vdupq_n_f64(0.0); qbits=vreinterpretq_u64_f64(magic); rh=ax; rl=vdupq_n_f64(0.0);\n    }} else {{\n        float64x2_t qmagic=vaddq_f64(vmulq_n_f64(ax,MATH_INVPIO2),magic);\n        qd=vsubq_f64(qmagic,magic); qbits=vreinterpretq_u64_f64(qmagic);\n        float64x2_t t=vsubq_f64(ax,vmulq_n_f64(qd,MATH_PIO2_P1));\n        rh=vfmaq_n_f64(t,qd,-MATH_PIO2_P2);\n        float64x2_t d=vsubq_f64(t,rh);\n        rl=vfmaq_n_f64(d,qd,-MATH_PIO2_P2);\n    }}\n    const uint64x2_t sm=vdupq_n_u64(UINT64_C(0x8000000000000000));\n    uint64x2_t rsign=vandq_u64(vreinterpretq_u64_f64(rh),sm);\n    float64x2_t ah=vreinterpretq_f64_u64(vbicq_u64(vreinterpretq_u64_f64(rh),sm));\n    float64x2_t al=vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(rl),rsign));\n    float64x2_t jmagic=vaddq_f64(vmulq_n_f64(ah,(double)MATH_K),magic);\n    float64x2_t jd=vsubq_f64(jmagic,magic);\n    uint64x2_t jbits=vreinterpretq_u64_f64(jmagic);\n    constexpr uint64_t JMASK=(UINT64_C(1)<<52)-1;\n    uint64_t j0=vgetq_lane_u64(jbits,0)&JMASK, j1=vgetq_lane_u64(jbits,1)&JMASK;\n    uint64x2_t qodd=vandq_u64(qbits,vdupq_n_u64(1));\n    unsigned use_sin0=(unsigned)vgetq_lane_u64(qodd,0);\n    unsigned use_sin1=(unsigned)vgetq_lane_u64(qodd,1);\n    float64x2_t delta=vfmaq_n_f64(ah,jd,MATH_NIK_HI);\n#if MATH_K != 256 && MATH_K != 512
    delta=vfmaq_n_f64(delta,jd,MATH_NIK_LO);\n#endif\n    delta=vaddq_f64(delta,al);\n'''+local_poly(deg)+f'''\n    uint64x2_t qbit1=vandq_u64(vshrq_n_u64(qbits,1),vdupq_n_u64(1));\n    uint64x2_t rneg01=vshrq_n_u64(rsign,63);\n    uint64x2_t neg01=veorq_u64(veorq_u64(qbit1,qodd),vandq_u64(qodd,rneg01));\n    uint64x2_t outsign=vshlq_n_u64(neg01,63);\n    p=vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(p),outsign));\n    vst1q_f64(y,p);\n    if (__builtin_expect((use_sin0 && j0<={rootj}) || (use_sin1 && j1<={rootj}),0)) {{\n        if(use_sin0 && j0<={rootj}) y[0]=std::cos(x[0]);\n        if(use_sin1 && j1<={rootj}) y[1]=std::cos(x[1]);\n    }}\n}}\n'''

def direct1_kernel():
    # Same PLUTO pi reducer/local D3, except unit pairs bypass q*pi entirely.
    return COMMON_HEAD+r'''\n__attribute__((always_inline)) static inline void opt_pair(const double* x,double* y)\n{\n    float64x2_t xv=vld1q_f64(x), ax=vabsq_f64(xv);\n    const float64x2_t magic=vdupq_n_f64(0x1p52);\n    uint64x2_t unit=vcltq_f64(ax,vdupq_n_f64(1.0));\n    bool allunit=((vgetq_lane_u64(unit,0)&vgetq_lane_u64(unit,1))==UINT64_MAX);\n    float64x2_t qd,rh,rl; uint64x2_t qbits;\n    if(__builtin_expect(allunit,0)){ qd=vdupq_n_f64(0.0);qbits=vreinterpretq_u64_f64(magic);rh=ax;rl=vdupq_n_f64(0.0);}\n    else {\n      float64x2_t qmagic=vaddq_f64(vmulq_n_f64(ax,MATH_INVPI),magic); qd=vsubq_f64(qmagic,magic); qbits=vreinterpretq_u64_f64(qmagic);\n      float64x2_t t=vsubq_f64(ax,vmulq_n_f64(qd,MATH_PI_P1)); rh=vfmaq_n_f64(t,qd,-MATH_PI_P2);\n      float64x2_t d=vsubq_f64(t,rh); rl=vfmaq_n_f64(d,qd,-MATH_PI_P2);\n    }\n    const uint64x2_t sm=vdupq_n_u64(UINT64_C(0x8000000000000000)); uint64x2_t rsign=vandq_u64(vreinterpretq_u64_f64(rh),sm);\n    float64x2_t ah=vreinterpretq_f64_u64(vbicq_u64(vreinterpretq_u64_f64(rh),sm)); float64x2_t al=vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(rl),rsign));\n    float64x2_t jmagic=vaddq_f64(vmulq_n_f64(ah,1280.0),magic); float64x2_t jd=vsubq_f64(jmagic,magic); uint64x2_t jbits=vreinterpretq_u64_f64(jmagic);\n    constexpr uint64_t JM=(UINT64_C(1)<<52)-1; uint64_t j0=vgetq_lane_u64(jbits,0)&JM,j1=vgetq_lane_u64(jbits,1)&JM;\n    float64x2_t delta=vfmaq_n_f64(ah,jd,-0x1.999999999999ap-11); delta=vfmaq_n_f64(delta,jd,0x1.999999999999ap-65); delta=vaddq_f64(delta,al);\n    float64x2_t a=vld1q_f64(opt_cos53_coeff_aos+2*j0),b=vld1q_f64(opt_cos53_coeff_aos+2*j1);\n    float64x2_t c0=vzip1q_f64(a,b),c1=vzip2q_f64(a,b); float64x2_t c2=vmulq_n_f64(c0,-0x1.ffffff92c5f94p-2),c3=vmulq_n_f64(c1,-0x1.5555551eb851fp-3);\n    float64x2_t p=vfmaq_f64(c2,c3,delta);p=vfmaq_f64(c1,p,delta);p=vfmaq_f64(c0,p,delta);\n    uint64x2_t parity=vandq_u64(qbits,vdupq_n_u64(1)); p=vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(p),vshlq_n_u64(parity,63))); vst1q_f64(y,p);\n    if(__builtin_expect(j0>=2009,0))y[0]=std::cos(x[0]); if(__builtin_expect(j1>=2009,0))y[1]=std::cos(x[1]);\n}\n'''

def nolut_kernel(t):
    h=(root/f'nolut_t{t}.h').read_text(); vals={int(k):v for k,v in re.findall(r'NL_C(\d+) = ([^;]+);',h)}
    ks=sorted(vals); top=ks[-1]
    poly=f'    float64x2_t z=vmulq_f64(r,r);\n    float64x2_t p=vdupq_n_f64({vals[top]});\n'
    for k in reversed(ks[:-1]): poly+=f'    p=vfmaq_n_f64(vdupq_n_f64({vals[k]}),p,1.0); /* marker */\n' if False else ''
    # Explicit Horner in z.
    poly=f'    float64x2_t z=vmulq_f64(r,r);\n    float64x2_t p=vdupq_n_f64({vals[top]});\n'
    for k in reversed(ks[:-1]): poly+=f'    p=vfmaq_f64(vdupq_n_f64({vals[k]}),p,z);\n'
    return COMMON_HEAD+f'''\n__attribute__((always_inline)) static inline void opt_pair(const double*x,double*y){{\n    float64x2_t ax=vabsq_f64(vld1q_f64(x)); const float64x2_t magic=vdupq_n_f64(0x1p52);\n    float64x2_t qmagic=vaddq_f64(vmulq_n_f64(ax,MATH_INVPI),magic); float64x2_t qd=vsubq_f64(qmagic,magic); uint64x2_t qbits=vreinterpretq_u64_f64(qmagic);\n    float64x2_t tt=vsubq_f64(ax,vmulq_n_f64(qd,MATH_PI_P1)); float64x2_t rh=vfmaq_n_f64(tt,qd,-MATH_PI_P2); float64x2_t dd=vsubq_f64(tt,rh); float64x2_t rl=vfmaq_n_f64(dd,qd,-MATH_PI_P2);\n    float64x2_t r=vaddq_f64(rh,rl);\n{poly}    uint64x2_t parity=vandq_u64(qbits,vdupq_n_u64(1));p=vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(p),vshlq_n_u64(parity,63)));vst1q_f64(y,p);\n    float64x2_t ar=vabsq_f64(r); if(__builtin_expect(vgetq_lane_f64(ar,0)>1.568 || vgetq_lane_f64(ar,1)>1.568,0)){{if(vgetq_lane_f64(ar,0)>1.568)y[0]=std::cos(x[0]);if(vgetq_lane_f64(ar,1)>1.568)y[1]=std::cos(x[1]);}}\n}}\n'''

def halfoct_kernel(t):
    h=(root/f'nolut_t{t}.h').read_text(); C={int(k):v for k,v in re.findall(r'NL_C(\d+) = ([^;]+);',h)}; S={int(k):v for k,v in re.findall(r'NL_S(\d+) = ([^;]+);',h)}
    ce=sorted(C); so=sorted(S); ctop=ce[-1]; stop=so[-1]
    cp=f'    float64x2_t z=vmulq_f64(r,r); float64x2_t cp=vdupq_n_f64({C[ctop]});\n'
    for k in reversed(ce[:-1]): cp+=f'    cp=vfmaq_f64(vdupq_n_f64({C[k]}),cp,z);\n'
    sp=f'    float64x2_t sp=vdupq_n_f64({S[stop]});\n'
    for k in reversed(so[:-1]): sp+=f'    sp=vfmaq_f64(vdupq_n_f64({S[k]}),sp,z);\n'
    sp+='    sp=vmulq_f64(sp,r);\n'
    return COMMON_HEAD+f'''\n__attribute__((always_inline)) static inline void opt_pair(const double*x,double*y){{\n    float64x2_t ax=vabsq_f64(vld1q_f64(x)); const float64x2_t magic=vdupq_n_f64(0x1p52);\n    float64x2_t qmagic=vaddq_f64(vmulq_n_f64(ax,MATH_FOUROPI),magic);float64x2_t qd=vsubq_f64(qmagic,magic);uint64x2_t qb=vreinterpretq_u64_f64(qmagic);\n    float64x2_t tt=vsubq_f64(ax,vmulq_n_f64(qd,MATH_PIO4_P1));float64x2_t rh=vfmaq_n_f64(tt,qd,-MATH_PIO4_P2);float64x2_t dd=vsubq_f64(tt,rh);float64x2_t rl=vfmaq_n_f64(dd,qd,-MATH_PIO4_P2);float64x2_t r=vaddq_f64(rh,rl);\n{cp}{sp}    constexpr double H=0x1.6a09e667f3bccp-1; static const double CQ[8]={{1.0,H,0.0,-H,-1.0,-H,0.0,H}}; static const double SQ[8]={{0.0,H,1.0,H,0.0,-H,-1.0,-H}};\n    uint64_t q0=vgetq_lane_u64(qb,0)&7,q1=vgetq_lane_u64(qb,1)&7;float64x2_t cq={{CQ[q0],CQ[q1]}},sq={{SQ[q0],SQ[q1]}};float64x2_t p=vfmsq_f64(vmulq_f64(cp,cq),sp,sq);vst1q_f64(y,p);\n    if(__builtin_expect(((q0&3)==2 && fabs(vgetq_lane_f64(r,0))<0.003)||((q1&3)==2 && fabs(vgetq_lane_f64(r,1))<0.003),0)){{if((q0&3)==2&&fabs(vgetq_lane_f64(r,0))<0.003)y[0]=std::cos(x[0]);if((q1&3)==2&&fabs(vgetq_lane_f64(r,1))<0.003)y[1]=std::cos(x[1]);}}\n}}\n'''

def write_variant(name,kernel,header=None):
    s=base[:start]+kernel+base[end:]
    if header:
        s=s.replace('#include "apple_cos53_coeff_aos.h"',f'#include "{header}"')
    (root/f'{name}.cpp').write_text(s)

write_variant('direct1_pi1280_d3',direct1_kernel())
for K,t in [(1280,1),(640,1),(512,1),(320,1),(256,1),(512,2),(256,2)]:
    nm=f'pio2_k{K}_d{2*t+1}';write_variant(nm,pio2_kernel(K,t,False),f'table_k{K}_t{t}.h')
write_variant('hybrid_k512_d3',pio2_kernel(512,1,True),'table_k512_t1.h')
write_variant('hybrid_k256_d5',pio2_kernel(256,2,True),'table_k256_t2.h')
for t in [9,10]:write_variant(f'nolut_t{t}',nolut_kernel(t))
for t in [5,6,7]:write_variant(f'halfoct_t{t}',halfoct_kernel(t))
PY

  MP="$(brew --prefix mpfr)"; GP="$(brew --prefix gmp)"
  COMMON="-O3 -DNDEBUG -std=c++20 -mcpu=native -fno-fast-math -ffp-contract=fast"
  for v in "${VARIANTS[@]}"; do
    clang++ $COMMON -I/tmp -I/tmp/pluto_math -I/tmp/pthreadpool-install/include \
      "/tmp/pluto_math/${v}.cpp" /tmp/pthreadpool-install/lib/libpthreadpool.a \
      -framework Accelerate -pthread -o "/tmp/pluto_math_${v}"
    clang++ $COMMON -DOPT_VALIDATE_MPFR -I/tmp -I/tmp/pluto_math -I/tmp/pthreadpool-install/include \
      -I"$MP/include" -I"$GP/include" "/tmp/pluto_math/${v}.cpp" \
      /tmp/pthreadpool-install/lib/libpthreadpool.a -framework Accelerate -pthread \
      -L"$MP/lib" -L"$GP/lib" -lmpfr -lgmp -o "/tmp/pluto_math_${v}_validate"
  done
  exit 0
fi

if [[ "$MODE" == variants ]]; then printf '%s\n' "${VARIANTS[@]}"; exit 0; fi
if [[ "$MODE" == validate ]]; then [[ $# -eq 2 ]]; exec "/tmp/pluto_math_$2_validate" validate; fi
if [[ "$MODE" == one ]]; then [[ $# -eq 3 ]]; exec "/tmp/pluto_math_$2" single "$3"; fi
if [[ "$MODE" == pluto ]]; then [[ $# -eq 2 ]]; exec /tmp/kernel_attack_no_p3 single "$2"; fi
if [[ "$MODE" == apple ]]; then [[ $# -eq 2 ]]; exec env VECLIB_MAXIMUM_THREADS=1 /tmp/apple_cos53_off_frozen apple "$2"; fi

echo "usage: $0 build | variants | validate VARIANT | one VARIANT N | pluto N | apple N" >&2
exit 2
