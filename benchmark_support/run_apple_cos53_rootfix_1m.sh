#!/usr/bin/env bash
set -euo pipefail
[[ "$(uname -m)" == arm64 ]]
[[ "$(sysctl -n machdep.cpu.brand_string)" == *"Apple M1"* ]]
brew list flint >/dev/null 2>&1 || brew install flint
brew list mpfr >/dev/null 2>&1 || brew install mpfr
brew list gmp >/dev/null 2>&1 || brew install gmp
rm -rf /tmp/highway
git clone --depth 1 --branch 1.4.0 https://github.com/google/highway.git /tmp/highway

FREEZE=aefbe778e860ef70e64fc8d6b6d470b3575f3bbc
git show "$FREEZE":benchmark_support/apple_cos53_highway_benchmark.cpp > /tmp/base.cpp
git show "$FREEZE":benchmark_support/sine_53_coeff_source.c > /tmp/src.c
git show "$FREEZE":benchmark_support/apple_cos53_coeff_bridge.c > /tmp/bridge.c
git show "$FREEZE":benchmark_support/apple_cos53_generate_constants.c > /tmp/gen.c
git show "$FREEZE":cosine53_apply_formula_conversion.py > /tmp/cosine53_apply_formula_conversion.py
python3 - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0,'/tmp')
p=Path('/tmp/src.c')
s=p.read_text().replace('#define SF_K 12','#define SF_K 11').replace('#define SF_LUT_N ((1UL << SF_K) + 1UL)','#define SF_LUT_N 3218UL')
p.write_text(s)
p=Path('/tmp/bridge.c')
s=p.read_text().replace('#include "apple_cosine53_coeff_source.c"','#include "src.c"').replace('s53_coeff_create_terms(2)','s53_coeff_create_terms(1)').replace('c->poly_deg != 5','c->poly_deg != 3')
p.write_text(s)
p=Path('/tmp/gen.c'); p.write_text(p.read_text().replace('#define LUTN 403','#define LUTN 3218'))
import cosine53_apply_formula_conversion as m
m.patch_coeff(Path('/tmp/src.c'))
PY
FP="$(brew --prefix flint)"; MP="$(brew --prefix mpfr)"; GP="$(brew --prefix gmp)"
clang -O2 -DNDEBUG -I/tmp -I$FP/include -I$MP/include -I$GP/include /tmp/bridge.c /tmp/gen.c -L$FP/lib -L$MP/lib -L$GP/lib -lflint -lmpfr -lgmp -lm -o /tmp/gen
/tmp/gen /tmp/apple_cos53_constants_2ulp.h

python3 - <<'PY'
from pathlib import Path
s=Path('/tmp/base.cpp').read_text()
s=s.replace('#include "apple_cos53_constants.h"','#include "apple_cos53_constants_2ulp.h"')
s=s.replace('static constexpr double KGRID = 256.0;','static constexpr double KGRID = 2048.0;').replace('static constexpr double INVK = 1.0 / 256.0;','static constexpr double INVK = 1.0 / 2048.0;')
s=s.replace('    const auto c24v = hn::Set(d, 1.0/24.0);\n    const auto c120v = hn::Set(d, 1.0/120.0);\n','')
start=s.index('        auto c2 = hn::Mul(c0, mh);')
end=s.index('        uint64_t sm0=', start)
s=s[:start]+'''        auto c2 = hn::Mul(c0, mh);\n        auto c3 = hn::Mul(c1, m6);\n        auto p = hn::MulAdd(c3, delta, c2);\n        p = hn::MulAdd(p, delta, c1);\n        p = hn::MulAdd(p, delta, c0);\n\n'''+s[end:]
helper=r'''
static inline void root_twos(double a,double b,double *h,double *l)
{
    double x=a+b,bv=x-a,av=x-bv,br=b-bv,ar=a-av; *h=x; *l=ar+br;
}
static double cos53_rootfix_one(double x)
{
    const double PIO2_HI=0x1.921fb54442d18p+0;
    const double PIO2_LO=0x1.1a62633145c07p-54;
    const double PIO2_TINY=-0x1.f1976b7ed8fbcp-110;
    double ax=std::fabs(x);
    long q=std::lround(ax*INVPI);
    if(q<0 || q>=APPLE_COS53_REDN) return std::cos(x);
    double s0=ax-apple_cos53_pih[q],rh,rl; root_twos(s0,-apple_cos53_pil[q],&rh,&rl);
    bool rn=(rh<0.0)||(rh==0.0&&rl<0.0); if(rn){rh=-rh;rl=-rl;}
    double eh,el; root_twos(PIO2_HI,-rh,&eh,&el);
    el += PIO2_LO - rl + PIO2_TINY;
    double e2h,e2l; root_twos(eh,el,&e2h,&e2l); eh=e2h; el=e2l;
    double z=eh*eh;
    double sp=std::fma(-1.0/5040.0,z,1.0/120.0);
    sp=std::fma(sp,z,-1.0/6.0);
    sp=std::fma(sp,z,1.0);
    double sh=eh*sp;
    double cp=std::fma(1.0/24.0,z,-0.5);
    cp=std::fma(cp,z,1.0);
    double v=std::fma(el,cp,sh);
    if(((q&1)^(int)rn)) v=-v;
    return v;
}

'''
s=s.replace('class AppleTwoCoreHighway',helper+'class AppleTwoCoreHighway',1)
a=s.index('static int validate_mode()')
b=s.index('\n#endif',a)
new=r'''static int validate_mode()
{
    const int N=1000000;
    std::vector<double> x(N),o(N);
    const double lo[3]={0.0,1.0,1000.0},hi[3]={1.0,500.0,10000.0};
    for(int i=0;i<N;i++) { int c=i%6,b=c/2; double u=unit52(mix64(UINT64_C(2026090599)+(uint64_t)i*UINT64_C(0x9e3779b97f4a7c15))); double v=std::fma(hi[b]-lo[b],u,lo[b]); x[i]=(c&1)?-v:v; }
    cos53_eval_hwy(x.data(),o.data(),N);
    int repaired=0;
    for(int i=0;i<N;i++) if(std::fabs(o[i])<1.0e-3) { o[i]=cos53_rootfix_one(x[i]); repaired++; }
    mpfr_t z,r; mpfr_init2(z,256); mpfr_init2(r,256);
    uint64_t maxulp=0; int exact=0,le1=0,le2=0,bad=0,worst=-1; double wx=0,wo=0,wr=0;
    for(int i=0;i<N;i++) { mpfr_set_d(z,x[i],MPFR_RNDN); mpfr_cos(r,z,MPFR_RNDN); double ref=mpfr_get_d(r,MPFR_RNDN); uint64_t u=ulpd(o[i],ref); if(u==0)exact++; if(u<=1)le1++; if(u<=2)le2++; else {bad++; if(bad<=20) std::printf("ROOT_OUTLIER i=%d x=%.17g out=%.17g ref=%.17g ulp=%llu\n",i,x[i],o[i],ref,(unsigned long long)u);} if(u>maxulp){maxulp=u;worst=i;wx=x[i];wo=o[i];wr=ref;} }
    const double badx=-2492.8537705956187; double broot=cos53_rootfix_one(badx); mpfr_set_d(z,badx,MPFR_RNDN);mpfr_cos(r,z,MPFR_RNDN);double bref=mpfr_get_d(r,MPFR_RNDN);
    std::printf("ROOT_BADINPUT x=%.17g out=%.17g ref=%.17g ulp=%llu\n",badx,broot,bref,(unsigned long long)ulpd(broot,bref));
    std::printf("ROOT_1M cases=%d repaired=%d exact=%d le1=%d le2=%d bad=%d maxulp=%llu worst_i=%d worst_x=%.17g worst_out=%.17g worst_ref=%.17g\n",N,repaired,exact,le1,le2,bad,(unsigned long long)maxulp,worst,wx,wo,wr);
    mpfr_clear(r);mpfr_clear(z); return 0;
}
'''
s=s[:a]+new+s[b:]
Path('/tmp/rootfix.cpp').write_text(s)
PY
MP="$(brew --prefix mpfr)"; GP="$(brew --prefix gmp)"
clang++ -O3 -DNDEBUG -std=c++20 -mcpu=native -fno-fast-math -ffp-contract=off -fblocks -DAPPLE_COS53_VALIDATE_MPFR -I/tmp -I/tmp/highway -I"$MP/include" -I"$GP/include" /tmp/rootfix.cpp -L"$MP/lib" -L"$GP/lib" -lmpfr -lgmp -framework Accelerate -pthread -o /tmp/rootfix
/tmp/rootfix validate | tee /tmp/rootfix_1m.txt
