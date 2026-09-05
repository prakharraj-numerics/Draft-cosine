#!/usr/bin/env bash
set -euo pipefail

[[ "$(uname -m)" == arm64 ]]
[[ "$(sysctl -n machdep.cpu.brand_string)" == *"Apple M1"* ]]

MODE="${1:-}"
BRANCH_ROOT="$(git rev-parse --show-toplevel)"
K=1280
LUTN=2012

if [[ "$MODE" == "build" ]]; then
  brew list flint >/dev/null 2>&1 || brew install flint
  brew list mpfr >/dev/null 2>&1 || brew install mpfr
  brew list gmp >/dev/null 2>&1 || brew install gmp
  brew list halide >/dev/null 2>&1 || brew install halide

  rm -rf /tmp/highway /tmp/pthreadpool /tmp/pthreadpool-build /tmp/pthreadpool-install
  git clone --depth 1 --branch 1.4.0 https://github.com/google/highway.git /tmp/highway
  git clone --depth 1 https://github.com/Maratyszcza/pthreadpool.git /tmp/pthreadpool
  cmake -S /tmp/pthreadpool -B /tmp/pthreadpool-build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/tmp/pthreadpool-install \
    -DPTHREADPOOL_LIBRARY_TYPE=static \
    -DPTHREADPOOL_SYNC_PRIMITIVE=condvar \
    -DPTHREADPOOL_BUILD_TESTS=OFF \
    -DPTHREADPOOL_BUILD_BENCHMARKS=OFF
  cmake --build /tmp/pthreadpool-build --target pthreadpool -j 4
  cmake --install /tmp/pthreadpool-build

  cp benchmark_support/apple_cos53_highway_benchmark.cpp /tmp/base.cpp
  cp benchmark_support/sine_53_coeff_source.c /tmp/src.c
  cp benchmark_support/apple_cos53_coeff_bridge.c /tmp/bridge.c
  cp benchmark_support/apple_cos53_generate_constants.c /tmp/gen.c
  cp cosine53_apply_formula_conversion.py /tmp/cosine53_apply_formula_conversion.py

  python3 - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0,'/tmp')
K=1280; LUTN=2012
p=Path('/tmp/src.c'); s=p.read_text()
s=s.replace('#define SF_K 12',f'#define SF_GRID {K}')
s=s.replace('#define SF_LUT_N ((1UL << SF_K) + 1UL)',f'#define SF_LUT_N {LUTN}UL')
old='st |= nfloat_mul_2exp_si(delta, delta, -SF_K, b->nctx);'
assert old in s
s=s.replace(old,f'st |= nfloat_div_ui(delta, delta, {K}UL, b->nctx);')
p.write_text(s)
p=Path('/tmp/bridge.c'); s=p.read_text()
s=s.replace('#include "apple_cosine53_coeff_source.c"','#include "src.c"')
s=s.replace('s53_coeff_create_terms(2)','s53_coeff_create_terms(1)').replace('c->poly_deg != 5','c->poly_deg != 3')
p.write_text(s)
p=Path('/tmp/gen.c'); p.write_text(p.read_text().replace('#define LUTN 403',f'#define LUTN {LUTN}'))
import cosine53_apply_formula_conversion as m
m.patch_coeff(Path('/tmp/src.c'))
PY

  FP="$(brew --prefix flint)"; MP="$(brew --prefix mpfr)"; GP="$(brew --prefix gmp)"
  clang -O2 -DNDEBUG -I/tmp -I"$FP/include" -I"$MP/include" -I"$GP/include" \
    /tmp/bridge.c /tmp/gen.c -L"$FP/lib" -L"$MP/lib" -L"$GP/lib" \
    -lflint -lmpfr -lgmp -lm -o /tmp/gen
  /tmp/gen /tmp/apple_cos53_constants_k1280_fastreduce.h

  python3 - <<'PY'
from pathlib import Path
from fractions import Fraction
import re
K=1280
h=Fraction(1,2*K); h2=h*h; h4=h2*h2
A0=Fraction(1)-h4/Fraction(192)
B1=Fraction(1)-h4/Fraction(384)
A2=-Fraction(1,2)+h2/Fraction(24)
B3=-Fraction(1,6)+h2/Fraction(96)
ar=A2/A0; br=B3/B1
p=Path('/tmp/apple_cos53_constants_k1280_fastreduce.h'); s=p.read_text()
def scale(text,name,factor):
    pat=rf'(static const double {name}\[APPLE_COS53_LUTN\] = \{{\s*)(.*?)(\}};)'
    m=re.search(pat,text,re.S); assert m,name
    body=m.group(2)
    def rr(mm):
        x=float.fromhex(mm.group(0)); return float(Fraction.from_float(x)*factor).hex()
    body=re.sub(r'-?0x[0-9a-fA-F]+(?:\.[0-9a-fA-F]*)?p[+-]?\d+',rr,body)
    return text[:m.start(2)]+body+text[m.end(2):]
s=scale(s,'apple_cos53_c0',A0)
s=scale(s,'apple_cos53_c1',B1)
p.write_text(s)

p=Path('/tmp/base.cpp'); s=p.read_text()
s=s.replace('#include "apple_cos53_constants.h"','#include "apple_cos53_constants_k1280_fastreduce.h"')
s=s.replace('static constexpr double KGRID = 256.0;','static constexpr double KGRID = 1280.0;')
s=s.replace('static constexpr double INVK = 1.0 / 256.0;','static constexpr double INVK = 1.0 / 1280.0;')
s=s.replace('    const auto mh = hn::Set(d, -0.5);',f'    const auto mh = hn::Set(d, {float(ar).hex()});')
s=s.replace('    const auto m6 = hn::Set(d, -1.0/6.0);',f'    const auto m6 = hn::Set(d, {float(br).hex()});')
s=s.replace('    const auto c24v = hn::Set(d, 1.0/24.0);\n    const auto c120v = hn::Set(d, 1.0/120.0);\n','')
needle='    const auto vik = hn::Set(d, INVK);'
assert needle in s
s=s.replace(needle,needle+'\n    const auto nvik = hn::Set(d, -0x1.999999999999ap-11);\n    const auto nviklo = hn::Set(d, 0x1.999999999999ap-65);',1)

oldred='''        auto s = hn::Sub(ax, ph);\n        auto b = hn::Neg(pl);\n\n        // Exact TwoSum, matching the current Apple NEON path operation-for-operation.\n        auto rh = hn::Add(s, b);\n        auto bv = hn::Sub(rh, s);\n        auto av = hn::Sub(rh, bv);\n        auto br = hn::Sub(b, bv);\n        auto ar = hn::Sub(s, av);\n        auto rl = hn::Add(ar, br);\n\n        auto rs = hn::Add(rh, rl);\n        auto neg = hn::Lt(rs, zero);\n        rh = hn::IfThenElse(neg, hn::Neg(rh), rh);\n        rl = hn::IfThenElse(neg, hn::Neg(rl), rl);\n        auto r = hn::Add(rh, rl);\n\n        auto ji = hn::NearestInt(hn::Mul(r, vk));'''
newred='''        // K=1280 fast reduction candidate.  The first subtraction is exact in\n        // the normal q*pi reduction region (Sterbenz); recover the low part\n        // with the shorter FastTwoSum form rather than generic TwoSum.\n        auto s = hn::Sub(ax, ph);\n        auto rh = hn::Sub(s, pl);\n        auto z = hn::Sub(s, rh);\n        auto rl = hn::Sub(z, pl);\n\n        // cos is even in the reduced residual.  Use the high component for\n        // sign and grid selection, and carry the correspondingly signed low\n        // component only into delta.  This removes r=(rh+rl) from the index path.\n        auto neg = hn::Lt(rh, zero);\n        auto ah = hn::Abs(rh);\n        auto al = hn::IfThenElse(neg, hn::Neg(rl), rl);\n        auto ji = hn::NearestInt(hn::Mul(ah, vk));'''
assert oldred in s
s=s.replace(oldred,newred,1)

olddelta='auto delta = hn::Add(hn::Sub(rh, hn::Mul(jd, vik)), rl);'
newdelta='''auto delta = hn::MulAdd(jd, nvik, ah);\n        delta = hn::MulAdd(jd, nviklo, delta);\n        delta = hn::Add(delta, al);'''
assert olddelta in s
s=s.replace(olddelta,newdelta,1)

start=s.index('        auto c2 = hn::Mul(c0, mh);')
end=s.index('        uint64_t sm0=',start)
s=s[:start]+'''        auto c2 = hn::Mul(c0, mh);\n        auto c3 = hn::Mul(c1, m6);\n        auto p = hn::MulAdd(c3, delta, c2);\n        p = hn::MulAdd(p, delta, c1);\n        p = hn::MulAdd(p, delta, c0);\n\n'''+s[end:]

# Pull the already-frozen production router into the candidate after the helper class exists.
needle='};\n\nstatic uint64_t mix64(uint64_t x)'
assert needle in s
s=s.replace(needle,'};\n\n#include "apple_cos53_production_routing.hpp"\n\nstatic uint64_t mix64(uint64_t x)',1)

needle='''static void run_hwy2(AppleTwoCoreHighway &tc,Buffers &b)\n{\n    for(int c=0;c<6;c++) tc.run(b.x[c],b.y[c],b.n);\n}\n'''
assert needle in s
s=s.replace(needle,needle+'''static void run_candidate(AppleTwoCoreHighway &tc,Buffers &b)\n{\n    for(int c=0;c<6;c++) apple_cos53_production::run(tc,b.x[c],b.y[c],b.n);\n}\n''',1)

old='''    AppleTwoCoreHighway *tc = stack=="hwy2" ? new AppleTwoCoreHighway() : nullptr;\n    auto once=[&]{ if(stack=="apple")run_apple(b); else run_hwy2(*tc,b); };'''
new='''    AppleTwoCoreHighway *tc = stack=="candidate" ? new AppleTwoCoreHighway() : nullptr;\n    auto once=[&]{ if(stack=="apple") run_apple(b); else run_candidate(*tc,b); };'''
assert old in s
s=s.replace(old,new,1)
old='if(stack!="hwy2" && stack!="apple") return 3;'
assert old in s
s=s.replace(old,'if(stack!="candidate" && stack!="apple") return 3;',1)
Path('/tmp/candidate_established.cpp').write_text(s)
Path('/tmp/candidate_bench.cpp').write_text(s)

# Stronger stress diagnostic: 980k random points + 20k values very near cosine zeros.
a=s.index('static int validate_mode()')
b=s.index('\n#endif',a)
stress=r'''static int validate_mode()\n{\n    const int N=1000000;\n    const int RANDOM_N=980000;\n    std::vector<double> x(N),o(N);\n    const double lo[3]={0.0,1.0,1000.0},hi[3]={1.0,500.0,10000.0};\n    for(int i=0;i<RANDOM_N;i++) {\n        int c=i%6,b=c/2;\n        double u=unit52(mix64(UINT64_C(2026090617)+(uint64_t)i*UINT64_C(0x9e3779b97f4a7c15)));\n        double v=std::fma(hi[b]-lo[b],u,lo[b]); x[i]=(c&1)?-v:v;\n    }\n    const long double PIL=acosl(-1.0L);\n    for(int i=RANDOM_N;i<N;i++) {\n        uint64_t h=mix64(UINT64_C(0xd2b74407b1ce6e93)+(uint64_t)i*UINT64_C(0x9e3779b97f4a7c15));\n        int k=(int)(h%6366ULL)-3183;\n        long double root=0.5L*PIL+(long double)k*PIL;\n        double v=(double)root;\n        int steps=(int)((h>>16)%33ULL);\n        double dir=((h>>24)&1ULL)?INFINITY:-INFINITY;\n        for(int j=0;j<steps;j++) v=std::nextafter(v,dir);\n        x[i]=v;\n    }\n    cos53_eval_hwy(x.data(),o.data(),N);\n    mpfr_t z,r; mpfr_init2(z,256); mpfr_init2(r,256);\n    uint64_t maxulp=0,max_random=0,max_danger=0;\n    int exact=0,le1=0,le2=0,le3=0,gt3=0,worst=-1;\n    double wx=0,wo=0,wr=0;\n    for(int i=0;i<N;i++) {\n        mpfr_set_d(z,x[i],MPFR_RNDN); mpfr_cos(r,z,MPFR_RNDN);\n        double ref=mpfr_get_d(r,MPFR_RNDN); uint64_t u=ulpd(o[i],ref);\n        if(u==0) exact++; if(u<=1) le1++; if(u<=2) le2++; if(u<=3) le3++; else gt3++;\n        if(i<RANDOM_N) max_random=std::max(max_random,u); else max_danger=std::max(max_danger,u);\n        if(u>maxulp){maxulp=u;worst=i;wx=x[i];wo=o[i];wr=ref;}\n    }\n    std::printf("APPLE_COS53_K1280_FASTREDUCE_STRESS cases=%d random=%d danger=%d exact=%d le1=%d le2=%d le3=%d gt3=%d maxulp=%llu max_random=%llu max_danger=%llu worst_i=%d worst_kind=%s worst_x=%.17g worst_out=%.17g worst_ref=%.17g\\n",\n      N,RANDOM_N,N-RANDOM_N,exact,le1,le2,le3,gt3,(unsigned long long)maxulp,(unsigned long long)max_random,(unsigned long long)max_danger,worst,(worst<RANDOM_N?"random":"danger"),wx,wo,wr);\n    mpfr_clear(r); mpfr_clear(z); return 0;\n}\n'''
st=s[:a]+stress+s[b:]
Path('/tmp/candidate_stress.cpp').write_text(st)
print('K1280_RETUNE', 'A0',float(A0).hex(),'B1',float(B1).hex(),'A2/A0',float(ar).hex(),'B3/B1',float(br).hex())
PY

  HP="$(brew --prefix halide)"
  cat >/tmp/genrt.cpp <<'CPP'
#include <Halide.h>
int main() {
    Halide::compile_standalone_runtime("/tmp/halide_runtime.a", Halide::get_host_target());
    return 0;
}
CPP
  clang++ -O2 -std=c++20 -I"$HP/include" /tmp/genrt.cpp -L"$HP/lib" -lHalide -o /tmp/genrt
  /tmp/genrt

  COMMON='-O3 -DNDEBUG -std=c++20 -mcpu=native -fno-fast-math -ffp-contract=off -fblocks'
  INCS="-I/tmp -I/tmp/highway -I/tmp/pthreadpool-install/include -I$HP/include -I$BRANCH_ROOT"
  LIBS="/tmp/pthreadpool-install/lib/libpthreadpool.a /tmp/halide_runtime.a -framework Accelerate -pthread -ldl"
  clang++ $COMMON $INCS /tmp/candidate_bench.cpp $LIBS -o /tmp/apple_cos53_k1280_fastreduce_bench
  clang++ $COMMON -DAPPLE_COS53_VALIDATE_MPFR $INCS -I"$MP/include" -I"$GP/include" \
    /tmp/candidate_established.cpp $LIBS -L"$MP/lib" -L"$GP/lib" -lmpfr -lgmp \
    -o /tmp/apple_cos53_k1280_fastreduce_established
  clang++ $COMMON -DAPPLE_COS53_VALIDATE_MPFR $INCS -I"$MP/include" -I"$GP/include" \
    /tmp/candidate_stress.cpp $LIBS -L"$MP/lib" -L"$GP/lib" -lmpfr -lgmp \
    -o /tmp/apple_cos53_k1280_fastreduce_stress
  exit 0
fi

if [[ "$MODE" == "validate" ]]; then
  echo "=== established 9600-case MPFR256 validator ==="
  /tmp/apple_cos53_k1280_fastreduce_established validate
  echo "=== stressed 1M MPFR256 diagnostic ==="
  /tmp/apple_cos53_k1280_fastreduce_stress validate
  exit 0
fi

if [[ "$MODE" == "one" ]]; then
  [[ $# -eq 3 ]]
  exec /tmp/apple_cos53_k1280_fastreduce_bench "$2" "$3"
fi

echo "usage: $0 build | validate | one {candidate|apple} N" >&2
exit 2
