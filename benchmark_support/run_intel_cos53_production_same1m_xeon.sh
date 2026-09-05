#!/usr/bin/env bash
set -euo pipefail

PROD=6b8aaf64b84221f664c700dfd51fc3140218e0b8
ROOT=/tmp/intel-cos53-same1m
rm -rf "$ROOT" && mkdir -p "$ROOT"
cd "$ROOT"

for f in \
  benchmark_support/bench_sine_53_wide_fast2.c \
  benchmark_support/bench_sine_53_wide_intel.c \
  benchmark_support/sine_53_coeff_source.c \
  cosine53_apply_formula_conversion.py \
  cosine53_x50_unit_production.c \
  cosine53_x67_wide_production.c \
  cosine53_engine_adapter.c \
  cosine53_batch_production.hpp \
  cosine53_custom_2core_1600_frozen.hpp; do
  git -C "$GITHUB_WORKSPACE" show "$PROD:$f" > "$(basename "$f")"
done

cp bench_sine_53_wide_fast2.c fast2.c
cp bench_sine_53_wide_intel.c intel.c
cp sine_53_coeff_source.c coeff.c
sed -i 's/^int main(void){int cpu=pin();mkl_set_num_threads_local(1);printf("S53F2_DOMAIN/int s53f2_disabled_main(void){int cpu=pin();mkl_set_num_threads_local(1);printf("S53F2_DOMAIN/' fast2.c
sed -i 's/#define SF_K 12/#define SF_K 8/' coeff.c
sed -i 's/#define SF_LUT_N ((1UL << SF_K) + 1UL)/#define SF_LUT_N 403UL/' coeff.c
sed -i 's/#define KGRID 4096.0/#define KGRID 256.0/' intel.c
sed -i 's|#define INVK (1.0/4096.0)|#define INVK (1.0/256.0)|' intel.c
cp cosine53_x50_unit_production.c generated_x50.c
cp cosine53_x67_wide_production.c generated_x67.c
python3 cosine53_apply_formula_conversion.py \
  --coeff coeff.c --base intel.c --fast2 fast2.c \
  --source generated_x50.c --source generated_x67.c
python3 - <<'PY'
from pathlib import Path
for name in ('generated_x50.c','generated_x67.c'):
    p=Path(name); s=p.read_text(); old='\nint main(void)\n{'
    if s.count(old)!=1: raise SystemExit(f'{name}: main count={s.count(old)}')
    p.write_text(s.replace(old,'\nint cosine53_embedded_main(void)\n{',1))
PY

cat > validate.cpp <<'CPP'
#define _GNU_SOURCE
#include <mpfr.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "cosine53_batch_production.hpp"

extern "C" int cos53_engine_init(void);
extern "C" void cos53_engine_eval(double *, const double *, size_t);
extern "C" void cos53_engine_cleanup(void);

#ifndef PART_UNIT
#define PART_UNIT 0
#endif

static uint64_t mix64(uint64_t x) {
    x^=x>>30; x*=UINT64_C(0xbf58476d1ce4e5b9); x^=x>>27;
    x*=UINT64_C(0x94d049bb133111eb); x^=x>>31; return x;
}
static double unit52(uint64_t h) { return ((double)(h>>12)+0.5)*0x1p-52; }
static uint64_t ordered_bits(double x) {
    uint64_t u; std::memcpy(&u,&x,8);
    return (u>>63) ? ~u : (u|UINT64_C(0x8000000000000000));
}
static uint64_t ulpd(double a,double b) {
    if(a==b) return 0; uint64_t x=ordered_bits(a),y=ordered_bits(b); return x>y?x-y:y-x;
}
int main() {
    constexpr int N=1000000;
    const double lo[3]={0.0,1.0,1000.0}, hi[3]={1.0,500.0,10000.0};
    std::vector<double> x,o; std::vector<int> orig;
    x.reserve(PART_UNIT ? 333334 : 666666); orig.reserve(x.capacity());
    for(int i=0;i<N;i++) {
        int c=i%6; bool take=PART_UNIT ? (c<2) : (c>=2); if(!take) continue;
        int b=c/2;
        double u=unit52(mix64(UINT64_C(2026090599)+(uint64_t)i*UINT64_C(0x9e3779b97f4a7c15)));
        double v=std::fma(hi[b]-lo[b],u,lo[b]);
        x.push_back((c&1)?-v:v); orig.push_back(i);
    }
    o.resize(x.size());
    if(!cos53_engine_init()) return 3;
    {
        Cosine53BatchProductionFrozen prod(cos53_engine_eval);
        prod.run(o.data(),x.data(),x.size());
    }
    mpfr_t z,r; mpfr_init2(z,256); mpfr_init2(r,256);
    uint64_t maxulp=0; size_t exact=0,le1=0,le2=0,bad=0,worst=0; double wr=0;
    for(size_t k=0;k<x.size();k++) {
        mpfr_set_d(z,x[k],MPFR_RNDN); mpfr_cos(r,z,MPFR_RNDN); double ref=mpfr_get_d(r,MPFR_RNDN);
        uint64_t u=ulpd(o[k],ref);
        if(u==0) exact++; if(u<=1) le1++; if(u<=2) le2++;
        if(u>2 && bad<20) std::printf("INTEL_COS53_OUTLIER part=%s i=%d x=%.17g out=%.17g ref=%.17g ulp=%llu abs_err=%.17g\n", PART_UNIT?"unit":"wide",orig[k],x[k],o[k],ref,(unsigned long long)u,std::fabs(o[k]-ref));
        if(u>2) bad++;
        if(u>maxulp){maxulp=u;worst=k;wr=ref;}
    }
    std::printf("INTEL_COS53_PART part=%s cases=%zu exact=%zu le1=%zu le2=%zu bad=%zu maxulp=%llu worst_i=%d worst_x=%.17g worst_out=%.17g worst_ref=%.17g reference=MPFR256\n",
        PART_UNIT?"unit":"wide",x.size(),exact,le1,le2,bad,(unsigned long long)maxulp,orig[worst],x[worst],o[worst],wr);
    mpfr_clear(r); mpfr_clear(z); cos53_engine_cleanup(); return 0;
}
CPP

CC=/opt/intel/oneapi/compiler/latest/bin/icx
CXX=/opt/intel/oneapi/compiler/latest/bin/icpx
MKLROOT=/opt/intel/oneapi/mkl/latest
COMMON='-O3 -xHost -qopt-zmm-usage=high -fp-model=precise -fno-math-errno -DNDEBUG'
INC="-I$FLINT_PREFIX/include -I$MKLROOT/include -I."
LIBS="-L$FLINT_PREFIX/lib -Wl,-rpath,$FLINT_PREFIX/lib -lflint -lmpfr -lgmp -L$MKLROOT/lib -Wl,-rpath,$MKLROOT/lib -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl"

cp cosine53_engine_adapter.c unit_adapter.c
cp cosine53_engine_adapter.c wide_adapter.c
"$CC" $COMMON $INC -DCOSINE53_GENERATED_SOURCE='"generated_x50.c"' -c unit_adapter.c -o unit_adapter.o
"$CC" $COMMON $INC -DCOSINE53_GENERATED_SOURCE='"generated_x67.c"' -c wide_adapter.c -o wide_adapter.o
"$CXX" -std=c++17 $COMMON $INC -DPART_UNIT=1 validate.cpp unit_adapter.o -o validate_unit $LIBS
"$CXX" -std=c++17 $COMMON $INC -DPART_UNIT=0 validate.cpp wide_adapter.o -o validate_wide $LIBS

export LD_LIBRARY_PATH="$FLINT_PREFIX/lib:$MKLROOT/lib:${LD_LIBRARY_PATH:-}"
(taskset -c 0,2 ./validate_unit) | tee unit.txt
(taskset -c 0,2 ./validate_wide) | tee wide.txt
python3 - <<'PY' | tee /tmp/intel_cos53_same1m_result.txt
import re
parts=[]
pat=re.compile(r'INTEL_COS53_PART part=(\w+) cases=(\d+) exact=(\d+) le1=(\d+) le2=(\d+) bad=(\d+) maxulp=(\d+) worst_i=(\d+) worst_x=([^ ]+) worst_out=([^ ]+) worst_ref=([^ ]+)')
for fn in ('unit.txt','wide.txt'):
    text=open(fn).read(); m=pat.search(text)
    if not m: raise SystemExit(f'missing result in {fn}')
    parts.append(dict(part=m[1],cases=int(m[2]),exact=int(m[3]),le1=int(m[4]),le2=int(m[5]),bad=int(m[6]),maxulp=int(m[7]),i=int(m[8]),x=m[9],out=m[10],ref=m[11]))
for p in parts: print('PART',p)
worst=max(parts,key=lambda p:p['maxulp'])
print(f"INTEL_COS53_PRODUCTION_SAME1M cases={sum(p['cases'] for p in parts)} exact={sum(p['exact'] for p in parts)} le1={sum(p['le1'] for p in parts)} le2={sum(p['le2'] for p in parts)} bad={sum(p['bad'] for p in parts)} maxulp={worst['maxulp']} worst_part={worst['part']} worst_i={worst['i']} worst_x={worst['x']} worst_out={worst['out']} worst_ref={worst['ref']} reference=MPFR256 prod_commit=6b8aaf64b84221f664c700dfd51fc3140218e0b8")
PY
