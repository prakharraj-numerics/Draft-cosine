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
p=Path('/tmp/gen.c')
p.write_text(p.read_text().replace('#define LUTN 403','#define LUTN 3218'))
import cosine53_apply_formula_conversion as m
m.patch_coeff(Path('/tmp/src.c'))
PY
FP="$(brew --prefix flint)"; MP="$(brew --prefix mpfr)"; GP="$(brew --prefix gmp)"
clang -O2 -DNDEBUG -I/tmp -I$FP/include -I$MP/include -I$GP/include /tmp/bridge.c /tmp/gen.c -L$FP/lib -L$MP/lib -L$GP/lib -lflint -lmpfr -lgmp -lm -o /tmp/gen
/tmp/gen /tmp/apple_cos53_constants_2ulp.h

cp /tmp/base.cpp /tmp/diag.cpp
python3 - <<'PY'
from pathlib import Path
p=Path('/tmp/diag.cpp'); s=p.read_text()
s=s.replace('#include "apple_cos53_constants.h"','#include "apple_cos53_constants_2ulp.h"')
s=s.replace('static constexpr double KGRID = 256.0;','static constexpr double KGRID = 2048.0;').replace('static constexpr double INVK = 1.0 / 256.0;','static constexpr double INVK = 1.0 / 2048.0;')
s=s.replace('    const auto c24v = hn::Set(d, 1.0/24.0);\n    const auto c120v = hn::Set(d, 1.0/120.0);\n','')
start=s.index('        auto c2 = hn::Mul(c0, mh);'); end=s.index('        uint64_t sm0=', start)
s=s[:start]+'''        auto c2 = hn::Mul(c0, mh);\n        auto c3 = hn::Mul(c1, m6);\n        auto p = hn::MulAdd(c3, delta, c2);\n        p = hn::MulAdd(p, delta, c1);\n        p = hn::MulAdd(p, delta, c0);\n\n'''+s[end:]
# Add a scalar near-zero repair after the vector store, using the SAME q*pi hi/lo reduction,
# then sin(pi/2-r) with split pi/2. This tests local-cancellation diagnosis without changing normal lanes.
needle='''        hn::StoreU(p, d, y+i);\n'''
repl='''        hn::StoreU(p, d, y+i);\n#ifdef COS53_NEARZERO_REPAIR\n        for (int lane=0; lane<2; ++lane) {\n            const size_t ii=i+(size_t)lane;\n            if (std::fabs(y[ii]) < 1.0e-3) {\n                const int qq = lane==0 ? q0 : q1;\n                const double axx = std::fabs(x[ii]);\n                const double ss = axx - apple_cos53_pih[qq];\n                const double bb = -apple_cos53_pil[qq];\n                const double rrh = ss + bb;\n                const double bvv = rrh - ss;\n                const double avv = rrh - bvv;\n                const double rrl = (ss - avv) + (bb - bvv);\n                const double PIO2_H = 0x1.921fb54442d18p+0;\n                const double PIO2_L = 0x1.1a62633145c07p-54;\n                double t = (PIO2_H - rrh);\n                t += (PIO2_L - rrl);\n                double z=t*t;\n                double sp = std::fma(std::fma(1.0/120.0,z,-1.0/6.0)*z,t,t);\n                if ((qq & 1)==0) sp = -sp;\n                y[ii]=sp;\n            }\n        }\n#endif\n'''
assert needle in s
s=s.replace(needle,repl,1)
# Replace validator with exact 1M + named bad-case reporting.
a=s.index('static int validate_mode()'); b=s.index('\n#endif',a)
new=r'''static int validate_mode()
{
    const int N=1000000;
    std::vector<double> x(N),o(N);
    const double lo[3]={0.0,1.0,1000.0},hi[3]={1.0,500.0,10000.0};
    for(int i=0;i<N;i++) {
        int c=i%6,b=c/2;
        double u=unit52(mix64(UINT64_C(2026090599)+(uint64_t)i*UINT64_C(0x9e3779b97f4a7c15)));
        double v=std::fma(hi[b]-lo[b],u,lo[b]); x[i]=(c&1)?-v:v;
    }
    cos53_eval_hwy(x.data(),o.data(),N);
    mpfr_t z,r; mpfr_init2(z,256); mpfr_init2(r,256);
    uint64_t maxulp=0; int le2=0,bad=0;
    for(int i=0;i<N;i++) {
        mpfr_set_d(z,x[i],MPFR_RNDN); mpfr_cos(r,z,MPFR_RNDN);
        double ref=mpfr_get_d(r,MPFR_RNDN); uint64_t u=ulpd(o[i],ref);
        if(u<=2) le2++; else { bad++; if (bad<=20) std::printf("OUTLIER i=%d x=%.17g out=%.17g ref=%.17g ulp=%llu abs_err=%.17g\n",i,x[i],o[i],ref,(unsigned long long)u,std::fabs(o[i]-ref)); }
        if(u>maxulp) maxulp=u;
    }
    mpfr_clear(r); mpfr_clear(z);
    std::printf("DIAG cases=%d le2=%d bad=%d maxulp=%llu mode=%s\n",N,le2,bad,(unsigned long long)maxulp,
#ifdef COS53_NEARZERO_REPAIR
      "nearzero_repair"
#else
      "baseline"
#endif
    );
    return 0;
}
'''
s=s[:a]+new+s[b:]
p.write_text(s)
PY

MP="$(brew --prefix mpfr)"; GP="$(brew --prefix gmp)"
COMMON=(-O3 -DNDEBUG -std=c++20 -mcpu=native -fno-fast-math -ffp-contract=off -fblocks -DAPPLE_COS53_VALIDATE_MPFR -I/tmp -I/tmp/highway -I"$MP/include" -I"$GP/include" -L"$MP/lib" -L"$GP/lib" -lmpfr -lgmp -framework Accelerate -pthread)
clang++ "${COMMON[@]}" /tmp/diag.cpp -o /tmp/diag_base
clang++ "${COMMON[@]}" -DCOS53_NEARZERO_REPAIR /tmp/diag.cpp -o /tmp/diag_repair
{
  echo '=== BASELINE ==='
  /tmp/diag_base validate
  echo '=== NEARZERO_REPAIR ==='
  /tmp/diag_repair validate
} | tee /tmp/nearzero_diag.txt
