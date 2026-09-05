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

python3 - <<'PY'
from pathlib import Path
s=Path('/tmp/base.cpp').read_text()
s=s.replace('#include "apple_cos53_constants.h"','#include "apple_cos53_constants_2ulp.h"')
s=s.replace('static constexpr double KGRID = 256.0;','static constexpr double KGRID = 2048.0;').replace('static constexpr double INVK = 1.0 / 256.0;','static constexpr double INVK = 1.0 / 2048.0;')
s=s.replace('    const auto c24v = hn::Set(d, 1.0/24.0);\n    const auto c120v = hn::Set(d, 1.0/120.0);\n','')
start=s.index('        auto c2 = hn::Mul(c0, mh);')
end=s.index('        uint64_t sm0=', start)
replacement='''        auto c2 = hn::Mul(c0, mh);\n        auto c3 = hn::Mul(c1, m6);\n\n#ifdef COMPENSATED_LOWWORD\n        // Intel scalar2-style compensated Horner:\n        // evaluate at d = rh - anchor, track derivative dp, then inject rl once.\n        auto dh = hn::Sub(rh, hn::Mul(jd, vik));\n        auto p = c3;\n        auto dp = zero;\n        dp = hn::MulAdd(dp, dh, p); p = hn::MulAdd(p, dh, c2);\n        dp = hn::MulAdd(dp, dh, p); p = hn::MulAdd(p, dh, c1);\n        dp = hn::MulAdd(dp, dh, p); p = hn::MulAdd(p, dh, c0);\n        p = hn::MulAdd(rl, dp, p);\n#else\n        auto p = hn::MulAdd(c3, delta, c2);\n        p = hn::MulAdd(p, delta, c1);\n        p = hn::MulAdd(p, delta, c0);\n#endif\n\n'''
s=s[:start]+replacement+s[end:]
# replace validator with exactly one bad input and report both output + MPFR ULP
marker='#ifdef APPLE_COS53_VALIDATE_MPFR\nstatic int validate_mode()'
a=s.index(marker)
b=s.index('\n#endif', a)
new='''#ifdef APPLE_COS53_VALIDATE_MPFR\nstatic int validate_mode()\n{\n    const double x = -2492.8537705956187;\n    double o=0.0; cos53_eval_hwy(&x,&o,1);\n    mpfr_t z,r; mpfr_init2(z,256); mpfr_init2(r,256);\n    mpfr_set_d(z,x,MPFR_RNDN); mpfr_cos(r,z,MPFR_RNDN);\n    double ref=mpfr_get_d(r,MPFR_RNDN);\n    uint64_t u=ulpd(o,ref);\n    std::printf("BADINPUT mode=%s x=%.17g out=%.17g ref=%.17g ulp=%llu abs_err=%.17g\\n",\n#ifdef COMPENSATED_LOWWORD\n      "compensated_lowword",\n#else\n      "baseline",\n#endif\n      x,o,ref,(unsigned long long)u,std::fabs(o-ref));\n    mpfr_clear(r); mpfr_clear(z); return 0;\n}\n#endif'''
s=s[:a]+new+s[b+len('\n#endif'):]
Path('/tmp/test.cpp').write_text(s)
PY

MP="$(brew --prefix mpfr)"; GP="$(brew --prefix gmp)"
COMMON=(-O3 -DNDEBUG -std=c++20 -mcpu=native -fno-fast-math -ffp-contract=off -fblocks -DAPPLE_COS53_VALIDATE_MPFR -I/tmp -I/tmp/highway -I"$MP/include" -I"$GP/include" /tmp/test.cpp -L"$MP/lib" -L"$GP/lib" -lmpfr -lgmp -framework Accelerate -pthread)
clang++ "${COMMON[@]}" -o /tmp/baseline
clang++ -DCOMPENSATED_LOWWORD "${COMMON[@]}" -o /tmp/comp
{
  /tmp/baseline validate
  /tmp/comp validate
} | tee /tmp/badinput_compensated.txt
