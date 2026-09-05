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
cp benchmark_support/apple_cos53_custom2_core.hpp /tmp/apple_cos53_custom2_core.hpp

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
clang -O2 -DNDEBUG -I/tmp -I"$FP/include" -I"$MP/include" -I"$GP/include" \
  /tmp/bridge.c /tmp/gen.c -L"$FP/lib" -L"$MP/lib" -L"$GP/lib" \
  -lflint -lmpfr -lgmp -lm -o /tmp/gen
/tmp/gen /tmp/apple_cos53_constants_custom2.h

python3 - <<'PY'
from pathlib import Path
p=Path('/tmp/base.cpp'); s=p.read_text()
s=s.replace('#include "apple_cos53_constants.h"','#include "apple_cos53_constants_custom2.h"\n#include "apple_cos53_custom2_core.hpp"')
s=s.replace('static constexpr double KGRID = 256.0;','static constexpr double KGRID = 2048.0;').replace('static constexpr double INVK = 1.0 / 256.0;','static constexpr double INVK = 1.0 / 2048.0;')
s=s.replace('    const auto c24v = hn::Set(d, 1.0/24.0);\n    const auto c120v = hn::Set(d, 1.0/120.0);\n','')
start=s.index('        auto c2 = hn::Mul(c0, mh);')
end=s.index('        uint64_t sm0=',start)
s=s[:start]+'''        auto c2 = hn::Mul(c0, mh);\n        auto c3 = hn::Mul(c1, m6);\n        auto p = hn::MulAdd(c3, delta, c2);\n        p = hn::MulAdd(p, delta, c1);\n        p = hn::MulAdd(p, delta, c0);\n\n'''+s[end:]

old='''static void run_hwy2(AppleTwoCoreHighway &tc,Buffers &b)\n{\n    for(int c=0;c<6;c++) tc.run(b.x[c],b.y[c],b.n);\n}\n'''
new=old+'''static void run_custom2(AppleCos53CustomPermanent2Core &tc,Buffers &b)\n{\n    for(int c=0;c<6;c++) tc.run(b.x[c],b.y[c],b.n);\n}\n'''
assert old in s; s=s.replace(old,new,1)

old='''    AppleTwoCoreHighway *tc = stack=="hwy2" ? new AppleTwoCoreHighway() : nullptr;\n    auto once=[&]{ if(stack=="apple")run_apple(b); else run_hwy2(*tc,b); };'''
new='''    AppleTwoCoreHighway *tc = stack=="baseline" ? new AppleTwoCoreHighway() : nullptr;\n    AppleCos53CustomPermanent2Core *c2 = stack=="custom2" ? new AppleCos53CustomPermanent2Core(cos53_eval_hwy) : nullptr;\n    auto once=[&]{ if(stack=="baseline")run_hwy2(*tc,b); else run_custom2(*c2,b); };'''
assert old in s; s=s.replace(old,new,1)
s=s.replace('    delete tc;\n    return 0;','    delete c2;\n    delete tc;\n    return 0;',1)

# Add an explicit bitwise-equivalence verifier across the routing range.
insert=r'''
static int verify_custom2_scheduler()
{
    static const size_t sizes[] = {64,100,400,1200,3000,5000,7500,10000,15000,20000,25000,29999};
    AppleTwoCoreHighway baseline;
    AppleCos53CustomPermanent2Core custom2(cos53_eval_hwy);
    size_t total_diff=0;
    for(size_t n: sizes) {
        Buffers b(n);
        std::vector<double*> ref(6), got(6);
        for(int c=0;c<6;c++) {
            posix_memalign((void**)&ref[c],64,n*sizeof(double));
            posix_memalign((void**)&got[c],64,n*sizeof(double));
            baseline.run(b.x[c],ref[c],n);
            custom2.run(b.x[c],got[c],n);
            size_t diff=0;
            for(size_t i=0;i<n;i++) diff += std::memcmp(ref[c]+i,got[c]+i,sizeof(double))!=0;
            total_diff += diff;
            std::printf("APPLE_CUSTOM2_VERIFY n=%zu case=%d bitdiff=%zu\n",n,c,diff);
            free(ref[c]); free(got[c]);
        }
    }
    std::printf("APPLE_CUSTOM2_VERIFY_DONE total_bitdiff=%zu\n",total_diff);
    return total_diff?21:0;
}

'''
marker='#ifdef APPLE_COS53_VALIDATE_MPFR\nstatic int validate_mode()'
s=s.replace(marker,insert+marker,1)

old='''    if(argc!=3) return 2;\n    std::string stack=argv[1]; size_t n=(size_t)std::strtoull(argv[2],nullptr,10);\n    if(stack!="hwy2" && stack!="apple") return 3;\n    return bench_mode(stack,n);'''
new='''    if(argc==2 && std::string(argv[1])=="verify-custom2") return verify_custom2_scheduler();\n    if(argc!=3) return 2;\n    std::string stack=argv[1]; size_t n=(size_t)std::strtoull(argv[2],nullptr,10);\n    if(stack!="baseline" && stack!="custom2") return 3;\n    return bench_mode(stack,n);'''
assert old in s; s=s.replace(old,new,1)
p.write_text(s)
PY

clang++ -O3 -DNDEBUG -std=c++20 -mcpu=native -fno-fast-math -ffp-contract=off -fblocks \
  -I/tmp -I/tmp/highway /tmp/base.cpp -framework Accelerate -pthread -o /tmp/apple_custom2_compare

if [[ "${2:-}" == "verify" ]]; then
  /tmp/apple_custom2_compare verify-custom2
  exit $?
fi

MODE="${1:?usage: $0 baseline|custom2 [verify]}"
shift || true
exec /tmp/apple_custom2_compare "$MODE" "$@"
