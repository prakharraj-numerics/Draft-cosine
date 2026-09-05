#!/usr/bin/env bash
set -euo pipefail

[[ "$(uname -m)" == arm64 ]]
[[ "$(sysctl -n machdep.cpu.brand_string)" == *"Apple M1"* ]]

brew list flint >/dev/null 2>&1 || brew install flint
brew list mpfr >/dev/null 2>&1 || brew install mpfr
brew list gmp >/dev/null 2>&1 || brew install gmp

rm -rf /tmp/highway /tmp/highway-build /tmp/fastflow
git clone --depth 1 --branch 1.4.0 https://github.com/google/highway.git /tmp/highway
git clone https://github.com/fastflow/fastflow.git /tmp/fastflow
git -C /tmp/fastflow checkout d476f66ab924d8d122f54b4b90aee00ef979aea8

cmake -S /tmp/highway -B /tmp/highway-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DHWY_ENABLE_TESTS=OFF \
  -DHWY_ENABLE_EXAMPLES=OFF \
  -DHWY_ENABLE_CONTRIB=ON >/tmp/highway-cmake.log
cmake --build /tmp/highway-build --target hwy_contrib -j2 >>/tmp/highway-cmake.log

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
clang -O2 -DNDEBUG -I/tmp -I"$FP/include" -I"$MP/include" -I"$GP/include" \
  /tmp/bridge.c /tmp/gen.c -L"$FP/lib" -L"$MP/lib" -L"$GP/lib" \
  -lflint -lmpfr -lgmp -lm -o /tmp/gen
/tmp/gen /tmp/apple_cos53_constants_3way.h

python3 - <<'PY'
from pathlib import Path
p=Path('/tmp/base.cpp'); s=p.read_text()
s=s.replace('#include <vector>','#include <vector>\n#include <hwy/contrib/thread_pool/thread_pool.h>\n#include <ff/parallel_for.hpp>')
s=s.replace('#include "apple_cos53_constants.h"','#include "apple_cos53_constants_3way.h"')
s=s.replace('static constexpr double KGRID = 256.0;','static constexpr double KGRID = 2048.0;').replace('static constexpr double INVK = 1.0 / 256.0;','static constexpr double INVK = 1.0 / 2048.0;')
s=s.replace('    const auto c24v = hn::Set(d, 1.0/24.0);\n    const auto c120v = hn::Set(d, 1.0/120.0);\n','')
start=s.index('        auto c2 = hn::Mul(c0, mh);')
end=s.index('        uint64_t sm0=',start)
s=s[:start]+'''        auto c2 = hn::Mul(c0, mh);\n        auto c3 = hn::Mul(c1, m6);\n        auto p = hn::MulAdd(c3, delta, c2);\n        p = hn::MulAdd(p, delta, c1);\n        p = hn::MulAdd(p, delta, c0);\n\n'''+s[end:]

needle='''};\n\nstatic uint64_t mix64(uint64_t x)'''
insert=r'''};

class AppleHighwayThreadPool2 {
    hwy::ThreadPool pool_;
public:
    AppleHighwayThreadPool2(): pool_(1) { pool_.SetWaitMode(hwy::PoolWaitMode::kSpin); }
    void run(const double *x,double *y,size_t n) {
        if (!n) return;
        if (n < 4) { cos53_eval_hwy(x,y,n); return; }
        const size_t mid=(n/2)&~(size_t)1;
        pool_.Run(0,2,[&](uint64_t task,size_t) {
            if (task==0) cos53_eval_hwy(x,y,mid);
            else cos53_eval_hwy(x+mid,y+mid,n-mid);
        });
    }
};

class AppleFastFlow2 {
    ff::ParallelFor pf_;
public:
    AppleFastFlow2(): pf_(2,true,true) { pf_.disableScheduler(true); }
    void run(const double *x,double *y,size_t n) {
        if (!n) return;
        if (n < 4) { cos53_eval_hwy(x,y,n); return; }
        const size_t mid=(n/2)&~(size_t)1;
        pf_.parallel_for(0L,2L,[&](const long task) {
            if (task==0) cos53_eval_hwy(x,y,mid);
            else cos53_eval_hwy(x+mid,y+mid,n-mid);
        },2);
    }
};

static uint64_t mix64(uint64_t x)'''
assert needle in s
s=s.replace(needle,insert,1)

old='''static void run_hwy2(AppleTwoCoreHighway &tc,Buffers &b)\n{\n    for(int c=0;c<6;c++) tc.run(b.x[c],b.y[c],b.n);\n}\n'''
new=old+'''static void run_highwaypool(AppleHighwayThreadPool2 &tc,Buffers &b)\n{\n    for(int c=0;c<6;c++) tc.run(b.x[c],b.y[c],b.n);\n}\nstatic void run_fastflow(AppleFastFlow2 &tc,Buffers &b)\n{\n    for(int c=0;c<6;c++) tc.run(b.x[c],b.y[c],b.n);\n}\n'''
assert old in s
s=s.replace(old,new,1)

bs=s.index('static int bench_mode(')
be=s.index('\n#ifdef APPLE_COS53_VALIDATE_MPFR',bs)
replacement=r'''template <class Runner>
static double measure_runner(Runner &runner,Buffers &b,size_t reps,double *cpu_out)
{
    auto once=[&]{ for(int c=0;c<6;c++) runner.run(b.x[c],b.y[c],b.n); };
    for(int w=0;w<16;w++) once();
    double c0=process_cpu_ns(); uint64_t w0=now_ticks();
    for(size_t r=0;r<reps;r++) once();
    uint64_t w1=now_ticks(); double c1=process_cpu_ns();
    const double den=(double)reps*(double)b.n*6.0;
    *cpu_out=(c1-c0)/den;
    g_sink += b.y[0][(b.n*7/11)%b.n];
    return ticks_to_ns(w1-w0)/den;
}

static int bench_one(const std::string &stack,size_t n)
{
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0);
    Buffers b(n);
    const size_t reps=reps_for(n);
    double cpu=0.0,wall=0.0;
    if(stack=="current") {
        AppleTwoCoreHighway runner;
        wall=measure_runner(runner,b,reps,&cpu);
    } else if(stack=="highway") {
        AppleHighwayThreadPool2 runner;
        wall=measure_runner(runner,b,reps,&cpu);
    } else if(stack=="fastflow") {
        AppleFastFlow2 runner;
        wall=measure_runner(runner,b,reps,&cpu);
    } else return 3;
    std::printf("APPLE_COS53_3WAY stack=%s n=%zu wall_ns_el=%.9f cpu_ns_el=%.9f effective_cores=%.6f reps=%zu sink=%.17g\n",
                stack.c_str(),n,wall,cpu,cpu/wall,reps,(double)g_sink);
    return 0;
}

static int verify_three()
{
    static const size_t sizes[]={64,100,400,1200,3000,5000,7500,10000,15000,20000,25000,29999};
    size_t dh=0,df=0;
    for(size_t n:sizes) {
        Buffers b(n);
        std::vector<double> a(n),h(n),f(n);
        {
            AppleTwoCoreHighway current;
            for(int c=0;c<6;c++) {
                current.run(b.x[c],a.data(),n);
                AppleHighwayThreadPool2 highway;
                highway.run(b.x[c],h.data(),n);
                AppleFastFlow2 fastflow;
                fastflow.run(b.x[c],f.data(),n);
                size_t xh=0,xf=0;
                for(size_t i=0;i<n;i++) {
                    xh += std::memcmp(&a[i],&h[i],sizeof(double))!=0;
                    xf += std::memcmp(&a[i],&f[i],sizeof(double))!=0;
                }
                dh+=xh; df+=xf;
                std::printf("APPLE_COS53_3WAY_VERIFY n=%zu case=%d highway_bitdiff=%zu fastflow_bitdiff=%zu\n",n,c,xh,xf);
            }
        }
    }
    std::printf("APPLE_COS53_3WAY_VERIFY_DONE highway_bitdiff=%zu fastflow_bitdiff=%zu\n",dh,df);
    return (dh||df)?21:0;
}
'''
s=s[:bs]+replacement+s[be:]

old='''    if(argc!=3) return 2;\n    std::string stack=argv[1]; size_t n=(size_t)std::strtoull(argv[2],nullptr,10);\n    if(stack!="hwy2" && stack!="apple") return 3;\n    return bench_mode(stack,n);'''
new='''    if(argc==2 && std::string(argv[1])=="verify-three") return verify_three();\n    if(argc!=4 || std::string(argv[1])!="one") return 2;\n    std::string stack=argv[2]; size_t n=(size_t)std::strtoull(argv[3],nullptr,10);\n    return bench_one(stack,n);'''
assert old in s
s=s.replace(old,new,1)
p.write_text(s)
PY

clang++ -O3 -DNDEBUG -std=c++20 -mcpu=native -fno-fast-math -ffp-contract=off -fblocks \
  -I/tmp -I/tmp/highway -I/tmp/fastflow \
  /tmp/base.cpp -L/tmp/highway-build -lhwy_contrib -lhwy \
  -framework Accelerate -pthread -ldl -o /tmp/apple_cos53_3way

if [[ "${1:-}" == "build" ]]; then
  echo "APPLE_COS53_3WAY_BUILD_OK"
  exit 0
fi
if [[ "${1:-}" == "verify" ]]; then
  exec /tmp/apple_cos53_3way verify-three
fi
if [[ "${1:-}" == "one" && $# -eq 3 ]]; then
  exec /tmp/apple_cos53_3way "$@"
fi

echo "usage: $0 build | verify | one current|highway|fastflow N" >&2
exit 2
