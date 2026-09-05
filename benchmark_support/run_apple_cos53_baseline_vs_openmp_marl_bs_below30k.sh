#!/usr/bin/env bash
set -euo pipefail

[[ "$(uname -m)" == arm64 ]]
[[ "$(sysctl -n machdep.cpu.brand_string)" == *"Apple M1"* ]]

MODE="${1:-}"

if [[ "$MODE" == "build" ]]; then
  brew list flint >/dev/null 2>&1 || brew install flint
  brew list mpfr >/dev/null 2>&1 || brew install mpfr
  brew list gmp >/dev/null 2>&1 || brew install gmp
  brew list libomp >/dev/null 2>&1 || brew install libomp

  rm -rf /tmp/highway /tmp/pthreadpool /tmp/pthreadpool-build /tmp/pthreadpool-install \
         /tmp/marl /tmp/marl-build /tmp/bs-thread-pool

  git clone --depth 1 --branch 1.4.0 https://github.com/google/highway.git /tmp/highway
  git clone --depth 1 https://github.com/Maratyszcza/pthreadpool.git /tmp/pthreadpool
  git clone --depth 1 https://github.com/google/marl.git /tmp/marl
  git clone --depth 1 https://github.com/bshoshany/thread-pool.git /tmp/bs-thread-pool

  # Frozen baseline contract: pthreadpool on Apple must use the pthread/condvar
  # backend rather than its default GCD backend.
  cmake -S /tmp/pthreadpool -B /tmp/pthreadpool-build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/tmp/pthreadpool-install \
    -DPTHREADPOOL_LIBRARY_TYPE=static \
    -DPTHREADPOOL_SYNC_PRIMITIVE=condvar \
    -DPTHREADPOOL_BUILD_TESTS=OFF \
    -DPTHREADPOOL_BUILD_BENCHMARKS=OFF
  cmake --build /tmp/pthreadpool-build --target pthreadpool -j 4
  cmake --install /tmp/pthreadpool-build

  cmake -S /tmp/marl -B /tmp/marl-build \
    -DCMAKE_BUILD_TYPE=Release \
    -DMARL_BUILD_EXAMPLES=OFF \
    -DMARL_BUILD_TESTS=OFF \
    -DMARL_BUILD_BENCHMARKS=OFF \
    -DMARL_BUILD_SHARED=OFF
  cmake --build /tmp/marl-build --target marl -j 4

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
  /tmp/gen /tmp/apple_cos53_constants_4way.h

  python3 - <<'PY'
from pathlib import Path
p=Path('/tmp/base.cpp'); s=p.read_text()
s=s.replace('#include <vector>', '#include <vector>\n#include <memory>\n#include <future>\n#include <omp.h>\n#include <pthreadpool.h>\n#include <marl/scheduler.h>\n#include <marl/waitgroup.h>\n#include <BS_thread_pool.hpp>')
s=s.replace('#include "apple_cos53_constants.h"','#include "apple_cos53_constants_4way.h"')
s=s.replace('static constexpr double KGRID = 256.0;','static constexpr double KGRID = 2048.0;').replace('static constexpr double INVK = 1.0 / 256.0;','static constexpr double INVK = 1.0 / 2048.0;')
s=s.replace('    const auto c24v = hn::Set(d, 1.0/24.0);\n    const auto c120v = hn::Set(d, 1.0/120.0);\n','')
start=s.index('        auto c2 = hn::Mul(c0, mh);')
end=s.index('        uint64_t sm0=',start)
s=s[:start]+'''        auto c2 = hn::Mul(c0, mh);\n        auto c3 = hn::Mul(c1, m6);\n        auto p = hn::MulAdd(c3, delta, c2);\n        p = hn::MulAdd(p, delta, c1);\n        p = hn::MulAdd(p, delta, c0);\n\n'''+s[end:]

needle='''};\n\nstatic uint64_t mix64(uint64_t x)'''
insert=r'''};

struct PThreadPoolCtx {
    const double* x;
    double* y;
    size_t n;
    size_t mid;
};
static void pthreadpool_cos53_chunk(void* vp, size_t task)
{
    auto* c = static_cast<PThreadPoolCtx*>(vp);
    if(task == 0) cos53_eval_hwy(c->x, c->y, c->mid);
    else cos53_eval_hwy(c->x + c->mid, c->y + c->mid, c->n - c->mid);
}
class ApplePThreadPool2 {
    pthreadpool_t pool_;
public:
    ApplePThreadPool2(): pool_(pthreadpool_create(2)) { if(!pool_) std::abort(); }
    ~ApplePThreadPool2() { pthreadpool_destroy(pool_); }
    void run(const double *x,double *y,size_t n) {
        if(!n) return;
        if(n < 4) { cos53_eval_hwy(x,y,n); return; }
        const size_t mid=(n/2)&~(size_t)1;
        PThreadPoolCtx c{x,y,n,mid};
        pthreadpool_parallelize_1d(pool_, pthreadpool_cos53_chunk, &c, 2, 0);
    }
};

// Exact sub-30K frozen baseline: current AppleTwoCoreHighway below 5K,
// pthreadpool-condvar with two tasks from 5K through 29,999.
class AppleFrozenBaselineBelow30K {
    AppleTwoCoreHighway tc_;
    std::unique_ptr<ApplePThreadPool2> pp_;
public:
    void run(const double *x,double *y,size_t n) {
        if(n < 5000) { tc_.run(x,y,n); return; }
        if(!pp_) pp_ = std::make_unique<ApplePThreadPool2>();
        pp_->run(x,y,n);
    }
};

class AppleOpenMP2 {
public:
    AppleOpenMP2() {
        omp_set_dynamic(0);
        omp_set_num_threads(2);
        #pragma omp parallel num_threads(2)
        { }
    }
    void run(const double *x,double *y,size_t n) {
        if(!n) return;
        if(n < 4) { cos53_eval_hwy(x,y,n); return; }
        const size_t mid=(n/2)&~(size_t)1;
        #pragma omp parallel num_threads(2)
        {
            const int tid=omp_get_thread_num();
            if(tid==0) cos53_eval_hwy(x,y,mid);
            else if(tid==1) cos53_eval_hwy(x+mid,y+mid,n-mid);
        }
    }
};

class AppleMarl2 {
    static marl::Scheduler::Config make_config() {
        marl::Scheduler::Config c;
        c.setWorkerThreadCount(1);
        return c;
    }
    marl::Scheduler scheduler_;
public:
    AppleMarl2(): scheduler_(make_config()) { scheduler_.bind(); }
    ~AppleMarl2() { marl::Scheduler::unbind(); }
    void run(const double *x,double *y,size_t n) {
        if(!n) return;
        if(n < 4) { cos53_eval_hwy(x,y,n); return; }
        const size_t mid=(n/2)&~(size_t)1;
        marl::WaitGroup done(1);
        marl::schedule([=] {
            cos53_eval_hwy(x+mid,y+mid,n-mid);
            done.done();
        });
        cos53_eval_hwy(x,y,mid);
        done.wait();
    }
};

class AppleBS2 {
    BS::thread_pool<> pool_{1};
public:
    void run(const double *x,double *y,size_t n) {
        if(!n) return;
        if(n < 4) { cos53_eval_hwy(x,y,n); return; }
        const size_t mid=(n/2)&~(size_t)1;
        auto fut=pool_.submit_task([=] { cos53_eval_hwy(x+mid,y+mid,n-mid); });
        cos53_eval_hwy(x,y,mid);
        fut.wait();
    }
};

static uint64_t mix64(uint64_t x)'''
assert needle in s
s=s.replace(needle,insert,1)

bs=s.index('static int bench_mode(')
be=s.index('\n#ifdef APPLE_COS53_VALIDATE_MPFR',bs)
replacement=r'''template<class Runner>
static double measure_runner(Runner &runner,Buffers &b,size_t reps,double *cpu_out)
{
    for(int w=0;w<12;w++) for(int c=0;c<6;c++) runner.run(b.x[c],b.y[c],b.n);
    double c0=process_cpu_ns(); uint64_t w0=now_ticks();
    for(size_t r=0;r<reps;r++) for(int c=0;c<6;c++) runner.run(b.x[c],b.y[c],b.n);
    uint64_t w1=now_ticks(); double c1=process_cpu_ns();
    const double den=(double)reps*(double)b.n*6.0;
    *cpu_out=(c1-c0)/den;
    g_sink += b.y[0][(b.n*7/11)%b.n];
    return ticks_to_ns(w1-w0)/den;
}

template<class Runner>
static int verify_runner(const std::string& stack,Runner &runner)
{
    static const size_t sizes[]={64,100,400,1200,3000,4999,5000,7500,10000,15000,20000,25000,29999};
    size_t diff=0;
    for(size_t n:sizes) {
        Buffers b(n); std::vector<double> ref(n),out(n);
        for(int c=0;c<6;c++) {
            cos53_eval_hwy(b.x[c],ref.data(),n);
            runner.run(b.x[c],out.data(),n);
            size_t d=0; for(size_t i=0;i<n;i++) d += std::memcmp(&ref[i],&out[i],sizeof(double))!=0;
            diff+=d;
            std::printf("APPLE_COS53_NEW4_VERIFY stack=%s n=%zu case=%d bitdiff=%zu\n",stack.c_str(),n,c,d);
        }
    }
    std::printf("APPLE_COS53_NEW4_VERIFY_DONE stack=%s bitdiff=%zu\n",stack.c_str(),diff);
    return diff?21:0;
}

static int bench_one(const std::string& stack,size_t n)
{
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0);
    Buffers b(n); const size_t reps=reps_for(n); double cpu=0.0,wall=0.0;
    if(stack=="baseline") { AppleFrozenBaselineBelow30K r; wall=measure_runner(r,b,reps,&cpu); }
    else if(stack=="openmp") { AppleOpenMP2 r; wall=measure_runner(r,b,reps,&cpu); }
    else if(stack=="marl") { AppleMarl2 r; wall=measure_runner(r,b,reps,&cpu); }
    else if(stack=="bs") { AppleBS2 r; wall=measure_runner(r,b,reps,&cpu); }
    else return 3;
    std::printf("APPLE_COS53_NEW4 stack=%s n=%zu wall_ns_el=%.9f cpu_ns_el=%.9f effective_cores=%.6f reps=%zu sink=%.17g\n",
                stack.c_str(),n,wall,cpu,cpu/wall,reps,(double)g_sink);
    return 0;
}

static int verify_one(const std::string& stack)
{
    if(stack=="baseline") { AppleFrozenBaselineBelow30K r; return verify_runner(stack,r); }
    if(stack=="openmp") { AppleOpenMP2 r; return verify_runner(stack,r); }
    if(stack=="marl") { AppleMarl2 r; return verify_runner(stack,r); }
    if(stack=="bs") { AppleBS2 r; return verify_runner(stack,r); }
    return 3;
}
'''
s=s[:bs]+replacement+s[be:]

old='''    if(argc!=3) return 2;\n    std::string stack=argv[1]; size_t n=(size_t)std::strtoull(argv[2],nullptr,10);\n    if(stack!="hwy2" && stack!="apple") return 3;\n    return bench_mode(stack,n);'''
new='''    if(argc==3 && std::string(argv[1])=="verify") return verify_one(argv[2]);\n    if(argc!=3) return 2;\n    return bench_one(argv[1],(size_t)std::strtoull(argv[2],nullptr,10));'''
assert old in s
s=s.replace(old,new,1)
p.write_text(s)
PY

  OMPP="$(brew --prefix libomp)"
  clang++ -O3 -DNDEBUG -std=c++20 -mcpu=native -fno-fast-math -ffp-contract=off -fblocks \
    -Xpreprocessor -fopenmp \
    -I/tmp -I/tmp/highway -I/tmp/pthreadpool-install/include -I/tmp/marl/include \
    -I/tmp/bs-thread-pool/include -I"$OMPP/include" \
    /tmp/base.cpp /tmp/pthreadpool-install/lib/libpthreadpool.a /tmp/marl-build/libmarl.a \
    -L"$OMPP/lib" -Wl,-rpath,"$OMPP/lib" -lomp \
    -framework Accelerate -pthread -ldl -o /tmp/apple_cos53_new4
  exit 0
fi

if [[ "$MODE" == "verify" ]]; then
  [[ $# -eq 2 ]]
  exec /tmp/apple_cos53_new4 verify "$2"
fi

if [[ "$MODE" == "one" ]]; then
  [[ $# -eq 3 ]]
  exec /tmp/apple_cos53_new4 "$2" "$3"
fi

echo "usage: $0 build | verify STACK | one STACK N" >&2
exit 2
