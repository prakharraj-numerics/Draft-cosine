#!/usr/bin/env bash
set -euo pipefail

[[ "$(uname -m)" == arm64 ]]
[[ "$(sysctl -n machdep.cpu.brand_string)" == *"Apple M1"* ]]

MODE="${1:-}"

if [[ "$MODE" == "build" ]]; then
  brew list flint >/dev/null 2>&1 || brew install flint
  brew list mpfr >/dev/null 2>&1 || brew install mpfr
  brew list gmp >/dev/null 2>&1 || brew install gmp
  brew list tbb >/dev/null 2>&1 || brew install tbb

  rm -rf /tmp/highway /tmp/pi-threadpool /tmp/pi-build /tmp/qthreads /tmp/qthreads-build /tmp/qthreads-install
  git clone --depth 1 --branch 1.4.0 https://github.com/google/highway.git /tmp/highway

  git clone --depth 1 https://github.com/PrimeIntellect-ai/threadpool.git /tmp/pi-threadpool
  git -C /tmp/pi-threadpool submodule update --init --recursive
  # macOS SDK exports a global constant named `pi`; rename PrimeIntellect's
  # namespace locally for this benchmark, without changing library behavior.
  sed -i '' 's/namespace pi::threadpool/namespace pithreadpool_apple::threadpool/g' /tmp/pi-threadpool/include/pithreadpool/threadpool.hpp
  sed -i '' 's/namespace pi::threadpool/namespace pithreadpool_apple::threadpool/g' /tmp/pi-threadpool/src/threadpool.cpp
  cmake -S /tmp/pi-threadpool -B /tmp/pi-build -DCMAKE_BUILD_TYPE=Release -DPI_THREADPOOL_BUILD_TESTS=OFF
  cmake --build /tmp/pi-build -j 4

  git clone --depth 1 https://github.com/sandialabs/qthreads.git /tmp/qthreads
  cmake -S /tmp/qthreads -B /tmp/qthreads-build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/tmp/qthreads-install \
    -DBUILD_SHARED_LIBS=ON
  cmake --build /tmp/qthreads-build --target qthread -j 4
  cmake --install /tmp/qthreads-build

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
s=s.replace('#include <vector>','#include <vector>\n#include <memory>\n#include <oneapi/tbb/task_arena.h>\n#include <oneapi/tbb/parallel_invoke.h>\n#include <pithreadpool/threadpool.hpp>\nextern "C" {\n#include <qthread/qthread.h>\n}')
s=s.replace('#include "apple_cos53_constants.h"','#include "apple_cos53_constants_4way.h"')
s=s.replace('static constexpr double KGRID = 256.0;','static constexpr double KGRID = 2048.0;').replace('static constexpr double INVK = 1.0 / 256.0;','static constexpr double INVK = 1.0 / 2048.0;')
s=s.replace('    const auto c24v = hn::Set(d, 1.0/24.0);\n    const auto c120v = hn::Set(d, 1.0/120.0);\n','')
start=s.index('        auto c2 = hn::Mul(c0, mh);')
end=s.index('        uint64_t sm0=',start)
s=s[:start]+'''        auto c2 = hn::Mul(c0, mh);\n        auto c3 = hn::Mul(c1, m6);\n        auto p = hn::MulAdd(c3, delta, c2);\n        p = hn::MulAdd(p, delta, c1);\n        p = hn::MulAdd(p, delta, c0);\n\n'''+s[end:]

needle='''};\n\nstatic uint64_t mix64(uint64_t x)'''
insert=r'''};

class AppleOneTBB2 {
    oneapi::tbb::task_arena arena_;
public:
    AppleOneTBB2(): arena_(2) { arena_.initialize(); }
    void run(const double *x,double *y,size_t n) {
        if (!n) return;
        if (n < 4) { cos53_eval_hwy(x,y,n); return; }
        const size_t mid=(n/2)&~(size_t)1;
        arena_.execute([&]{
            oneapi::tbb::parallel_invoke(
                [&]{ cos53_eval_hwy(x,y,mid); },
                [&]{ cos53_eval_hwy(x+mid,y+mid,n-mid); });
        });
    }
};

class ApplePrime2 {
    pithreadpool_apple::threadpool::ThreadPool pool_;
public:
    ApplePrime2(): pool_(1, 64) { pool_.startup(); }
    ~ApplePrime2() { pool_.shutdown(); }
    void run(const double *x,double *y,size_t n) {
        if (!n) return;
        if (n < 4) { cos53_eval_hwy(x,y,n); return; }
        const size_t mid=(n/2)&~(size_t)1;
        auto fut=pool_.scheduleTask([=]{ cos53_eval_hwy(x+mid,y+mid,n-mid); });
        cos53_eval_hwy(x,y,mid);
        fut.join();
    }
};

struct QTArgs { const double* x; double* y; size_t n; };
static aligned_t qt_cos53_half(void* vp) {
    QTArgs* a=(QTArgs*)vp;
    cos53_eval_hwy(a->x,a->y,a->n);
    return 0;
}
class AppleQthreads2 {
public:
    AppleQthreads2() {
        setenv("QTHREAD_NUM_SHEPHERDS","1",1);
        setenv("QTHREAD_NUM_WORKERS_PER_SHEPHERD","1",1);
        if (qthread_initialize()!=QTHREAD_SUCCESS) std::abort();
    }
    void run(const double *x,double *y,size_t n) {
        if (!n) return;
        if (n < 4) { cos53_eval_hwy(x,y,n); return; }
        const size_t mid=(n/2)&~(size_t)1;
        QTArgs arg{x+mid,y+mid,n-mid};
        aligned_t done=0;
        if (qthread_fork(qt_cos53_half,&arg,&done)!=QTHREAD_SUCCESS) std::abort();
        cos53_eval_hwy(x,y,mid);
        qthread_readFF(nullptr,&done);
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
    static const size_t sizes[]={64,100,400,1200,3000,5000,7500,10000,15000,20000,25000,29999};
    size_t diff=0;
    for(size_t n:sizes) {
        Buffers b(n); std::vector<double> ref(n),out(n);
        for(int c=0;c<6;c++) {
            cos53_eval_hwy(b.x[c],ref.data(),n);
            runner.run(b.x[c],out.data(),n);
            size_t d=0; for(size_t i=0;i<n;i++) d += std::memcmp(&ref[i],&out[i],sizeof(double))!=0;
            diff+=d;
            std::printf("APPLE_COS53_4WAY_VERIFY stack=%s n=%zu case=%d bitdiff=%zu\n",stack.c_str(),n,c,d);
        }
    }
    std::printf("APPLE_COS53_4WAY_VERIFY_DONE stack=%s bitdiff=%zu\n",stack.c_str(),diff);
    return diff?21:0;
}

static int bench_one(const std::string& stack,size_t n)
{
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0);
    Buffers b(n); const size_t reps=reps_for(n); double cpu=0.0,wall=0.0;
    if(stack=="current") { AppleTwoCoreHighway r; wall=measure_runner(r,b,reps,&cpu); }
    else if(stack=="tbb") { AppleOneTBB2 r; wall=measure_runner(r,b,reps,&cpu); }
    else if(stack=="prime") { ApplePrime2 r; wall=measure_runner(r,b,reps,&cpu); }
    else if(stack=="qthreads") { AppleQthreads2 r; wall=measure_runner(r,b,reps,&cpu); }
    else return 3;
    std::printf("APPLE_COS53_4WAY stack=%s n=%zu wall_ns_el=%.9f cpu_ns_el=%.9f effective_cores=%.6f reps=%zu sink=%.17g\n",
                stack.c_str(),n,wall,cpu,cpu/wall,reps,(double)g_sink);
    return 0;
}

static int verify_one(const std::string& stack)
{
    if(stack=="current") { AppleTwoCoreHighway r; return verify_runner(stack,r); }
    if(stack=="tbb") { AppleOneTBB2 r; return verify_runner(stack,r); }
    if(stack=="prime") { ApplePrime2 r; return verify_runner(stack,r); }
    if(stack=="qthreads") { AppleQthreads2 r; return verify_runner(stack,r); }
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

  TBBP="$(brew --prefix tbb)"
  clang++ -O3 -DNDEBUG -std=c++20 -mcpu=native -fno-fast-math -ffp-contract=off -fblocks \
    -I/tmp -I/tmp/highway -I/tmp/pi-threadpool/include -I/tmp/pi-threadpool/third_party/threadpark/include \
    -I/tmp/qthreads-install/include -I"$TBBP/include" \
    /tmp/base.cpp /tmp/pi-build/libthreadpool.a /tmp/pi-build/third_party/threadpark/libthreadpark.a \
    -L/tmp/qthreads-install/lib -Wl,-rpath,/tmp/qthreads-install/lib \
    -L"$TBBP/lib" -lqthread -ltbb \
    -framework Accelerate -pthread -ldl -o /tmp/apple_cos53_4way
  exit 0
fi

if [[ "$MODE" == "verify" ]]; then
  [[ $# -eq 2 ]]
  exec /tmp/apple_cos53_4way verify "$2"
fi

if [[ "$MODE" == "one" ]]; then
  [[ $# -eq 3 ]]
  exec /tmp/apple_cos53_4way "$2" "$3"
fi

echo "usage: $0 build | verify STACK | one STACK N" >&2
exit 2
