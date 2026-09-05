#!/usr/bin/env bash
set -euo pipefail

[[ "$(uname -m)" == arm64 ]]
[[ "$(sysctl -n machdep.cpu.brand_string)" == *"Apple M1"* ]]

MODE="${1:-}"

if [[ "$MODE" == "build" ]]; then
  brew list flint >/dev/null 2>&1 || brew install flint
  brew list mpfr >/dev/null 2>&1 || brew install mpfr
  brew list gmp >/dev/null 2>&1 || brew install gmp

  rm -rf /tmp/highway /tmp/pthreadpool /tmp/pthreadpool-build /tmp/pthreadpool-install /tmp/corsix-amx
  git clone --depth 1 --branch 1.4.0 https://github.com/google/highway.git /tmp/highway
  git clone --depth 1 https://github.com/Maratyszcza/pthreadpool.git /tmp/pthreadpool
  git clone --depth 1 https://github.com/corsix/amx.git /tmp/corsix-amx

  # Match the frozen 5K-30K production baseline: pthreadpool with the portable
  # pthread/condvar backend, not pthreadpool's default Apple GCD backend.
  cmake -S /tmp/pthreadpool -B /tmp/pthreadpool-build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/tmp/pthreadpool-install \
    -DPTHREADPOOL_LIBRARY_TYPE=static \
    -DPTHREADPOOL_SYNC_PRIMITIVE=condvar \
    -DPTHREADPOOL_BUILD_TESTS=OFF \
    -DPTHREADPOOL_BUILD_BENCHMARKS=OFF
  cmake --build /tmp/pthreadpool-build --target pthreadpool -j 4
  cmake --install /tmp/pthreadpool-build

  FREEZE=aefbe778e860ef70e64fc8d6b6d470b3575f3bbc
  git show "$FREEZE":benchmark_support/apple_cos53_highway_benchmark.cpp > /tmp/base.cpp
  git show "$FREEZE":benchmark_support/sine_53_coeff_source.c > /tmp/src.c
  git show "$FREEZE":benchmark_support/apple_cos53_coeff_bridge.c > /tmp/bridge.c
  git show "$FREEZE":benchmark_support/apple_cos53_generate_constants.c > /tmp/gen.c
  git show "$FREEZE":cosine53_apply_formula_conversion.py > /tmp/cosine53_apply_formula_conversion.py

  # Recreate the current frozen K=2048, degree-3, terms=1 coefficient state.
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
  /tmp/gen /tmp/apple_cos53_constants_amx.h

  python3 - <<'PY'
from pathlib import Path
p=Path('/tmp/base.cpp'); s=p.read_text()
s=s.replace('#include <vector>', '#include <vector>\n#include <arm_neon.h>\n#include <pthreadpool.h>\n#include "aarch64.h"')
s=s.replace('#include "apple_cos53_constants.h"','#include "apple_cos53_constants_amx.h"')
s=s.replace('static constexpr double KGRID = 256.0;','static constexpr double KGRID = 2048.0;').replace('static constexpr double INVK = 1.0 / 256.0;','static constexpr double INVK = 1.0 / 2048.0;')
s=s.replace('    const auto c24v = hn::Set(d, 1.0/24.0);\n    const auto c120v = hn::Set(d, 1.0/120.0);\n','')
start=s.index('        auto c2 = hn::Mul(c0, mh);')
end=s.index('        uint64_t sm0=',start)
s=s[:start]+'''        auto c2 = hn::Mul(c0, mh);\n        auto c3 = hn::Mul(c1, m6);\n        auto p = hn::MulAdd(c3, delta, c2);\n        p = hn::MulAdd(p, delta, c1);\n        p = hn::MulAdd(p, delta, c0);\n\n'''+s[end:]

needle='''};\n\nstatic uint64_t mix64(uint64_t x)'''
insert=r'''};

// Direct Apple AMX candidate. Highway is deliberately not used anywhere in
// this evaluator: manual AArch64 NEON does reduction/index preparation, then
// AMX VECFP in f64 mode performs the degree-3 Horner chain 8 lanes at a time.
// AMX ABI is respected: SET once on function entry, CLR before return.
static inline void cos53_eval_amx(const double* x, double* y, size_t n)
{
    if (n == 0) return;
    AMX_SET();

    alignas(64) double xin[8], delta[8], c0a[8], c1a[8], c2a[8], c3a[8], pout[8];
    alignas(64) int64_t qbuf[8], jbuf[8];

    const float64x2_t vinvpi = vdupq_n_f64(INVPI);
    const float64x2_t vk = vdupq_n_f64(KGRID);
    const float64x2_t vik = vdupq_n_f64(INVK);
    const float64x2_t z = vdupq_n_f64(0.0);
    const float64x2_t mh = vdupq_n_f64(-0.5);
    const float64x2_t m6 = vdupq_n_f64(-1.0/6.0);

    for (size_t base=0; base<n; base+=8) {
        const size_t lanes = std::min<size_t>(8, n-base);
        if (lanes == 8) {
            std::memcpy(xin, x+base, 8*sizeof(double));
        } else {
            for (size_t k=0;k<lanes;k++) xin[k]=x[base+k];
            for (size_t k=lanes;k<8;k++) xin[k]=xin[lanes-1];
        }

        bool fallback=false;
        float64x2_t rhv[4], rlv[4];
        for (int g=0;g<4;g++) {
            const int o=2*g;
            float64x2_t ax=vabsq_f64(vld1q_f64(xin+o));
            int64x2_t qi=vcvtnq_s64_f64(vmulq_f64(ax,vinvpi));
            vst1q_s64(qbuf+o,qi);
            if ((uint64_t)qbuf[o] >= APPLE_COS53_REDN || (uint64_t)qbuf[o+1] >= APPLE_COS53_REDN) {
                fallback=true; continue;
            }
            double pha[2]={apple_cos53_pih[qbuf[o]],apple_cos53_pih[qbuf[o+1]]};
            double pla[2]={apple_cos53_pil[qbuf[o]],apple_cos53_pil[qbuf[o+1]]};
            float64x2_t ph=vld1q_f64(pha), pl=vld1q_f64(pla);
            float64x2_t ss=vsubq_f64(ax,ph);
            float64x2_t bb=vnegq_f64(pl);
            float64x2_t rh=vaddq_f64(ss,bb);
            float64x2_t bv=vsubq_f64(rh,ss);
            float64x2_t av=vsubq_f64(rh,bv);
            float64x2_t br=vsubq_f64(bb,bv);
            float64x2_t ar=vsubq_f64(ss,av);
            float64x2_t rl=vaddq_f64(ar,br);
            float64x2_t rs=vaddq_f64(rh,rl);
            uint64x2_t neg=vcltq_f64(rs,z);
            rh=vbslq_f64(neg,vnegq_f64(rh),rh);
            rl=vbslq_f64(neg,vnegq_f64(rl),rl);
            rhv[g]=rh; rlv[g]=rl;
            float64x2_t r=vaddq_f64(rh,rl);
            int64x2_t ji=vcvtnq_s64_f64(vmulq_f64(r,vk));
            vst1q_s64(jbuf+o,ji);
            for (int k=0;k<2;k++) {
                if (jbuf[o+k] < 0) jbuf[o+k]=0;
                else if (jbuf[o+k] >= APPLE_COS53_LUTN) jbuf[o+k]=APPLE_COS53_LUTN-1;
            }
            // Re-load clamped ji only for the table index; mathematically ji is
            // already in range on this domain, so this preserves the frozen path.
            int64_t jiraw[2]; vst1q_s64(jiraw,ji);
            float64x2_t jd=vcvtq_f64_s64(vld1q_s64(jiraw));
            float64x2_t dd=vaddq_f64(vsubq_f64(rh,vmulq_f64(jd,vik)),rl);
            vst1q_f64(delta+o,dd);
            double c0t[2]={apple_cos53_c0[jbuf[o]],apple_cos53_c0[jbuf[o+1]]};
            double c1t[2]={apple_cos53_c1[jbuf[o]],apple_cos53_c1[jbuf[o+1]]};
            float64x2_t c0=vld1q_f64(c0t), c1=vld1q_f64(c1t);
            vst1q_f64(c0a+o,c0); vst1q_f64(c1a+o,c1);
            vst1q_f64(c2a+o,vmulq_f64(c0,mh));
            vst1q_f64(c3a+o,vmulq_f64(c1,m6));
        }

        if (fallback) {
            // This is outside the intended <=10K validation/benchmark domain.
            // Keep the function safe and honor the AMX ABI.
            AMX_CLR();
            for (size_t k=0;k<lanes;k++) y[base+k]=std::cos(x[base+k]);
            if (base+lanes<n) cos53_eval_amx(x+base+lanes,y+base+lanes,n-base-lanes);
            return;
        }

        // p = fma(c3,d,c2); p=fma(p,d,c1); p=fma(p,d,c0)
        // AMX VECFP operand (7<<42): f64 lanes, z += x*y, Z row 0,
        // X offset 0, Y offset 0, all 8 lanes enabled.
        AMX_LDX((uint64_t)c3a);
        AMX_LDY((uint64_t)delta);
        AMX_LDZ((uint64_t)c2a);
        AMX_VECFP(7ull << 42);
        // EXTRX opcode with bits 27=26=0 decodes as EXTRH: copy Z row 0 to X0.
        AMX_EXTRX(0);
        AMX_LDZ((uint64_t)c1a);
        AMX_VECFP(7ull << 42);
        AMX_EXTRX(0);
        AMX_LDZ((uint64_t)c0a);
        AMX_VECFP(7ull << 42);
        AMX_STZ((uint64_t)pout);

        // Match the frozen implementation's final sign-bit XOR exactly.
        for (int g=0;g<4;g++) {
            const int o=2*g;
            uint64_t smv[2]={
                (qbuf[o]&1)?UINT64_C(0x8000000000000000):0,
                (qbuf[o+1]&1)?UINT64_C(0x8000000000000000):0
            };
            uint64x2_t pu=vreinterpretq_u64_f64(vld1q_f64(pout+o));
            pu=veorq_u64(pu,vld1q_u64(smv));
            vst1q_f64(pout+o,vreinterpretq_f64_u64(pu));
        }
        std::memcpy(y+base,pout,lanes*sizeof(double));
    }
    AMX_CLR();
}

class AppleTwoCoreAMX {
    pthread_t th_{};
    dispatch_semaphore_t wake_;
    std::atomic<uint64_t> seq_{0}, done_{0};
    std::atomic<bool> stop_{false};
    const double *x_=nullptr; double *y_=nullptr; size_t begin_=0, end_=0;
    static void *entry(void *p) { ((AppleTwoCoreAMX*)p)->loop(); return nullptr; }
    void loop() {
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0);
        uint64_t seen=0;
        for (;;) {
            uint64_t ss=seq_.load(std::memory_order_acquire);
            if (ss!=seen) {
                seen=ss;
                if (stop_.load(std::memory_order_relaxed)) break;
                cos53_eval_amx(x_+begin_,y_+begin_,end_-begin_);
                done_.store(seen,std::memory_order_release);
                for(int k=0;k<4000;k++) {
                    uint64_t zz=seq_.load(std::memory_order_acquire);
                    if(zz!=seen) break;
                    __asm__ volatile("yield");
                }
                continue;
            }
            dispatch_semaphore_wait(wake_,DISPATCH_TIME_FOREVER);
        }
    }
public:
    AppleTwoCoreAMX():wake_(dispatch_semaphore_create(0)) { pthread_create(&th_,nullptr,&AppleTwoCoreAMX::entry,this); }
    ~AppleTwoCoreAMX() {
        stop_.store(true,std::memory_order_relaxed); seq_.fetch_add(1,std::memory_order_release);
        dispatch_semaphore_signal(wake_); pthread_join(th_,nullptr);
    }
    void run(const double *x,double *y,size_t n) {
        size_t mid=(n/2)&~(size_t)1;
        x_=x; y_=y; begin_=mid; end_=n;
        uint64_t ss=seq_.fetch_add(1,std::memory_order_acq_rel)+1;
        dispatch_semaphore_signal(wake_);
        cos53_eval_amx(x,y,mid);
        while(done_.load(std::memory_order_acquire)!=ss) __asm__ volatile("yield");
    }
};

struct PoolCtx { const double* x; double* y; size_t n; size_t mid; bool amx; };
static void pool_piece(void* vp,size_t task) {
    auto* c=static_cast<PoolCtx*>(vp);
    const size_t a=task?c->mid:0, b=task?c->n:c->mid;
    if(c->amx) cos53_eval_amx(c->x+a,c->y+a,b-a);
    else cos53_eval_hwy(c->x+a,c->y+a,b-a);
}
class FrozenBaselineRunner {
    AppleTwoCoreHighway tc_;
    pthreadpool_t pool_;
public:
    FrozenBaselineRunner():pool_(pthreadpool_create(2)){if(!pool_)std::abort();}
    ~FrozenBaselineRunner(){pthreadpool_destroy(pool_);}
    void run(const double*x,double*y,size_t n){
        if(n<5000){tc_.run(x,y,n);return;}
        size_t mid=(n/2)&~(size_t)1; PoolCtx c{x,y,n,mid,false};
        pthreadpool_parallelize_1d(pool_,pool_piece,&c,2,0);
    }
};
class DirectAMXRunner {
    AppleTwoCoreAMX tc_;
    pthreadpool_t pool_;
public:
    DirectAMXRunner():pool_(pthreadpool_create(2)){if(!pool_)std::abort();}
    ~DirectAMXRunner(){pthreadpool_destroy(pool_);}
    void run(const double*x,double*y,size_t n){
        if(n<5000){tc_.run(x,y,n);return;}
        size_t mid=(n/2)&~(size_t)1; PoolCtx c{x,y,n,mid,true};
        pthreadpool_parallelize_1d(pool_,pool_piece,&c,2,0);
    }
};

static uint64_t mix64(uint64_t x)'''
assert needle in s
s=s.replace(needle,insert,1)

bs=s.index('static int bench_mode(')
be=s.index('\n#ifdef APPLE_COS53_VALIDATE_MPFR',bs)
replacement=r'''template<class Runner>
static double measure_runner(Runner& r,Buffers& b,size_t reps,double* cpu_out) {
    for(int w=0;w<12;w++) for(int c=0;c<6;c++) r.run(b.x[c],b.y[c],b.n);
    double c0=process_cpu_ns(); uint64_t w0=now_ticks();
    for(size_t rr=0;rr<reps;rr++) for(int c=0;c<6;c++) r.run(b.x[c],b.y[c],b.n);
    uint64_t w1=now_ticks(); double c1=process_cpu_ns();
    double den=(double)reps*(double)b.n*6.0;
    *cpu_out=(c1-c0)/den; g_sink+=b.y[0][(b.n*7/11)%b.n];
    return ticks_to_ns(w1-w0)/den;
}

static int probe_amx() {
    AMX_SET(); AMX_CLR();
    std::puts("APPLE_COS53_AMX_PROBE ok=1");
    return 0;
}

static int verify_amx() {
    static const size_t sizes[]={64,100,400,1200,3000,4999,5000,7500,10000,15000,20000,25000,29999};
    size_t total=0;
    for(size_t n:sizes) {
        Buffers b(n); std::vector<double> ref(n),out(n);
        FrozenBaselineRunner baseline; DirectAMXRunner amx;
        for(int c=0;c<6;c++) {
            baseline.run(b.x[c],ref.data(),n);
            amx.run(b.x[c],out.data(),n);
            size_t d=0; uint64_t mu=0;
            for(size_t i=0;i<n;i++){ d += std::memcmp(&ref[i],&out[i],8)!=0; mu=std::max(mu,ulpd(ref[i],out[i])); }
            total+=d;
            std::printf("APPLE_COS53_AMX_VERIFY n=%zu case=%d bitdiff=%zu max_vs_baseline_ulp=%llu\n",n,c,d,(unsigned long long)mu);
        }
    }
    std::printf("APPLE_COS53_AMX_VERIFY_DONE bitdiff=%zu\n",total);
    return 0;
}

static int bench_one(const std::string& stack,size_t n) {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0);
    Buffers b(n); size_t reps=reps_for(n); double cpu=0,wall=0;
    if(stack=="baseline") { FrozenBaselineRunner r; wall=measure_runner(r,b,reps,&cpu); }
    else if(stack=="amx") { DirectAMXRunner r; wall=measure_runner(r,b,reps,&cpu); }
    else return 3;
    std::printf("APPLE_COS53_AMX_RESULT stack=%s n=%zu wall_ns_el=%.9f cpu_ns_el=%.9f effective_cores=%.6f reps=%zu sink=%.17g\n",
        stack.c_str(),n,wall,cpu,cpu/wall,reps,(double)g_sink);
    return 0;
}
'''
s=s[:bs]+replacement+s[be:]

old='''    if(argc!=3) return 2;\n    std::string stack=argv[1]; size_t n=(size_t)std::strtoull(argv[2],nullptr,10);\n    if(stack!="hwy2" && stack!="apple") return 3;\n    return bench_mode(stack,n);'''
new='''    if(argc==2 && std::string(argv[1])=="probe") return probe_amx();\n    if(argc==2 && std::string(argv[1])=="verify") return verify_amx();\n    if(argc!=3) return 2;\n    return bench_one(argv[1],(size_t)std::strtoull(argv[2],nullptr,10));'''
assert old in s
s=s.replace(old,new,1)
p.write_text(s)
PY

  clang++ -O3 -DNDEBUG -std=c++20 -mcpu=native -fno-fast-math -ffp-contract=off -fblocks \
    -I/tmp -I/tmp/highway -I/tmp/pthreadpool-install/include -I/tmp/corsix-amx \
    /tmp/base.cpp /tmp/pthreadpool-install/lib/libpthreadpool.a \
    -framework Accelerate -pthread -ldl -o /tmp/apple_cos53_amx
  exit 0
fi

if [[ "$MODE" == "probe" ]]; then
  exec /tmp/apple_cos53_amx probe
fi
if [[ "$MODE" == "verify" ]]; then
  exec /tmp/apple_cos53_amx verify
fi
if [[ "$MODE" == "one" ]]; then
  [[ $# -eq 3 ]]
  exec /tmp/apple_cos53_amx "$2" "$3"
fi

echo "usage: $0 build | probe | verify | one baseline|amx N" >&2
exit 2
