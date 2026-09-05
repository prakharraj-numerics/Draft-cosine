#!/usr/bin/env bash
set -euo pipefail

[[ "$(uname -m)" == arm64 ]]
[[ "$(sysctl -n machdep.cpu.brand_string)" == *"Apple M1"* ]]

MODE="${1:-}"

if [[ "$MODE" == "build" ]]; then
  # Frozen build already installed these in the workflow, but keep this standalone.
  brew list mpfr >/dev/null 2>&1 || brew install mpfr
  brew list gmp >/dev/null 2>&1 || brew install gmp

  test -f /tmp/apple_cos53_constants_k1280_fastreduce.h
  test -f /tmp/pthreadpool-install/lib/libpthreadpool.a

  python3 - <<'PY'
from pathlib import Path
import re
h=Path('/tmp/apple_cos53_constants_k1280_fastreduce.h').read_text()
def vals(name):
    m=re.search(rf'static const double {name}\[APPLE_COS53_LUTN\] = \{{\s*(.*?)\}};',h,re.S)
    assert m,name
    v=re.findall(r'-?0x[0-9a-fA-F]+(?:\.[0-9a-fA-F]*)?p[+-]?\d+',m.group(1))
    assert len(v)==2012,(name,len(v))
    return v
c0=vals('apple_cos53_c0'); c1=vals('apple_cos53_c1')
out=['#pragma once','#include <cstddef>','#include <cstdint>',
     'static constexpr std::size_t OPT_COS53_LUTN = 2012;',
     'alignas(16) static const double opt_cos53_coeff_aos[OPT_COS53_LUTN*2] = {']
for a,b in zip(c0,c1):
    out.append(f'  {a}, {b},')
out.append('};')
Path('/tmp/apple_cos53_coeff_aos.h').write_text('\n'.join(out)+'\n')
PY

  cat >/tmp/apple_cos53_cpu_eff.cpp <<'CPP'
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <pthreadpool.h>
#include <pthread.h>
#include <sys/resource.h>
#include <mach/mach_time.h>
#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#ifdef OPT_VALIDATE_MPFR
#include <mpfr.h>
#endif

#include "apple_cos53_coeff_aos.h"

static constexpr double INVPI = 0x1.45f306dc9c883p-2;
static constexpr double KGRID = 1280.0;
static constexpr double NINVK_HI = -0x1.999999999999ap-11;
static constexpr double NINVK_LO = 0x1.999999999999ap-65;

// pi = P1 + P2 + P3 + O(2^-149).
// P1 has only ~40 significant bits, so q*P1 is exact for q<=3183.
static constexpr double PI_P1 = 0x1.921fb54442000p+1;
static constexpr double PI_P2 = 0x1.a308d313198a3p-40;
static constexpr double PI_P3 = -0x1.fc8f8cbb5bf6cp-96;

static constexpr double MH = -0x1.ffffff92c5f94p-2;
static constexpr double M6 = -0x1.5555551eb851fp-3;
static constexpr double XMAX = 10000.0;
static volatile double g_sink = 0.0;

static inline uint64_t now_ticks() { return mach_absolute_time(); }
static double ticks_to_ns(uint64_t t) {
    static mach_timebase_info_data_t tb=[] { mach_timebase_info_data_t x{}; mach_timebase_info(&x); return x; }();
    return (double)t*(double)tb.numer/(double)tb.denom;
}
static double process_cpu_ns() {
    rusage r{}; getrusage(RUSAGE_SELF,&r);
    return 1e9*(double)(r.ru_utime.tv_sec+r.ru_stime.tv_sec)
         + 1e3*(double)(r.ru_utime.tv_usec+r.ru_stime.tv_usec);
}
static uint64_t mix64(uint64_t x) {
    x^=x>>30; x*=UINT64_C(0xbf58476d1ce4e5b9); x^=x>>27;
    x*=UINT64_C(0x94d049bb133111eb); x^=x>>31; return x;
}
static double unit52(uint64_t h) { return ((double)(h>>12)+0.5)*0x1p-52; }

__attribute__((always_inline)) static inline void opt_pair(const double* x,double* y)
{
    float64x2_t xv=vld1q_f64(x);
    float64x2_t ax=vabsq_f64(xv);

    uint64x2_t valid=vcleq_f64(ax,vdupq_n_f64(XMAX));
    if ((vgetq_lane_u64(valid,0)&vgetq_lane_u64(valid,1)) != UINT64_MAX) {
        y[0]=std::cos(x[0]); y[1]=std::cos(x[1]); return;
    }

    // q = nearest integer to |x|/pi, entirely vectorized.
    int64x2_t qi=vcvtnq_s64_f64(vmulq_n_f64(ax,INVPI));
    float64x2_t qd=vcvtq_f64_s64(qi);

    // Split-pi Cody-Waite reduction. q*P1 is exact over the supported range.
    float64x2_t qp1=vmulq_n_f64(qd,PI_P1);
    float64x2_t t=vsubq_f64(ax,qp1);
    float64x2_t rh=vfmaq_n_f64(t,qd,-PI_P2);

    // Recover the rounding error of the P2 FMA into a low component, then P3.
    // t-rh is exact in the normal region by Sterbenz; the tiny exceptional
    // region is around multiples of pi where cosine is least sensitive.
    float64x2_t d=vsubq_f64(t,rh);
    float64x2_t rl=vfmaq_n_f64(d,qd,-PI_P2);
    rl=vfmaq_n_f64(rl,qd,-PI_P3);

    const uint64x2_t signmask=vdupq_n_u64(UINT64_C(0x8000000000000000));
    uint64x2_t rsign=vandq_u64(vreinterpretq_u64_f64(rh),signmask);
    float64x2_t ah=vreinterpretq_f64_u64(vbicq_u64(vreinterpretq_u64_f64(rh),signmask));
    float64x2_t al=vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(rl),rsign));

    // K=1280 and |r|<=pi/2 imply j is provably in [0,2011], so no clamps.
    int64x2_t ji=vcvtnq_s64_f64(vmulq_n_f64(ah,KGRID));
    const int64_t j0=vgetq_lane_s64(ji,0);
    const int64_t j1=vgetq_lane_s64(ji,1);
    float64x2_t jd=vcvtq_f64_s64(ji);

    // Split reciprocal subtraction: delta = |r| - j/1280 + low.
    float64x2_t delta=vfmaq_n_f64(ah,jd,NINVK_HI);
    delta=vfmaq_n_f64(delta,jd,NINVK_LO);
    delta=vaddq_f64(delta,al);

    // AoS coefficient layout: two 128-bit loads, then native zip/deinterleave.
    float64x2_t a=vld1q_f64(opt_cos53_coeff_aos+2*j0);
    float64x2_t b=vld1q_f64(opt_cos53_coeff_aos+2*j1);
    float64x2_t c0=vzip1q_f64(a,b);
    float64x2_t c1=vzip2q_f64(a,b);

    // Frozen degree-3 minimax-retuned polynomial.
    float64x2_t c2=vmulq_n_f64(c0,MH);
    float64x2_t c3=vmulq_n_f64(c1,M6);
    float64x2_t p=vfmaq_f64(c2,c3,delta);
    p=vfmaq_f64(c1,p,delta);
    p=vfmaq_f64(c0,p,delta);

    // (-1)^q sign without scalar q extraction.
    uint64x2_t parity=vandq_u64(vreinterpretq_u64_s64(qi),vdupq_n_u64(1));
    uint64x2_t outsign=vshlq_n_u64(parity,63);
    p=vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(p),outsign));
    vst1q_f64(y,p);

    // Root-only accuracy repair. j>=2009 means |r| is within ~1.3e-3 of
    // pi/2. This region is vanishingly rare for ordinary inputs but ULP
    // sensitivity is extreme there, so use Apple's scalar libm only for the
    // affected lane(s). The common path remains fully vectorized and retains
    // the split-pi/no-qpi-LUT reduction.
    if (__builtin_expect(j0 >= 2009, 0)) y[0]=std::cos(x[0]);
    if (__builtin_expect(j1 >= 2009, 0)) y[1]=std::cos(x[1]);
}

static inline void opt_cos53_eval(const double* x,double* y,size_t n)
{
    // Two independent Full128-equivalent vectors per iteration: lets the M1
    // overlap range-reduction, coefficient-load, and FMA dependency chains.
    size_t i=0;
    for (; i+4<=n; i+=4) {
        opt_pair(x+i,y+i);
        opt_pair(x+i+2,y+i+2);
    }
    if (i+2<=n) { opt_pair(x+i,y+i); i+=2; }
    if (i<n) {
        double xx[2]={x[i],x[i]}, yy[2];
        opt_pair(xx,yy); y[i]=yy[0];
    }
}

class Adaptive2 {
    std::thread helper_;
    alignas(64) std::atomic<uint64_t> generation_{0};
    alignas(64) std::atomic<uint64_t> completed_{0};
    std::atomic<bool> ready_{false},stop_{false};
    const double* x_=nullptr; double* y_=nullptr; size_t begin_=0,end_=0;

    static inline void relax() { __asm__ volatile("yield"); }

    void loop() {
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0);
        uint64_t seen=generation_.load(std::memory_order_relaxed);
        ready_.store(true,std::memory_order_release);
        for (;;) {
            uint64_t g=generation_.load(std::memory_order_acquire);
            if (g==seen) {
                // Short hot-spin for back-to-back calls, then park via atomic wait.
                for (int k=0;k<96 && (g=generation_.load(std::memory_order_acquire))==seen;k++) relax();
                if (g==seen) {
                    generation_.wait(seen,std::memory_order_acquire);
                    g=generation_.load(std::memory_order_acquire);
                }
            }
            seen=g;
            if (stop_.load(std::memory_order_relaxed)) return;
            opt_cos53_eval(x_+begin_,y_+begin_,end_-begin_);
            completed_.store(g,std::memory_order_release);
            completed_.notify_one();
        }
    }
public:
    Adaptive2() {
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0);
        helper_=std::thread([this]{loop();});
        while(!ready_.load(std::memory_order_acquire)) relax();
    }
    ~Adaptive2() {
        stop_.store(true,std::memory_order_relaxed);
        generation_.fetch_add(1,std::memory_order_release);
        generation_.notify_one();
        helper_.join();
    }
    void run(const double*x,double*y,size_t n) {
        if(n<4){opt_cos53_eval(x,y,n);return;}
        const size_t mid=(n/2)&~size_t(1);
        x_=x;y_=y;begin_=mid;end_=n;
        const uint64_t g=generation_.fetch_add(1,std::memory_order_release)+1;
        generation_.notify_one();
        opt_cos53_eval(x,y,mid);

        // Preserve hot-path latency for short imbalance, but do not burn a core
        // indefinitely waiting for the helper.
        for(int k=0;k<96;k++) {
            if(completed_.load(std::memory_order_acquire)==g) return;
            relax();
        }
        uint64_t old=completed_.load(std::memory_order_acquire);
        while(old!=g) {
            completed_.wait(old,std::memory_order_acquire);
            old=completed_.load(std::memory_order_acquire);
        }
    }
};

struct PoolCtx { const double* x; double* y; size_t n,p; };
static void pool_piece(void* vp,size_t j) {
    auto*c=(PoolCtx*)vp;
    auto boundary=[&](size_t k)->size_t {
        if(k==0) return 0;
        if(k==c->p) return c->n;
        return ((c->n*k)/c->p)&~size_t(1);
    };
    size_t a=boundary(j), b=boundary(j+1);
    opt_cos53_eval(c->x+a,c->y+a,b-a);
}
class PoolRunner {
    pthreadpool_t pool_;
    size_t p_;
public:
    explicit PoolRunner(size_t p):pool_(pthreadpool_create(p)),p_(p){if(!pool_)std::abort();}
    ~PoolRunner(){pthreadpool_destroy(pool_);}
    void run(const double*x,double*y,size_t n){
        if(n<2*p_){opt_cos53_eval(x,y,n);return;}
        PoolCtx c{x,y,n,p_};
        pthreadpool_parallelize_1d(pool_,pool_piece,&c,p_,0);
    }
};

class AutoRunner {
    Adaptive2 a2_;
    PoolRunner p2_{2},p3_{3};
public:
    void run(const double*x,double*y,size_t n){
        if(n<400) opt_cos53_eval(x,y,n);
        else if(n<5000) a2_.run(x,y,n);
        else if(n<100000) p2_.run(x,y,n);
        else p3_.run(x,y,n);
    }
};

static void fill_case(double*x,size_t n,int c) {
    const double lo[3]={0.0,1.0,1000.0}, hi[3]={1.0,500.0,10000.0};
    int b=c/2;
    uint64_t seed=UINT64_C(2026090501)+(uint64_t)c*UINT64_C(0x9e3779b97f4a7c15)+(uint64_t)n;
    for(size_t i=0;i<n;i++) {
        double u=unit52(mix64(seed+(uint64_t)i*UINT64_C(0x9e3779b97f4a7c15)));
        double v=std::fma(hi[b]-lo[b],u,lo[b]); x[i]=(c&1)?-v:v;
    }
}
struct Buffers {
    size_t n; std::vector<double*> x,y;
    explicit Buffers(size_t nn):n(nn),x(6),y(6) {
        for(int c=0;c<6;c++) {
            posix_memalign((void**)&x[c],64,n*sizeof(double));
            posix_memalign((void**)&y[c],64,n*sizeof(double));
            fill_case(x[c],n,c);
        }
    }
    ~Buffers(){for(auto p:x)free(p);for(auto p:y)free(p);}
};
static size_t reps_for(size_t n) { return std::max<size_t>(2,2000000/n); }

template<class R> static int bench_runner(const char* name,R&r,size_t n) {
    Buffers b(n); size_t reps=reps_for(n);
    for(int w=0;w<12;w++) for(int c=0;c<6;c++) r.run(b.x[c],b.y[c],n);
    double c0=process_cpu_ns(); uint64_t w0=now_ticks();
    for(size_t rr=0;rr<reps;rr++) for(int c=0;c<6;c++) r.run(b.x[c],b.y[c],n);
    uint64_t w1=now_ticks(); double c1=process_cpu_ns();
    double den=(double)reps*(double)n*6.0;
    double wall=ticks_to_ns(w1-w0)/den, cpu=(c1-c0)/den;
    g_sink+=b.y[0][(n*7/11)%n];
    std::printf("APPLE_COS53_CPU_EFF_RESULT stack=%s n=%zu wall_ns_el=%.9f cpu_ns_el=%.9f effective_cores=%.6f reps=%zu sink=%.17g\n",
                name,n,wall,cpu,cpu/wall,reps,(double)g_sink);
    return 0;
}
struct SingleRunner { void run(const double*x,double*y,size_t n){opt_cos53_eval(x,y,n);} };
struct AppleRunner {
    void run(const double*x,double*y,size_t n){int nn=(int)n;vvcos(y,x,&nn);}
};

#ifdef OPT_VALIDATE_MPFR
static inline uint64_t ordered_bits(double x) {
    uint64_t u; std::memcpy(&u,&x,8);
    return (u>>63)?~u:(u|UINT64_C(0x8000000000000000));
}
static inline uint64_t ulpd(double a,double b) {
    if(a==b)return 0; uint64_t x=ordered_bits(a),y=ordered_bits(b); return x>y?x-y:y-x;
}
static void score_set(const char* tag,const std::vector<double>&x) {
    std::vector<double> o(x.size()); opt_cos53_eval(x.data(),o.data(),x.size());
    mpfr_t z,r; mpfr_init2(z,256); mpfr_init2(r,256);
    uint64_t maxu=0; size_t exact=0,le1=0,le2=0,le3=0,gt3=0,worst=0;
    double wr=0;
    for(size_t i=0;i<x.size();i++) {
        mpfr_set_d(z,x[i],MPFR_RNDN); mpfr_cos(r,z,MPFR_RNDN);
        double ref=mpfr_get_d(r,MPFR_RNDN); uint64_t u=ulpd(o[i],ref);
        exact+=(u==0); le1+=(u<=1); le2+=(u<=2); le3+=(u<=3); gt3+=(u>3);
        if(u>maxu){maxu=u;worst=i;wr=ref;}
    }
    std::printf("APPLE_COS53_CPU_EFF_ACCURACY tag=%s cases=%zu exact=%zu le1=%zu le2=%zu le3=%zu gt3=%zu maxulp=%llu worst_i=%zu worst_x=%.17g worst_out=%.17g worst_ref=%.17g\n",
        tag,x.size(),exact,le1,le2,le3,gt3,(unsigned long long)maxu,worst,x[worst],o[worst],wr);
    mpfr_clear(r);mpfr_clear(z);
}
static int validate() {
    std::vector<double>x9600(9600);
    const double lo[3]={0.0,1.0,1000.0},hi[3]={1.0,500.0,10000.0};
    for(int i=0;i<9600;i++) {
        int c=i%6,b=c/2;
        double u=unit52(mix64(UINT64_C(2026090619)+(uint64_t)i*UINT64_C(0x9e3779b97f4a7c15)));
        double v=std::fma(hi[b]-lo[b],u,lo[b]);x9600[i]=(c&1)?-v:v;
    }
    score_set("9600",x9600);

    const int N=1000000,RN=980000;
    std::vector<double>x(N);
    for(int i=0;i<RN;i++) {
        int c=i%6,b=c/2;
        double u=unit52(mix64(UINT64_C(2026090621)+(uint64_t)i*UINT64_C(0x9e3779b97f4a7c15)));
        double v=std::fma(hi[b]-lo[b],u,lo[b]);x[i]=(c&1)?-v:v;
    }
    const long double PIL=acosl(-1.0L);
    for(int i=RN;i<N;i++) {
        uint64_t h=mix64(UINT64_C(0xd2b74407b1ce6e93)+(uint64_t)i*UINT64_C(0x9e3779b97f4a7c15));
        int k=(int)(h%6366ULL)-3183;
        long double root=0.5L*PIL+(long double)k*PIL;
        double v=(double)root;
        int steps=(int)((h>>16)%33ULL);
        double dir=((h>>24)&1ULL)?INFINITY:-INFINITY;
        for(int j=0;j<steps;j++)v=std::nextafter(v,dir);
        x[i]=v;
    }
    score_set("1m_stress",x);
    return 0;
}
#endif

int main(int argc,char**argv) {
#ifdef OPT_VALIDATE_MPFR
    if(argc==2 && std::string(argv[1])=="validate") return validate();
#endif
    if(argc!=3)return 2;
    std::string mode=argv[1]; size_t n=(size_t)std::strtoull(argv[2],nullptr,10);
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0);
    if(mode=="single"){SingleRunner r;return bench_runner("single",r,n);}
    if(mode=="adapt2"){Adaptive2 r;return bench_runner("adapt2",r,n);}
    if(mode=="pool2"){PoolRunner r(2);return bench_runner("pool2",r,n);}
    if(mode=="pool3"){PoolRunner r(3);return bench_runner("pool3",r,n);}
    if(mode=="auto"){AutoRunner r;return bench_runner("auto",r,n);}
    if(mode=="apple"){AppleRunner r;return bench_runner("apple",r,n);}
    return 3;
}
CPP

  MP="$(brew --prefix mpfr)"; GP="$(brew --prefix gmp)"
  COMMON="-O3 -DNDEBUG -std=c++20 -mcpu=native -fno-fast-math -ffp-contract=off"
  clang++ $COMMON -I/tmp -I/tmp/pthreadpool-install/include \
    /tmp/apple_cos53_cpu_eff.cpp /tmp/pthreadpool-install/lib/libpthreadpool.a \
    -framework Accelerate -pthread -o /tmp/apple_cos53_cpu_eff_bench
  clang++ $COMMON -DOPT_VALIDATE_MPFR -I/tmp -I/tmp/pthreadpool-install/include \
    -I"$MP/include" -I"$GP/include" /tmp/apple_cos53_cpu_eff.cpp \
    /tmp/pthreadpool-install/lib/libpthreadpool.a -framework Accelerate -pthread \
    -L"$MP/lib" -L"$GP/lib" -lmpfr -lgmp -o /tmp/apple_cos53_cpu_eff_validate
  exit 0
fi

if [[ "$MODE" == "validate" ]]; then
  exec /tmp/apple_cos53_cpu_eff_validate validate
fi

if [[ "$MODE" == "one" ]]; then
  [[ $# -eq 3 ]]
  exec /tmp/apple_cos53_cpu_eff_bench "$2" "$3"
fi

echo "usage: $0 build | validate | one {single|adapt2|pool2|pool3|auto|apple} N" >&2
exit 2
