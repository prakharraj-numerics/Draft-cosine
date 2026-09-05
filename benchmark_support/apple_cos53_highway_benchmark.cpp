#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
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
#include <vector>
#ifdef APPLE_COS53_VALIDATE_MPFR
#include <mpfr.h>
#endif

#define HWY_COMPILE_ONLY_STATIC
#include <hwy/highway.h>

#include "apple_cos53_constants.h"

namespace hn = hwy::HWY_NAMESPACE;

static constexpr double INVPI = 0x1.45f306dc9c883p-2;
static constexpr double KGRID = 256.0;
static constexpr double INVK = 1.0 / 256.0;
static volatile double g_sink = 0.0;

static inline uint64_t now_ticks() { return mach_absolute_time(); }
static double ticks_to_ns(uint64_t t)
{
    static mach_timebase_info_data_t tb = []{ mach_timebase_info_data_t x{}; mach_timebase_info(&x); return x; }();
    return (double)t * (double)tb.numer / (double)tb.denom;
}
static double process_cpu_ns()
{
    rusage r{};
    getrusage(RUSAGE_SELF, &r);
    return 1.0e9 * (double)(r.ru_utime.tv_sec + r.ru_stime.tv_sec)
         + 1.0e3 * (double)(r.ru_utime.tv_usec + r.ru_stime.tv_usec);
}
static inline uint64_t ordered_bits(double x)
{
    uint64_t u; std::memcpy(&u,&x,8);
    return (u >> 63) ? ~u : (u | UINT64_C(0x8000000000000000));
}
static inline uint64_t ulpd(double a, double b)
{
    if (a == b) return 0;
    uint64_t x=ordered_bits(a), y=ordered_bits(b);
    return x>y?x-y:y-x;
}

HWY_ATTR static inline void cos53_eval_hwy(const double * HWY_RESTRICT x,
                                           double * HWY_RESTRICT y,
                                           size_t n)
{
    hn::Full128<double> d;
    hn::RebindToSigned<decltype(d)> di;
    hn::RebindToUnsigned<decltype(d)> du;
    const auto zero = hn::Zero(d);
    const auto vinvpi = hn::Set(d, INVPI);
    const auto vk = hn::Set(d, KGRID);
    const auto vik = hn::Set(d, INVK);
    const auto mh = hn::Set(d, -0.5);
    const auto m6 = hn::Set(d, -1.0/6.0);
    const auto c24v = hn::Set(d, 1.0/24.0);
    const auto c120v = hn::Set(d, 1.0/120.0);

    size_t i=0;
    for (; i+2<=n; i+=2) {
        auto ax = hn::Abs(hn::LoadU(d, x+i));
        auto qi = hn::NearestInt(hn::Mul(ax, vinvpi));
        int64_t qv[2]; hn::StoreU(qi, di, qv);
        int q0=(int)qv[0], q1=(int)qv[1];
        if ((unsigned)q0 >= APPLE_COS53_REDN || (unsigned)q1 >= APPLE_COS53_REDN) {
            y[i]=std::cos(x[i]); y[i+1]=std::cos(x[i+1]);
            continue;
        }

        auto ph = hn::Dup128VecFromValues(d, apple_cos53_pih[q0], apple_cos53_pih[q1]);
        auto pl = hn::Dup128VecFromValues(d, apple_cos53_pil[q0], apple_cos53_pil[q1]);
        auto s = hn::Sub(ax, ph);
        auto b = hn::Neg(pl);

        // Exact TwoSum, matching the current Apple NEON path operation-for-operation.
        auto rh = hn::Add(s, b);
        auto bv = hn::Sub(rh, s);
        auto av = hn::Sub(rh, bv);
        auto br = hn::Sub(b, bv);
        auto ar = hn::Sub(s, av);
        auto rl = hn::Add(ar, br);

        auto rs = hn::Add(rh, rl);
        auto neg = hn::Lt(rs, zero);
        rh = hn::IfThenElse(neg, hn::Neg(rh), rh);
        rl = hn::IfThenElse(neg, hn::Neg(rl), rl);
        auto r = hn::Add(rh, rl);

        auto ji = hn::NearestInt(hn::Mul(r, vk));
        int64_t jv[2]; hn::StoreU(ji, di, jv);
        int j0=(int)jv[0], j1=(int)jv[1];
        if (j0<0) j0=0; else if (j0>=APPLE_COS53_LUTN) j0=APPLE_COS53_LUTN-1;
        if (j1<0) j1=0; else if (j1>=APPLE_COS53_LUTN) j1=APPLE_COS53_LUTN-1;

        auto jd = hn::ConvertTo(d, ji);
        auto delta = hn::Add(hn::Sub(rh, hn::Mul(jd, vik)), rl);
        auto c0 = hn::Dup128VecFromValues(d, apple_cos53_c0[j0], apple_cos53_c0[j1]);
        auto c1 = hn::Dup128VecFromValues(d, apple_cos53_c1[j0], apple_cos53_c1[j1]);
        auto c2 = hn::Mul(c0, mh);
        auto c3 = hn::Mul(c1, m6);
        auto c4 = hn::Mul(c0, c24v);
        auto c5 = hn::Mul(c1, c120v);

        auto p = hn::MulAdd(c5, delta, c4);
        p = hn::MulAdd(p, delta, c3);
        p = hn::MulAdd(p, delta, c2);
        p = hn::MulAdd(p, delta, c1);
        p = hn::MulAdd(p, delta, c0);

        uint64_t sm0=(q0&1)?UINT64_C(0x8000000000000000):0;
        uint64_t sm1=(q1&1)?UINT64_C(0x8000000000000000):0;
        auto sm = hn::Dup128VecFromValues(du, sm0, sm1);
        p = hn::BitCast(d, hn::Xor(hn::BitCast(du, p), sm));
        hn::StoreU(p, d, y+i);
    }
    if (i<n) {
        double xx[2]={x[i],x[i]}, yy[2];
        cos53_eval_hwy(xx,yy,2); y[i]=yy[0];
    }
}

class AppleTwoCoreHighway {
    pthread_t th_{};
    dispatch_semaphore_t wake_;
    std::atomic<uint64_t> seq_{0}, done_{0};
    std::atomic<bool> stop_{false};
    const double *x_=nullptr; double *y_=nullptr; size_t begin_=0, end_=0;
    static void *entry(void *p) { ((AppleTwoCoreHighway*)p)->loop(); return nullptr; }
    void loop()
    {
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0);
        uint64_t seen=0;
        for (;;) {
            uint64_t s=seq_.load(std::memory_order_acquire);
            if (s!=seen) {
                seen=s;
                if (stop_.load(std::memory_order_relaxed)) break;
                cos53_eval_hwy(x_+begin_,y_+begin_,end_-begin_);
                done_.store(seen,std::memory_order_release);
                for (int k=0;k<4000;k++) {
                    uint64_t z=seq_.load(std::memory_order_acquire);
                    if (z!=seen) break;
                    __asm__ volatile("yield");
                }
                continue;
            }
            dispatch_semaphore_wait(wake_,DISPATCH_TIME_FOREVER);
        }
    }
public:
    AppleTwoCoreHighway():wake_(dispatch_semaphore_create(0))
    {
        pthread_create(&th_,nullptr,&AppleTwoCoreHighway::entry,this);
    }
    ~AppleTwoCoreHighway()
    {
        stop_.store(true,std::memory_order_relaxed);
        seq_.fetch_add(1,std::memory_order_release);
        dispatch_semaphore_signal(wake_);
        pthread_join(th_,nullptr);
    }
    void run(const double *x,double *y,size_t n)
    {
        size_t mid=(n/2)&~(size_t)1;
        x_=x; y_=y; begin_=mid; end_=n;
        uint64_t s=seq_.fetch_add(1,std::memory_order_acq_rel)+1;
        dispatch_semaphore_signal(wake_);
        cos53_eval_hwy(x,y,mid);
        while(done_.load(std::memory_order_acquire)!=s) __asm__ volatile("yield");
    }
};

static uint64_t mix64(uint64_t x)
{
    x^=x>>30; x*=UINT64_C(0xbf58476d1ce4e5b9); x^=x>>27;
    x*=UINT64_C(0x94d049bb133111eb); x^=x>>31; return x;
}
static double unit52(uint64_t h) { return ((double)(h>>12)+0.5)*0x1p-52; }
static void fill_case(double *x,size_t n,int c)
{
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
static void run_apple(Buffers &b)
{
    int nn=(int)b.n;
    for(int c=0;c<6;c++) vvcos(b.y[c],b.x[c],&nn);
}
static void run_hwy2(AppleTwoCoreHighway &tc,Buffers &b)
{
    for(int c=0;c<6;c++) tc.run(b.x[c],b.y[c],b.n);
}
static size_t reps_for(size_t n)
{
    size_t r=(size_t)12000000/(n*6);
    if(r<2)r=2; if(r>30000)r=30000; return r;
}
static double median7(double a[7]) { std::sort(a,a+7); return a[3]; }

static int bench_mode(const std::string &stack,size_t n)
{
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0);
    Buffers b(n);
    AppleTwoCoreHighway *tc = stack=="hwy2" ? new AppleTwoCoreHighway() : nullptr;
    auto once=[&]{ if(stack=="apple")run_apple(b); else run_hwy2(*tc,b); };
    for(int w=0;w<20;w++) once();
    size_t reps=reps_for(n);
    double wt[7],ct[7];
    for(int t=0;t<7;t++) {
        double c0=process_cpu_ns(); uint64_t w0=now_ticks();
        for(size_t r=0;r<reps;r++) once();
        uint64_t w1=now_ticks(); double c1=process_cpu_ns();
        double den=(double)reps*(double)n*6.0;
        wt[t]=ticks_to_ns(w1-w0)/den; ct[t]=(c1-c0)/den;
        g_sink += b.y[t%6][(n*7/11)%n];
    }
    double wall=median7(wt),cpu=median7(ct);
    rusage ru{}; getrusage(RUSAGE_SELF,&ru);
    std::printf("APPLE_COS53_HIGHWAY_RESULT stack=%s n=%zu wall_ns_el=%.9f cpu_ns_el=%.9f effective_cores=%.6f maxrss_bytes=%ld reps=%zu sink=%.17g\n",
                stack.c_str(),n,wall,cpu,cpu/wall,(long)ru.ru_maxrss,reps,(double)g_sink);
    delete tc;
    return 0;
}

#ifdef APPLE_COS53_VALIDATE_MPFR
static int validate_mode()
{
    const int N=9600;
    std::vector<double> x(N),o(N),a(N);
    for(int i=0;i<N;i++) {
        int c=i%6,b=c/2;
        const double lo[3]={0.0,1.0,1000.0},hi[3]={1.0,500.0,10000.0};
        double u=unit52(mix64(UINT64_C(2026090599)+(uint64_t)i*UINT64_C(0x9e3779b97f4a7c15)));
        double v=std::fma(hi[b]-lo[b],u,lo[b]); x[i]=(c&1)?-v:v;
    }
    cos53_eval_hwy(x.data(),o.data(),N);
    int nn=N; vvcos(a.data(),x.data(),&nn);
    mpfr_t z,r; mpfr_init2(z,256); mpfr_init2(r,256);
    uint64_t mo=0,ma=0; int oe=0,o1=0,ae=0,a1=0;
    for(int i=0;i<N;i++) {
        mpfr_set_d(z,x[i],MPFR_RNDN); mpfr_cos(r,z,MPFR_RNDN);
        double ref=mpfr_get_d(r,MPFR_RNDN);
        uint64_t uo=ulpd(o[i],ref), ua=ulpd(a[i],ref);
        mo=std::max(mo,uo); ma=std::max(ma,ua);
        if(uo==0)oe++; if(uo<=1)o1++; if(ua==0)ae++; if(ua<=1)a1++;
    }
    mpfr_clear(r);mpfr_clear(z);
    std::printf("APPLE_COS53_HIGHWAY_VALIDATE cases=%d hwy_exact=%d hwy_le1=%d hwy_maxulp=%llu apple_exact=%d apple_le1=%d apple_maxulp=%llu reference=MPFR256\n",
                N,oe,o1,(unsigned long long)mo,ae,a1,(unsigned long long)ma);
    return 0;
}
#endif

int main(int argc,char **argv)
{
#ifdef APPLE_COS53_VALIDATE_MPFR
    if(argc==2 && std::string(argv[1])=="validate") return validate_mode();
#endif
    if(argc!=3) return 2;
    std::string stack=argv[1]; size_t n=(size_t)std::strtoull(argv[2],nullptr,10);
    if(stack!="hwy2" && stack!="apple") return 3;
    return bench_mode(stack,n);
}
