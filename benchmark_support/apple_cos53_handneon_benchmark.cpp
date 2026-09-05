#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
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

#include "apple_cos53_constants.h"

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
static inline float64x2_t gather2_fast(const double *p, int a, int b)
{
    float64x2_t v = vdupq_n_f64(p[a]);
    return vsetq_lane_f64(p[b], v, 1);
}

// Hand NEON candidate derived from the Highway 1.4.0 code shape that won the
// previous Apple experiment. Key differences from the old NEON path:
//  * keep rounded j as an int64x2_t and convert it directly to double in-register;
//  * extract q/j lanes instead of spilling whole vectors to stack arrays;
//  * build the final parity sign mask vectorially from qi;
//  * preserve the exact Highway/NEON FMA Horner ordering.
__attribute__((always_inline)) static inline void cos53_eval_handneon(const double * __restrict x,
                                                                       double * __restrict y,
                                                                       size_t n)
{
    const float64x2_t zero = vdupq_n_f64(0.0);
    const float64x2_t vk = vdupq_n_f64(KGRID);
    const float64x2_t vik = vdupq_n_f64(INVK);
    const float64x2_t mh = vdupq_n_f64(-0.5);
    const float64x2_t m6 = vdupq_n_f64(-1.0/6.0);
    const float64x2_t c24 = vdupq_n_f64(1.0/24.0);
    const float64x2_t c120 = vdupq_n_f64(1.0/120.0);
    const uint64x2_t one = vdupq_n_u64(1);

    size_t i=0;
    for (; i+2<=n; i+=2) {
        const float64x2_t ax = vabsq_f64(vld1q_f64(x+i));
        const float64x2_t qround = vrndnq_f64(vmulq_n_f64(ax, INVPI));
        const int64x2_t qi = vcvtq_s64_f64(qround);
        const int q0 = (int)vgetq_lane_s64(qi,0);
        const int q1 = (int)vgetq_lane_s64(qi,1);
        if ((unsigned)q0 >= APPLE_COS53_REDN || (unsigned)q1 >= APPLE_COS53_REDN) {
            y[i]=std::cos(x[i]); y[i+1]=std::cos(x[i+1]);
            continue;
        }

        const float64x2_t ph = gather2_fast(apple_cos53_pih,q0,q1);
        const float64x2_t pl = gather2_fast(apple_cos53_pil,q0,q1);
        const float64x2_t s = vsubq_f64(ax,ph);
        const float64x2_t b = vnegq_f64(pl);

        const float64x2_t rh0 = vaddq_f64(s,b);
        const float64x2_t bv = vsubq_f64(rh0,s);
        const float64x2_t av = vsubq_f64(rh0,bv);
        const float64x2_t br = vsubq_f64(b,bv);
        const float64x2_t ar = vsubq_f64(s,av);
        float64x2_t rh = rh0;
        float64x2_t rl = vaddq_f64(ar,br);

        const float64x2_t rs = vaddq_f64(rh,rl);
        const uint64x2_t neg = vcltq_f64(rs,zero);
        rh = vbslq_f64(neg,vnegq_f64(rh),rh);
        rl = vbslq_f64(neg,vnegq_f64(rl),rl);
        const float64x2_t r = vaddq_f64(rh,rl);

        const int64x2_t ji = vcvtnq_s64_f64(vmulq_f64(r,vk));
        int j0=(int)vgetq_lane_s64(ji,0);
        int j1=(int)vgetq_lane_s64(ji,1);
        if (j0<0) j0=0; else if (j0>=APPLE_COS53_LUTN) j0=APPLE_COS53_LUTN-1;
        if (j1<0) j1=0; else if (j1>=APPLE_COS53_LUTN) j1=APPLE_COS53_LUTN-1;

        // For valid inputs ji is already in range, so preserve Highway's cheap
        // in-register int64->double conversion instead of rebuilding a vector.
        const float64x2_t jd = vcvtq_f64_s64(ji);
        const float64x2_t d = vaddq_f64(vsubq_f64(rh,vmulq_f64(jd,vik)),rl);

        const float64x2_t c0=gather2_fast(apple_cos53_c0,j0,j1);
        const float64x2_t c1=gather2_fast(apple_cos53_c1,j0,j1);
        const float64x2_t c2=vmulq_f64(c0,mh);
        const float64x2_t c3=vmulq_f64(c1,m6);
        const float64x2_t c4=vmulq_f64(c0,c24);
        const float64x2_t c5=vmulq_f64(c1,c120);

        float64x2_t p=vfmaq_f64(c4,c5,d);
        p=vfmaq_f64(c3,p,d);
        p=vfmaq_f64(c2,p,d);
        p=vfmaq_f64(c1,p,d);
        p=vfmaq_f64(c0,p,d);

        const uint64x2_t parity = vandq_u64(vreinterpretq_u64_s64(qi),one);
        const uint64x2_t sm = vshlq_n_u64(parity,63);
        p=vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(p),sm));
        vst1q_f64(y+i,p);
    }
    if (i<n) {
        double xx[2]={x[i],x[i]}, yy[2];
        cos53_eval_handneon(xx,yy,2); y[i]=yy[0];
    }
}

class AppleTwoCoreHandNEON {
    pthread_t th_{};
    dispatch_semaphore_t wake_;
    std::atomic<uint64_t> seq_{0}, done_{0};
    std::atomic<bool> stop_{false};
    const double *x_=nullptr; double *y_=nullptr; size_t begin_=0, end_=0;
    static void *entry(void *p) { ((AppleTwoCoreHandNEON*)p)->loop(); return nullptr; }
    void loop()
    {
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0);
        uint64_t seen=0;
        for (;;) {
            uint64_t s=seq_.load(std::memory_order_acquire);
            if (s!=seen) {
                seen=s;
                if (stop_.load(std::memory_order_relaxed)) break;
                cos53_eval_handneon(x_+begin_,y_+begin_,end_-begin_);
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
    AppleTwoCoreHandNEON():wake_(dispatch_semaphore_create(0)) { pthread_create(&th_,nullptr,&AppleTwoCoreHandNEON::entry,this); }
    ~AppleTwoCoreHandNEON()
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
        cos53_eval_handneon(x,y,mid);
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
static void run_apple(Buffers &b) { int nn=(int)b.n; for(int c=0;c<6;c++) vvcos(b.y[c],b.x[c],&nn); }
static void run_hand2(AppleTwoCoreHandNEON &tc,Buffers &b) { for(int c=0;c<6;c++) tc.run(b.x[c],b.y[c],b.n); }
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
    AppleTwoCoreHandNEON *tc = stack=="hand2" ? new AppleTwoCoreHandNEON() : nullptr;
    auto once=[&]{ if(stack=="apple")run_apple(b); else run_hand2(*tc,b); };
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
    std::printf("APPLE_COS53_HANDNEON_RESULT stack=%s n=%zu wall_ns_el=%.9f cpu_ns_el=%.9f effective_cores=%.6f maxrss_bytes=%ld reps=%zu sink=%.17g\n",
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
    cos53_eval_handneon(x.data(),o.data(),N);
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
    std::printf("APPLE_COS53_HANDNEON_VALIDATE cases=%d hand_exact=%d hand_le1=%d hand_maxulp=%llu apple_exact=%d apple_le1=%d apple_maxulp=%llu reference=MPFR256\n",
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
    if(stack!="hand2" && stack!="apple") return 3;
    return bench_mode(stack,n);
}
