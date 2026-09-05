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
static inline void two_sum_vec(float64x2_t a, float64x2_t b, float64x2_t *h, float64x2_t *l)
{
    float64x2_t s = vaddq_f64(a,b);
    float64x2_t bv = vsubq_f64(s,a);
    float64x2_t av = vsubq_f64(s,bv);
    float64x2_t br = vsubq_f64(b,bv);
    float64x2_t ar = vsubq_f64(a,av);
    *h = s; *l = vaddq_f64(ar,br);
}
static inline float64x2_t gather2(const double *p, int a, int b)
{
    float64x2_t v = vdupq_n_f64(p[a]);
    return vsetq_lane_f64(p[b], v, 1);
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

static inline void cos53_eval_neon(const double * __restrict x, double * __restrict y, size_t n)
{
    const float64x2_t zero = vdupq_n_f64(0.0);
    const float64x2_t vk = vdupq_n_f64(KGRID);
    const float64x2_t vik = vdupq_n_f64(INVK);
    const float64x2_t mh = vdupq_n_f64(-0.5);
    const float64x2_t m6 = vdupq_n_f64(-1.0/6.0);
    const float64x2_t c24 = vdupq_n_f64(1.0/24.0);
    const float64x2_t c120 = vdupq_n_f64(1.0/120.0);
    size_t i=0;
    for (; i+2<=n; i+=2) {
        float64x2_t ax = vabsq_f64(vld1q_f64(x+i));
        float64x2_t qround = vrndnq_f64(vmulq_n_f64(ax, INVPI));
        int64x2_t qi = vcvtq_s64_f64(qround);
        int64_t qv[2]; vst1q_s64(qv,qi);
        int q0=(int)qv[0], q1=(int)qv[1];
        if ((unsigned)q0 >= APPLE_COS53_REDN || (unsigned)q1 >= APPLE_COS53_REDN) {
            y[i]=std::cos(x[i]); y[i+1]=std::cos(x[i+1]);
            continue;
        }
        float64x2_t ph = gather2(apple_cos53_pih,q0,q1);
        float64x2_t pl = gather2(apple_cos53_pil,q0,q1);
        float64x2_t s = vsubq_f64(ax,ph), rh,rl;
        two_sum_vec(s,vnegq_f64(pl),&rh,&rl);
        float64x2_t rs = vaddq_f64(rh,rl);
        uint64x2_t neg = vcltq_f64(rs,zero);
        rh = vbslq_f64(neg,vnegq_f64(rh),rh);
        rl = vbslq_f64(neg,vnegq_f64(rl),rl);
        float64x2_t r = vaddq_f64(rh,rl);
        int64x2_t ji = vcvtnq_s64_f64(vmulq_f64(r,vk));
        int64_t jv[2]; vst1q_s64(jv,ji);
        int j0=(int)jv[0], j1=(int)jv[1];
        if (j0<0) j0=0; else if (j0>=APPLE_COS53_LUTN) j0=APPLE_COS53_LUTN-1;
        if (j1<0) j1=0; else if (j1>=APPLE_COS53_LUTN) j1=APPLE_COS53_LUTN-1;
        float64x2_t jd = { (double)j0, (double)j1 };
        float64x2_t d = vaddq_f64(vsubq_f64(rh,vmulq_f64(jd,vik)),rl);
        float64x2_t c0=gather2(apple_cos53_c0,j0,j1);
        float64x2_t c1=gather2(apple_cos53_c1,j0,j1);
        float64x2_t c2=vmulq_f64(c0,mh);
        float64x2_t c3=vmulq_f64(c1,m6);
        float64x2_t c4=vmulq_f64(c0,c24);
        float64x2_t c5=vmulq_f64(c1,c120);
        float64x2_t p=vfmaq_f64(c4,c5,d);
        p=vfmaq_f64(c3,p,d);
        p=vfmaq_f64(c2,p,d);
        p=vfmaq_f64(c1,p,d);
        p=vfmaq_f64(c0,p,d);
        uint64_t sm0=(q0&1)?UINT64_C(0x8000000000000000):0;
        uint64_t sm1=(q1&1)?UINT64_C(0x8000000000000000):0;
        uint64x2_t sm={sm0,sm1};
        p=vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(p),sm));
        vst1q_f64(y+i,p);
    }
    if (i<n) {
        double xx[2]={x[i],x[i]}, yy[2];
        cos53_eval_neon(xx,yy,2); y[i]=yy[0];
    }
}

class AppleTwoCore {
    pthread_t th_{};
    dispatch_semaphore_t wake_;
    std::atomic<uint64_t> seq_{0}, done_{0};
    std::atomic<bool> stop_{false};
    const double *x_=nullptr; double *y_=nullptr; size_t begin_=0, end_=0;
    static void *entry(void *p) { ((AppleTwoCore*)p)->loop(); return nullptr; }
    void loop()
    {
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0);
        uint64_t seen=0;
        for (;;) {
            uint64_t s=seq_.load(std::memory_order_acquire);
            if (s!=seen) {
                seen=s;
                if (stop_.load(std::memory_order_relaxed)) break;
                cos53_eval_neon(x_+begin_,y_+begin_,end_-begin_);
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
    AppleTwoCore():wake_(dispatch_semaphore_create(0))
    {
        pthread_create(&th_,nullptr,&AppleTwoCore::entry,this);
    }
    ~AppleTwoCore()
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
        cos53_eval_neon(x,y,mid);
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
static void run_ours1(Buffers &b)
{
    for(int c=0;c<6;c++) cos53_eval_neon(b.x[c],b.y[c],b.n);
}
static void run_ours2(AppleTwoCore &tc,Buffers &b)
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
    AppleTwoCore *tc = stack=="ours2" ? new AppleTwoCore() : nullptr;
    auto once=[&]{ if(stack=="apple")run_apple(b); else if(stack=="ours2")run_ours2(*tc,b); else run_ours1(b); };
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
    std::printf("APPLE_COS53_RESULT stack=%s n=%zu wall_ns_el=%.9f cpu_ns_el=%.9f effective_cores=%.6f maxrss_bytes=%ld reps=%zu sink=%.17g\n",
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
    cos53_eval_neon(x.data(),o.data(),N);
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
    std::printf("APPLE_COS53_VALIDATE cases=%d ours_exact=%d ours_le1=%d ours_maxulp=%llu apple_exact=%d apple_le1=%d apple_maxulp=%llu reference=MPFR256\n",
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
    if(stack!="ours1" && stack!="ours2" && stack!="apple") return 3;
    return bench_mode(stack,n);
}
