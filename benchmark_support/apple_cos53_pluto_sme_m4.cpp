#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <arm_sve.h>
#include <pthread.h>
#include <sys/resource.h>
#include <mach/mach_time.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#ifdef OPT_VALIDATE_MPFR
#include <mpfr.h>
#endif
#include "apple_cos53_coeff_aos.h"

static constexpr double INVPI = 0x1.45f306dc9c883p-2;
static constexpr double KGRID = 1280.0;
static constexpr double NINVK_HI = -0x1.999999999999ap-11;
static constexpr double NINVK_LO = 0x1.999999999999ap-65;
static constexpr double PI_P1 = 0x1.921fb54442000p+1;
static constexpr double PI_P2 = 0x1.a308d313198a3p-40;
static constexpr double MH = -0x1.ffffff92c5f94p-2;
static constexpr double M6 = -0x1.5555551eb851fp-3;
static volatile double g_sink = 0.0;

static inline uint64_t now_ticks(){return mach_absolute_time();}
static double ticks_to_ns(uint64_t t){static mach_timebase_info_data_t tb=[] {mach_timebase_info_data_t x{};mach_timebase_info(&x);return x;}();return (double)t*tb.numer/tb.denom;}
static double process_cpu_ns(){rusage r{};getrusage(RUSAGE_SELF,&r);return 1e9*(r.ru_utime.tv_sec+r.ru_stime.tv_sec)+1e3*(r.ru_utime.tv_usec+r.ru_stime.tv_usec);}
static uint64_t mix64(uint64_t x){x^=x>>30;x*=UINT64_C(0xbf58476d1ce4e5b9);x^=x>>27;x*=UINT64_C(0x94d049bb133111eb);x^=x>>31;return x;}
static double unit52(uint64_t h){return ((double)(h>>12)+0.5)*0x1p-52;}

// Exact frozen PLUTO no-P3 NEON kernel, copied from the generated source.
__attribute__((always_inline)) static inline void pluto_pair(const double* x,double* y){
    float64x2_t xv=vld1q_f64(x); float64x2_t ax=vabsq_f64(xv);
    const float64x2_t magic=vdupq_n_f64(0x1p52);
    float64x2_t qscaled=vmulq_n_f64(ax,INVPI); float64x2_t qmagic=vaddq_f64(qscaled,magic); float64x2_t qd=vsubq_f64(qmagic,magic); uint64x2_t qbits=vreinterpretq_u64_f64(qmagic);
    float64x2_t qp1=vmulq_n_f64(qd,PI_P1); float64x2_t t=vsubq_f64(ax,qp1); float64x2_t rh=vfmaq_n_f64(t,qd,-PI_P2);
    float64x2_t d=vsubq_f64(t,rh); float64x2_t rl=vfmaq_n_f64(d,qd,-PI_P2);
    const uint64x2_t signmask=vdupq_n_u64(UINT64_C(0x8000000000000000));
    uint64x2_t rsign=vandq_u64(vreinterpretq_u64_f64(rh),signmask);
    float64x2_t ah=vreinterpretq_f64_u64(vbicq_u64(vreinterpretq_u64_f64(rh),signmask));
    float64x2_t al=vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(rl),rsign));
    float64x2_t jscaled=vmulq_n_f64(ah,KGRID); float64x2_t jmagic=vaddq_f64(jscaled,magic); float64x2_t jd=vsubq_f64(jmagic,magic); uint64x2_t jbits=vreinterpretq_u64_f64(jmagic);
    constexpr uint64_t JMASK=(UINT64_C(1)<<52)-1; const uint64_t j0=vgetq_lane_u64(jbits,0)&JMASK; const uint64_t j1=vgetq_lane_u64(jbits,1)&JMASK;
    float64x2_t delta=vfmaq_n_f64(ah,jd,NINVK_HI); delta=vfmaq_n_f64(delta,jd,NINVK_LO); delta=vaddq_f64(delta,al);
    float64x2_t a=vld1q_f64(opt_cos53_coeff_aos+2*j0); float64x2_t b=vld1q_f64(opt_cos53_coeff_aos+2*j1); float64x2_t c0=vzip1q_f64(a,b); float64x2_t c1=vzip2q_f64(a,b);
    float64x2_t c2=vmulq_n_f64(c0,MH); float64x2_t c3=vmulq_n_f64(c1,M6); float64x2_t p=vfmaq_f64(c2,c3,delta); p=vfmaq_f64(c1,p,delta); p=vfmaq_f64(c0,p,delta);
    uint64x2_t parity=vandq_u64(qbits,vdupq_n_u64(1)); uint64x2_t outsign=vshlq_n_u64(parity,63); p=vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(p),outsign)); vst1q_f64(y,p);
    if(__builtin_expect(j0>=2009,0))y[0]=std::cos(x[0]); if(__builtin_expect(j1>=2009,0))y[1]=std::cos(x[1]);
}
static inline void pluto_neon(const double* __restrict x,double* __restrict y,size_t n){size_t i=0;for(;i+8<=n;i+=8){pluto_pair(x+i,y+i);pluto_pair(x+i+2,y+i+2);pluto_pair(x+i+4,y+i+4);pluto_pair(x+i+6,y+i+6);}for(;i+2<=n;i+=2)pluto_pair(x+i,y+i);if(i<n){double xx[2]={x[i],x[i]},yy[2];pluto_pair(xx,yy);y[i]=yy[0];}}

// SME integration: the same PLUTO arithmetic in SME streaming mode. On M4 Pro
// the streaming vector length is 512 bits, so this processes 8 binary64 lanes.
// ZA is deliberately not used: PLUTO is lane-wise with indexed LUT loads, so
// forcing an outer-product/tile formulation would change the computation rather
// than accelerate it. SSVE is the elementwise execution engine supplied by SME.
__arm_locally_streaming static size_t pluto_sme_core(const double* __restrict x,double* __restrict y,size_t n,uint64_t* __restrict repairs){
    const svuint64_t signmask=svdup_u64(UINT64_C(0x8000000000000000));
    const svuint64_t oneu=svdup_u64(1), jmask=svdup_u64((UINT64_C(1)<<52)-1);
    size_t nr=0;
    for(size_t i=0;i<n;i+=svcntd()){
        svbool_t pg=svwhilelt_b64((uint64_t)i,(uint64_t)n);
        svfloat64_t xv=svld1_f64(pg,x+i); svfloat64_t ax=svabs_f64_x(pg,xv);
        svfloat64_t qscaled=svmul_n_f64_x(pg,ax,INVPI); svfloat64_t qmagic=svadd_n_f64_x(pg,qscaled,0x1p52); svfloat64_t qd=svsub_n_f64_x(pg,qmagic,0x1p52); svuint64_t qbits=svreinterpret_u64_f64(qmagic);
        svfloat64_t qp1=svmul_n_f64_x(pg,qd,PI_P1); svfloat64_t t=svsub_f64_x(pg,ax,qp1); svfloat64_t rh=svmla_n_f64_x(pg,t,qd,-PI_P2);
        svfloat64_t d=svsub_f64_x(pg,t,rh); svfloat64_t rl=svmla_n_f64_x(pg,d,qd,-PI_P2);
        svuint64_t rhu=svreinterpret_u64_f64(rh); svuint64_t rsign=svand_u64_x(pg,rhu,signmask); svfloat64_t ah=svreinterpret_f64_u64(svbic_u64_x(pg,rhu,signmask)); svfloat64_t al=svreinterpret_f64_u64(sveor_u64_x(pg,svreinterpret_u64_f64(rl),rsign));
        svfloat64_t jscaled=svmul_n_f64_x(pg,ah,KGRID); svfloat64_t jmagic=svadd_n_f64_x(pg,jscaled,0x1p52); svfloat64_t jd=svsub_n_f64_x(pg,jmagic,0x1p52); svuint64_t jidx=svand_u64_x(pg,svreinterpret_u64_f64(jmagic),jmask);
        svfloat64_t delta=svmla_n_f64_x(pg,ah,jd,NINVK_HI); delta=svmla_n_f64_x(pg,delta,jd,NINVK_LO); delta=svadd_f64_x(pg,delta,al);
        svuint64_t idx2=svlsl_n_u64_x(pg,jidx,1); svfloat64_t c0=svld1_gather_u64index_f64(pg,opt_cos53_coeff_aos,idx2); svfloat64_t c1=svld1_gather_u64index_f64(pg,opt_cos53_coeff_aos,svadd_n_u64_x(pg,idx2,1));
        svfloat64_t c2=svmul_n_f64_x(pg,c0,MH); svfloat64_t c3=svmul_n_f64_x(pg,c1,M6); svfloat64_t p=svmla_f64_x(pg,c2,c3,delta); p=svmla_f64_x(pg,c1,p,delta); p=svmla_f64_x(pg,c0,p,delta);
        svuint64_t parity=svand_u64_x(pg,qbits,oneu); svuint64_t outsign=svlsl_n_u64_x(pg,parity,63); p=svreinterpret_f64_u64(sveor_u64_x(pg,svreinterpret_u64_f64(p),outsign)); svst1_f64(pg,y+i,p);
        svbool_t rp=svcmpge_n_u64(pg,jidx,2009); uint64_t cnt=svcntp_b64(pg,rp);
        if(cnt){svuint64_t lane=svindex_u64((uint64_t)i,1); svuint64_t packed=svcompact_u64(rp,lane); svbool_t cp=svwhilelt_b64((uint64_t)0,cnt); svst1_u64(cp,repairs+nr,packed); nr+=cnt;}
    }
    return nr;
}
static inline void pluto_sme(const double* __restrict x,double* __restrict y,size_t n){
    // Worst-case repair list is n entries; allocation is outside the timed inner
    // arithmetic but inside one benchmark call, so use thread-local retained storage.
    static thread_local std::vector<uint64_t> repairs; if(repairs.size()<n)repairs.resize(n);
    size_t nr=pluto_sme_core(x,y,n,repairs.data());
    for(size_t k=0;k<nr;k++){size_t i=(size_t)repairs[k];y[i]=std::cos(x[i]);}
}

static void fill_case(double*x,size_t n,int c){const double lo[3]={0.0,1.0,1000.0},hi[3]={1.0,500.0,10000.0};int b=c/2;uint64_t seed=UINT64_C(2026090501)+(uint64_t)c*UINT64_C(0x9e3779b97f4a7c15)+(uint64_t)n;for(size_t i=0;i<n;i++){double u=unit52(mix64(seed+(uint64_t)i*UINT64_C(0x9e3779b97f4a7c15)));double v=std::fma(hi[b]-lo[b],u,lo[b]);x[i]=(c&1)?-v:v;}}
struct Buffers{size_t n;std::vector<double*>x,y;explicit Buffers(size_t nn):n(nn),x(6),y(6){for(int c=0;c<6;c++){posix_memalign((void**)&x[c],64,n*sizeof(double));posix_memalign((void**)&y[c],64,n*sizeof(double));fill_case(x[c],n,c);}}~Buffers(){for(auto p:x)free(p);for(auto p:y)free(p);}};
static size_t reps_for(size_t n){return std::max<size_t>(2,2000000/n);}
struct NeonRunner{void run(const double*x,double*y,size_t n){pluto_neon(x,y,n);}}; struct SmeRunner{void run(const double*x,double*y,size_t n){pluto_sme(x,y,n);}}; struct AppleRunner{void run(const double*x,double*y,size_t n){int nn=(int)n;vvcos(y,x,&nn);}};
template<class R> static int bench(const char*name,R&r,size_t n){Buffers b(n);size_t reps=reps_for(n);for(int w=0;w<12;w++)for(int c=0;c<6;c++)r.run(b.x[c],b.y[c],n);double c0=process_cpu_ns();uint64_t w0=now_ticks();for(size_t rr=0;rr<reps;rr++)for(int c=0;c<6;c++)r.run(b.x[c],b.y[c],n);uint64_t w1=now_ticks();double c1=process_cpu_ns();double den=(double)reps*n*6.0;double wall=ticks_to_ns(w1-w0)/den,cpu=(c1-c0)/den;g_sink+=b.y[0][(n*7/11)%n];std::printf("PLUTO_SME_RESULT stack=%s n=%zu wall_ns_el=%.9f cpu_ns_el=%.9f effective_cores=%.6f reps=%zu sink=%.17g svl_bytes=%llu\n",name,n,wall,cpu,cpu/wall,reps,(double)g_sink,(unsigned long long)(std::string(name)=="sme"?64:0));return 0;}

#ifdef OPT_VALIDATE_MPFR
static inline uint64_t ordered_bits(double x){uint64_t u;std::memcpy(&u,&x,8);return(u>>63)?~u:(u|UINT64_C(0x8000000000000000));} static inline uint64_t ulpd(double a,double b){if(a==b)return 0;uint64_t x=ordered_bits(a),y=ordered_bits(b);return x>y?x-y:y-x;}
typedef void(*evalfn)(const double*,double*,size_t);
static void score(const char*stack,const char*tag,const std::vector<double>&x,evalfn fn){std::vector<double>o(x.size());fn(x.data(),o.data(),x.size());mpfr_t z,r;mpfr_init2(z,256);mpfr_init2(r,256);uint64_t maxu=0;size_t exact=0,le1=0,le2=0,gt2=0,worst=0;double wr=0;for(size_t i=0;i<x.size();i++){mpfr_set_d(z,x[i],MPFR_RNDN);mpfr_cos(r,z,MPFR_RNDN);double ref=mpfr_get_d(r,MPFR_RNDN);uint64_t u=ulpd(o[i],ref);exact+=(u==0);le1+=(u<=1);le2+=(u<=2);gt2+=(u>2);if(u>maxu){maxu=u;worst=i;wr=ref;}}std::printf("PLUTO_SME_ACCURACY stack=%s tag=%s cases=%zu exact=%zu le1=%zu le2=%zu gt2=%zu maxulp=%llu worst_i=%zu worst_x=%.17g worst_out=%.17g worst_ref=%.17g\n",stack,tag,x.size(),exact,le1,le2,gt2,(unsigned long long)maxu,worst,x[worst],o[worst],wr);mpfr_clear(r);mpfr_clear(z);}
static int validate(){std::vector<double>x(9600);const double lo[3]={0,1,1000},hi[3]={1,500,10000};for(int i=0;i<9600;i++){int c=i%6,b=c/2;double u=unit52(mix64(UINT64_C(2026090619)+(uint64_t)i*UINT64_C(0x9e3779b97f4a7c15)));double v=std::fma(hi[b]-lo[b],u,lo[b]);x[i]=(c&1)?-v:v;}score("neon","9600",x,pluto_neon);score("sme","9600",x,pluto_sme);const int N=1000000,RN=980000;x.resize(N);for(int i=0;i<RN;i++){int c=i%6,b=c/2;double u=unit52(mix64(UINT64_C(2026090621)+(uint64_t)i*UINT64_C(0x9e3779b97f4a7c15)));double v=std::fma(hi[b]-lo[b],u,lo[b]);x[i]=(c&1)?-v:v;}const long double P=acosl(-1.0L);for(int i=RN;i<N;i++){uint64_t h=mix64(UINT64_C(0xd2b74407b1ce6e93)+(uint64_t)i*UINT64_C(0x9e3779b97f4a7c15));int k=(int)(h%6366ULL)-3183;double v=(double)(0.5L*P+(long double)k*P);int steps=(int)((h>>16)%33ULL);double dir=((h>>24)&1ULL)?INFINITY:-INFINITY;for(int j=0;j<steps;j++)v=std::nextafter(v,dir);x[i]=v;}score("sme","1m_stress",x,pluto_sme);return 0;}
#endif
int main(int argc,char**argv){
#ifdef OPT_VALIDATE_MPFR
if(argc==2&&std::string(argv[1])=="validate")return validate();
#endif
if(argc!=3)return 2;std::string mode=argv[1];size_t n=strtoull(argv[2],nullptr,10);pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0);if(mode=="neon"){NeonRunner r;return bench("neon",r,n);}if(mode=="sme"){SmeRunner r;return bench("sme",r,n);}if(mode=="apple"){AppleRunner r;return bench("apple",r,n);}return 3;}
