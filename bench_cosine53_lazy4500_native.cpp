#define _GNU_SOURCE
#include <mkl.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <time.h>
#include <sys/resource.h>
#include <pthread.h>
#include <sched.h>
#include <vector>

#include "cosine53_batch_production.hpp"
#include "cosine53_batch_lazy_4500_candidate.hpp"

extern "C" int cos53_engine_init(void);
extern "C" void cos53_engine_eval(double *, const double *, size_t);
extern "C" void cos53_engine_cleanup(void);

#ifndef COS53_ENGINE_WIDE
#define COS53_ENGINE_WIDE 0
#endif

static volatile double g_sink = 0.0;
static const size_t kSizes[] = {100,700,3500,4500,5000,8000,15000,50000,1000000,2000000};

static uint64_t mix64(uint64_t x){x+=UINT64_C(0x9e3779b97f4a7c15);x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);return x^(x>>31);} 
static double u01(uint64_t x){return ((double)(mix64(x)>>11)+0.5)*0x1p-53;}
static void pin_current(int cpu){cpu_set_t set;CPU_ZERO(&set);CPU_SET(cpu,&set);(void)pthread_setaffinity_np(pthread_self(),sizeof(set),&set);} 
static void *al64(size_t bytes){void*p=nullptr;if(posix_memalign(&p,64,bytes)!=0)return nullptr;return p;}
static double clock_ns(clockid_t id){timespec t;clock_gettime(id,&t);return(double)t.tv_sec*1e9+(double)t.tv_nsec;}
static void fill(double*x,size_t n,int c){const double edge=0x1p-20;double lo,hi;if(c<2){lo=edge;hi=1.0-edge;}else if(c<4){lo=1.0+edge;hi=500.0-edge;}else{lo=1000.0+edge;hi=10000.0-edge;}uint64_t seed=UINT64_C(0xd1b54a32d192ed03)^((uint64_t)n*UINT64_C(0x94d049bb133111eb))^((uint64_t)c<<58);for(size_t i=0;i<n;++i){double q=u01(seed+(uint64_t)i*UINT64_C(0x9e3779b97f4a7c15));double v=lo+(hi-lo)*q;x[i]=(c&1)?-v:v;}}
static uint64_t ordered_bits(double x){uint64_t u;memcpy(&u,&x,sizeof(u));if(u>>63)return~u;return u|UINT64_C(0x8000000000000000);} 
static uint64_t ulp_distance(double a,double b){if(std::isnan(a)||std::isnan(b))return UINT64_MAX;uint64_t x=ordered_bits(a),y=ordered_bits(b);return x>y?x-y:y-x;}

struct Buffers{std::vector<double*>x,y,ref;size_t n=0;int c0=0,c1=0;~Buffers(){for(double*p:x)free(p);for(double*p:y)free(p);for(double*p:ref)free(p);}};
static bool alloc_buffers(Buffers&b,size_t n){b.n=n;b.c0=COS53_ENGINE_WIDE?2:0;b.c1=COS53_ENGINE_WIDE?6:2;int nc=b.c1-b.c0;b.x.resize(nc);b.y.resize(nc);b.ref.resize(nc);for(int j=0;j<nc;++j){b.x[j]=(double*)al64(n*sizeof(double));b.y[j]=(double*)al64(n*sizeof(double));b.ref[j]=(double*)al64(n*sizeof(double));if(!b.x[j]||!b.y[j]||!b.ref[j])return false;fill(b.x[j],n,b.c0+j);}return true;}
static size_t reps_for(size_t n,size_t cases){const size_t target=4000000;size_t r=target/(n*cases);if(r<1)r=1;if(r>20000)r=20000;return r;}
static double median5(double v[5]){std::sort(v,v+5);return v[2];}

static int run_native(const std::string&which){
    Cosine53BatchProductionFrozen*prod=nullptr;Cosine53BatchLazy4500Candidate*lazy=nullptr;
    if(which=="prod")prod=new Cosine53BatchProductionFrozen(cos53_engine_eval);else if(which=="lazy4500")lazy=new Cosine53BatchLazy4500Candidate(cos53_engine_eval);
    for(size_t n:kSizes){
        Buffers b;if(!alloc_buffers(b,n)){delete lazy;delete prod;return 10;}size_t cases=b.x.size(),reps=reps_for(n,cases);
        auto once=[&](){for(size_t j=0;j<cases;++j){if(which=="prod")prod->run(b.y[j],b.x[j],n);else if(which=="lazy4500")lazy->run(b.y[j],b.x[j],n);else vmdCos((MKL_INT)n,b.x[j],b.y[j],VML_HA);}};
        uint64_t maxulp=0;if(which!="intel"){once();for(size_t j=0;j<cases;++j)vmdCos((MKL_INT)n,b.x[j],b.ref[j],VML_HA);for(size_t j=0;j<cases;++j)for(size_t i=0;i<n;++i)maxulp=std::max(maxulp,ulp_distance(b.y[j][i],b.ref[j][i]));}
        for(int w=0;w<3;++w)once();double wall[5],cpu[5];for(int t=0;t<5;++t){double w0=clock_ns(CLOCK_MONOTONIC_RAW),c0=clock_ns(CLOCK_PROCESS_CPUTIME_ID);for(size_t r=0;r<reps;++r)once();double c1=clock_ns(CLOCK_PROCESS_CPUTIME_ID),w1=clock_ns(CLOCK_MONOTONIC_RAW);double denom=(double)reps*(double)n*(double)cases;wall[t]=(w1-w0)/denom;cpu[t]=(c1-c0)/denom;g_sink+=b.y[(size_t)t%cases][(n*7u/11u)%n];}
        rusage ru{};getrusage(RUSAGE_SELF,&ru);double mw=median5(wall),mc=median5(cpu);int helper=lazy?(lazy->helper_started()?1:0):(prod?1:0);
        printf("L4500_NATIVE engine=%s stack=%s n=%zu cases=%zu reps=%zu wall_ns_el=%.9f cpu_ns_el=%.9f effective_cores=%.6f maxrss_kib=%ld helper_started=%d maxulp_vs_intel=%llu\n",COS53_ENGINE_WIDE?"wide":"unit",which.c_str(),n,cases,reps,mw,mc,mw?mc/mw:0.0,ru.ru_maxrss,helper,(unsigned long long)maxulp);
    }
    delete lazy;delete prod;printf("L4500_NATIVE_DONE engine=%s stack=%s sink=%.17g\n",COS53_ENGINE_WIDE?"wide":"unit",which.c_str(),(double)g_sink);return 0;
}

int main(int argc,char**argv){if(argc!=2)return 2;pin_current(0);mkl_set_num_threads_local(1);if(!cos53_engine_init())return 3;std::string which=argv[1];int rc=2;if(which=="prod"||which=="lazy4500"||which=="intel")rc=run_native(which);cos53_engine_cleanup();return rc;}
