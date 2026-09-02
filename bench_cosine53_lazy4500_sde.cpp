#define _GNU_SOURCE
#include <mkl.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <pthread.h>
#include <sched.h>

#include "cosine53_batch_production.hpp"
#include "cosine53_batch_lazy_4500_candidate.hpp"

extern "C" int cos53_engine_init(void);
extern "C" void cos53_engine_eval(double *, const double *, size_t);
extern "C" void cos53_engine_cleanup(void);

#ifndef COS53_ENGINE_WIDE
#define COS53_ENGINE_WIDE 0
#endif

static volatile double g_sink=0.0;
extern "C" __attribute__((noinline)) void cos53_profile_start(void){asm volatile("":::"memory");}
extern "C" __attribute__((noinline)) void cos53_profile_stop(void){asm volatile("":::"memory");}
static uint64_t mix64(uint64_t x){x+=UINT64_C(0x9e3779b97f4a7c15);x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);return x^(x>>31);} 
static double u01(uint64_t x){return((double)(mix64(x)>>11)+0.5)*0x1p-53;}
static void pin_current(int cpu){cpu_set_t set;CPU_ZERO(&set);CPU_SET(cpu,&set);(void)pthread_setaffinity_np(pthread_self(),sizeof(set),&set);} 
static void*al64(size_t bytes){void*p=nullptr;if(posix_memalign(&p,64,bytes)!=0)return nullptr;return p;}
static void fill(double*x,size_t n,int c){const double edge=0x1p-20;double lo,hi;if(c<2){lo=edge;hi=1.0-edge;}else if(c<4){lo=1.0+edge;hi=500.0-edge;}else{lo=1000.0+edge;hi=10000.0-edge;}uint64_t seed=UINT64_C(0xd1b54a32d192ed03)^((uint64_t)n*UINT64_C(0x94d049bb133111eb))^((uint64_t)c<<58);for(size_t i=0;i<n;++i){double q=u01(seed+(uint64_t)i*UINT64_C(0x9e3779b97f4a7c15));double v=lo+(hi-lo)*q;x[i]=(c&1)?-v:v;}}
struct Buffers{std::vector<double*>x,y;size_t n=0;int c0=0,c1=0;~Buffers(){for(double*p:x)free(p);for(double*p:y)free(p);}};
static bool alloc_buffers(Buffers&b,size_t n){b.n=n;b.c0=COS53_ENGINE_WIDE?2:0;b.c1=COS53_ENGINE_WIDE?6:2;int nc=b.c1-b.c0;b.x.resize(nc);b.y.resize(nc);for(int j=0;j<nc;++j){b.x[j]=(double*)al64(n*sizeof(double));b.y[j]=(double*)al64(n*sizeof(double));if(!b.x[j]||!b.y[j])return false;fill(b.x[j],n,b.c0+j);}return true;}
static int run_sde(const std::string&which,size_t n){Buffers b;if(!alloc_buffers(b,n))return 10;Cosine53BatchProductionFrozen*prod=nullptr;Cosine53BatchLazy4500Candidate*lazy=nullptr;if(which=="prod")prod=new Cosine53BatchProductionFrozen(cos53_engine_eval);else if(which=="lazy4500")lazy=new Cosine53BatchLazy4500Candidate(cos53_engine_eval);auto once=[&](){for(size_t j=0;j<b.x.size();++j){if(which=="prod")prod->run(b.y[j],b.x[j],n);else if(which=="lazy4500")lazy->run(b.y[j],b.x[j],n);else if(which=="intel")vmdCos((MKL_INT)n,b.x[j],b.y[j],VML_HA);else asm volatile(""::"r"(b.x[j]),"r"(b.y[j]),"r"(n):"memory");}};if(which!="noop")once();cos53_profile_start();once();cos53_profile_stop();if(which!="noop")g_sink+=b.y[0][(n*5u/13u)%n];int helper=lazy?(lazy->helper_started()?1:0):(prod?1:0);printf("L4500_SDE engine=%s stack=%s n=%zu cases=%zu helper_started=%d sink=%.17g\n",COS53_ENGINE_WIDE?"wide":"unit",which.c_str(),n,b.x.size(),helper,(double)g_sink);delete lazy;delete prod;return 0;}
int main(int argc,char**argv){if(argc!=3)return 2;pin_current(0);mkl_set_num_threads_local(1);if(!cos53_engine_init())return 3;std::string which=argv[1];size_t n=(size_t)strtoull(argv[2],nullptr,10);int rc=2;if(which=="prod"||which=="lazy4500"||which=="intel"||which=="noop")rc=run_sde(which,n);cos53_engine_cleanup();return rc;}
