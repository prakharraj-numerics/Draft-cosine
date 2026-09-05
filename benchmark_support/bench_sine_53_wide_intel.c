#define _GNU_SOURCE
#include "sine_53_coeff_source.c"
#include <flint/arb.h>
#include <mkl.h>
#include <immintrin.h>
#include <math.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CASES 150
#define STRESS 32768
#define TRIALS 11
#define ROUNDS 200000
#define LUTN SF_LUT_N
#define KGRID 4096.0
#define INVK (1.0/4096.0)

/* Nearest binary64 pi plus the exact residual pi-pi_hi, used as a split. */
#define PI_HI 0x1.921fb54442d18p+1
#define PI_LO 0x1.1a62633145c07p-53
#define HALFPI_HI 0x1.921fb54442d18p+0
#define INVPI 0x1.45f306dc9c883p-2

typedef struct { sine_fixed_ctx *ctx; int terms,deg; double *tab; } s53w_kernel;

static uint64_t now_ns(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return(uint64_t)t.tv_sec*UINT64_C(1000000000)+(uint64_t)t.tv_nsec;}
static int cmpd(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return(x>y)-(x<y);}
static uint64_t mix64(uint64_t x){x^=x>>30;x*=UINT64_C(0xbf58476d1ce4e5b9);x^=x>>27;x*=UINT64_C(0x94d049bb133111eb);x^=x>>31;return x;}
static int pin(void){cpu_set_t s,o;if(sched_getaffinity(0,sizeof(s),&s))return-1;int c=-1;for(int i=0;i<CPU_SETSIZE;i++)if(CPU_ISSET(i,&s)){c=i;break;}CPU_ZERO(&o);if(c>=0)CPU_SET(c,&o);return c<0||sched_setaffinity(0,sizeof(o),&o)?-1:c;}
static void *al64(size_t n){void*p=NULL;if(posix_memalign(&p,64,n?n:64))return NULL;return p;}
static uint64_t dbits(double x){uint64_t u;memcpy(&u,&x,8);return u;}
static uint64_t ordered_bits(double x){uint64_t u=dbits(x);return(u>>63)?~u:(u|UINT64_C(0x8000000000000000));}
static uint64_t ulpd(double a,double b){if(a==b)return 0;uint64_t x=ordered_bits(a),y=ordered_bits(b);return x>y?x-y:y-x;}

static double coeff_to_double(const mp_limb_t q[2],int neg){long double v=ldexpl((long double)q[1],64-103)+ldexpl((long double)q[0],-103);double d=(double)v;return neg?-d:d;}
static s53w_kernel *kernel_create(int terms){if(terms<1||terms>3)return NULL;s53w_kernel*k=calloc(1,sizeof(*k));if(!k)return NULL;k->ctx=s53_coeff_create_terms(terms);if(!k->ctx){free(k);return NULL;}k->terms=terms;k->deg=k->ctx->poly_deg;k->tab=al64((size_t)(k->deg+1)*LUTN*sizeof(double));if(!k->tab){s53_coeff_destroy(k->ctx);free(k);return NULL;}for(int a=0;a<LUTN;a++){size_t off=(size_t)a*(size_t)(k->deg+1);for(int j=0;j<=k->deg;j++)k->tab[(size_t)j*LUTN+(size_t)a]=coeff_to_double(k->ctx->coef+2*(off+(size_t)j),k->ctx->coef_sign[off+(size_t)j]!=0);}return k;}
static void kernel_destroy(s53w_kernel*k){if(!k)return;free(k->tab);s53_coeff_destroy(k->ctx);free(k);}

static inline double reduce_scalar(double x,int64_t *qout){double ax=fabs(x);int64_t q=(int64_t)(ax*INVPI);double qd=(double)q;double r=fma(-qd,PI_HI,ax);r=fma(-qd,PI_LO,r);if(r<0.0){q--;r+=PI_HI;r+=PI_LO;}else if(r>PI_HI){q++;r-=PI_HI;r-=PI_LO;}if(r>HALFPI_HI){r=(PI_HI-r)+PI_LO;}*qout=q;return r;}
static inline double eval_scalar_one(const s53w_kernel*k,double x){int64_t q;double r=reduce_scalar(x,&q);long a=lround(r*KGRID);if(a<0)a=0;if(a>=LUTN)a=LUTN-1;double d=fma(-(double)a,INVK,r);double y=k->tab[(size_t)k->deg*LUTN+(size_t)a];for(int j=k->deg-1;j>=0;j--)y=fma(y,d,k->tab[(size_t)j*LUTN+(size_t)a]);if(signbit(x)^(int)(q&1))y=-y;return y;}

#if defined(__x86_64__) || defined(__i386__)
#define T512 __attribute__((target("avx512f,avx512dq,fma")))
T512 static void eval_e2e_avx512(const s53w_kernel*k,const double*x,double*y,size_t n){
    const __m512d Z=_mm512_setzero_pd(),VPI=_mm512_set1_pd(PI_HI),VPLO=_mm512_set1_pd(PI_LO),VHPI=_mm512_set1_pd(HALFPI_HI),VIPI=_mm512_set1_pd(INVPI),VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK);
    const __m512i ONE=_mm512_set1_epi64(1),IZ=_mm512_setzero_si512(),ABSM=_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff));
    size_t i=0;
    for(;i+8<=n;i+=8){
        __m512d vx=_mm512_loadu_pd(x+i);__mmask8 minput=_mm512_cmp_pd_mask(vx,Z,_CMP_LT_OQ);
        __m512d ax=_mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(vx),ABSM));
        __m512d qe=_mm512_mul_pd(ax,VIPI);__m512i qi=_mm512_cvttpd_epi64(qe);__m512d qd=_mm512_cvtepi64_pd(qi);
        __m512d r=_mm512_fnmadd_pd(qd,VPI,ax);r=_mm512_fnmadd_pd(qd,VPLO,r);
        __mmask8 mn=_mm512_cmp_pd_mask(r,Z,_CMP_LT_OQ);r=_mm512_mask_add_pd(r,mn,r,VPI);r=_mm512_mask_add_pd(r,mn,r,VPLO);qi=_mm512_mask_sub_epi64(qi,mn,qi,ONE);
        __mmask8 mh=_mm512_cmp_pd_mask(r,VPI,_CMP_GT_OQ);r=_mm512_mask_sub_pd(r,mh,r,VPI);r=_mm512_mask_sub_pd(r,mh,r,VPLO);qi=_mm512_mask_add_epi64(qi,mh,qi,ONE);
        __mmask8 mr=_mm512_cmp_pd_mask(r,VHPI,_CMP_GT_OQ);__m512d rr=_mm512_add_pd(_mm512_sub_pd(VPI,r),VPLO);r=_mm512_mask_mov_pd(r,mr,rr);
        __m512d jd=_mm512_roundscale_pd(_mm512_mul_pd(r,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);__m512i ji=_mm512_cvttpd_epi64(jd);
        __m512d d=_mm512_fnmadd_pd(jd,VIK,r);__m512d out=_mm512_i64gather_pd(ji,k->tab+(size_t)k->deg*LUTN,8);
        for(int j=k->deg-1;j>=0;j--){__m512d c=_mm512_i64gather_pd(ji,k->tab+(size_t)j*LUTN,8);out=_mm512_fmadd_pd(out,d,c);}
        __m512i odd=_mm512_and_epi64(qi,ONE);__mmask8 modd=_mm512_cmpneq_epi64_mask(odd,IZ);__mmask8 msign=(__mmask8)(minput^modd);out=_mm512_mask_sub_pd(out,msign,Z,out);
        _mm512_storeu_pd(y+i,out);
    }
    for(;i<n;i++)y[i]=eval_scalar_one(k,x[i]);
}
#endif

static const char *backend(void){
#if defined(__x86_64__) || defined(__i386__)
if(__builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("avx512dq")&&__builtin_cpu_supports("fma"))return "avx512-wide-binary64-fma";
#endif
return "scalar-wide-binary64-fma";}
static void eval_e2e(const s53w_kernel*k,const double*x,double*y,size_t n){
#if defined(__x86_64__) || defined(__i386__)
if(__builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("avx512dq")&&__builtin_cpu_supports("fma")){eval_e2e_avx512(k,x,y,n);return;}
#endif
for(size_t i=0;i<n;i++)y[i]=eval_scalar_one(k,x[i]);}
static double unit52(uint64_t h){uint64_t m=h>>12;return((double)m+0.5)*0x1p-52;}
static void make_bench(double*x){const double lo[3]={0.0,1.0,1000.0},hi[3]={1.0,500.0,10000.0};for(int b=0;b<3;b++)for(int j=0;j<50;j++){int i=b*50+j;uint64_t h=mix64(UINT64_C(2026082803)+(uint64_t)i*UINT64_C(0x9e3779b97f4a7c15));double u=unit52(h);double mag=fma(hi[b]-lo[b],u,lo[b]);x[i]=j<25?mag:-mag;}}
static void make_stress(double*x){for(int i=0;i<STRESS;i++){uint64_t h=mix64(UINT64_C(2026082891)+(uint64_t)i*UINT64_C(0x9e3779b97f4a7c15));if(i<16384){int b=i%3;double lo=b==0?0.0:(b==1?1.0:1000.0),hi=b==0?1.0:(b==1?500.0:10000.0);double v=fma(hi-lo,unit52(h),lo);x[i]=(h&1)?-v:v;}else{uint64_t k=(h>>16)%3183U;double base=(i&1)?((double)k*PI_HI):(((double)k+0.5)*PI_HI);switch((i>>1)&3){case 0:break;case 1:base=nextafter(base,INFINITY);break;case 2:base=nextafter(base,-INFINITY);break;default:base=nextafter(nextafter(base,INFINITY),INFINITY);break;}if(base>10000.0)base=10000.0;x[i]=(h&2)?-base:base;}}x[0]=0.0;x[1]=nextafter(0.0,1.0);x[2]=nextafter(1.0,0.0);x[3]=1.0;x[4]=500.0;x[5]=1000.0;x[6]=nextafter(10000.0,0.0);x[7]=-nextafter(10000.0,0.0);}

static int verify_array(const char*tag,const s53w_kernel*k,const double*x,int n,double*ours,double*intel){eval_e2e(k,x,ours,(size_t)n);vmdSin(n,x,intel,VML_HA);arb_t ax,ay;arf_t lo,hi;arb_init(ax);arb_init(ay);arf_init(lo);arf_init(hi);uint64_t mo=0,mi=0;int eo=0,ei=0,o1=0,i1=0,unique=0;for(int i=0;i<n;i++){arb_set_d(ax,x[i]);arb_sin(ay,ax,256);arb_get_lbound_arf(lo,ay,256);arb_get_ubound_arf(hi,ay,256);double rl=arf_get_d(lo,ARF_RND_NEAR),rh=arf_get_d(hi,ARF_RND_NEAR);if(dbits(rl)!=dbits(rh))continue;unique++;uint64_t uo=ulpd(ours[i],rl),ui=ulpd(intel[i],rl);if(uo>mo)mo=uo;if(ui>mi)mi=ui;if(!uo)eo++;if(!ui)ei++;if(uo<=1)o1++;if(ui<=1)i1++;}printf("S53W_VERIFY tag=%s terms=%d degree=%d cases=%d unique_ref=%d ours_exact=%d ours_le1ulp=%d ours_max_ulp=%lu intel_exact=%d intel_le1ulp=%d intel_max_ulp=%lu reference=Arb256\n",tag,k->terms,k->deg,n,unique,eo,o1,(unsigned long)mo,ei,i1,(unsigned long)mi);arf_clear(hi);arf_clear(lo);arb_clear(ay);arb_clear(ax);return unique==n&&mo<=1;}

static void verify_bands(const s53w_kernel*k,const double*x){double o[CASES],in[CASES];eval_e2e(k,x,o,CASES);vmdSin(CASES,x,in,VML_HA);arb_t ax,ay;arf_t lo,hi;arb_init(ax);arb_init(ay);arf_init(lo);arf_init(hi);for(int b=0;b<3;b++){int exo=0,exi=0,o1=0,i1=0;uint64_t mo=0,mi=0;for(int j=0;j<50;j++){int i=50*b+j;arb_set_d(ax,x[i]);arb_sin(ay,ax,256);arb_get_lbound_arf(lo,ay,256);arb_get_ubound_arf(hi,ay,256);double rl=arf_get_d(lo,ARF_RND_NEAR),rh=arf_get_d(hi,ARF_RND_NEAR);if(dbits(rl)!=dbits(rh))continue;uint64_t uo=ulpd(o[i],rl),ui=ulpd(in[i],rl);if(!uo)exo++;if(!ui)exi++;if(uo<=1)o1++;if(ui<=1)i1++;if(uo>mo)mo=uo;if(ui>mi)mi=ui;}const char*name=b==0?"0_to_1":(b==1?"1_to_500":"1000_to_10000");printf("S53W_BAND band=%s cases=50 positive=25 negative=25 ours_exact=%d ours_le1ulp=%d ours_max_ulp=%lu intel_exact=%d intel_le1ulp=%d intel_max_ulp=%lu\n",name,exo,o1,(unsigned long)mo,exi,i1,(unsigned long)mi);}arf_clear(hi);arf_clear(lo);arb_clear(ay);arb_clear(ax);}

static uint64_t run_ours(const s53w_kernel*k,const double*x,int rounds,volatile double*sink){double y[CASES];uint64_t t=now_ns();for(int r=0;r<rounds;r++)eval_e2e(k,x,y,CASES);t=now_ns()-t;*sink+=y[CASES-1];return t;}
static uint64_t run_intel(const double*x,int rounds,volatile double*sink){double y[CASES];uint64_t t=now_ns();for(int r=0;r<rounds;r++)vmdSin(CASES,x,y,VML_HA);t=now_ns()-t;*sink+=y[CASES-1];return t;}

static int benchmark(const s53w_kernel*k,const double*x){if(strcmp(backend(),"avx512-wide-binary64-fma")!=0){printf("S53W_SKIP_TIMING reason=no_avx512 backend=%s\n",backend());return 0;}volatile double sink=0;run_ours(k,x,5000,&sink);run_intel(x,5000,&sink);double ot[TRIALS],it[TRIALS],calls=(double)ROUNDS*CASES;for(int t=0;t<TRIALS;t++){uint64_t a,b;if(t&1){b=run_intel(x,ROUNDS,&sink);a=run_ours(k,x,ROUNDS,&sink);}else{a=run_ours(k,x,ROUNDS,&sink);b=run_intel(x,ROUNDS,&sink);}ot[t]=(double)a/calls;it[t]=(double)b/calls;printf("S53W_TRIAL trial=%d ours_e2e_ns_per_input=%.6f intel_ha_ns_per_input=%.6f ours_over_intel=%.6fx intel_over_ours=%.6fx\n",t+1,ot[t],it[t],ot[t]/it[t],it[t]/ot[t]);}qsort(ot,TRIALS,sizeof(double),cmpd);qsort(it,TRIALS,sizeof(double),cmpd);double om=ot[TRIALS/2],im=it[TRIALS/2],sp=im/om;printf("S53W_RESULT terms=%d degree=%d target=binary64_53bit accuracy_contract=le1ulp cases=150 bands=50_50_50 signs=25pos_25neg_each ours_e2e_ns_per_150=%.3f ours_e2e_ns_per_input=%.6f intel_ha_ns_per_150=%.3f intel_ha_ns_per_input=%.6f ours_over_intel=%.6fx intel_over_ours=%.6fx throughput_advantage_pct=%.3f range_reduction=included coeff_gather=included formula=unchanged_Mode5_secant_spine arithmetic=binary64_AVX512_FMA lut_anchors=%d backend=%s sink=%.17g\n",k->terms,k->deg,om*CASES,om,im*CASES,im,om/im,sp,(sp-1.0)*100.0,LUTN,backend(),(double)sink);return 0;}

int main(void){int cpu=pin();mkl_set_num_threads_local(1);printf("S53W_DOMAIN target=binary64_53bit bands=0_to_1,1_to_500,1000_to_10000 cases_each=50 signs_each_band=25pos_25neg vector_input=150 K=12 lut_anchors=%d cpu_pin=%d intel=oneMKL_vmdSin_VML_HA backend=%s certification=offline_Arb256 formula=unchanged_Mode5_secant_spine range_reduction=mod_pi_reflect_halfpi\n",LUTN,cpu,backend());double*stress=al64(STRESS*sizeof(double)),*ours=al64(STRESS*sizeof(double)),*intel=al64(STRESS*sizeof(double)),bench[CASES];if(!stress||!ours||!intel)return 2;make_stress(stress);int winner=0;for(int terms=1;terms<=3;terms++){s53w_kernel*k=kernel_create(terms);if(!k)return 3;int ok=verify_array("wide_stress",k,stress,STRESS,ours,intel);printf("S53W_PROFILE terms=%d degree=%d accepted=%d criterion=ours_max_ulp_le_1\n",terms,k->deg,ok);kernel_destroy(k);if(ok&&!winner)winner=terms;}if(!winner){fprintf(stderr,"No wide profile met <=1 ULP\n");return 4;}printf("S53W_WINNER terms=%d degree=%d selection=minimal_verified_profile\n",winner,2*winner+1);s53w_kernel*k=kernel_create(winner);if(!k)return 5;make_bench(bench);if(!verify_array("benchmark150",k,bench,CASES,ours,intel)){kernel_destroy(k);return 6;}verify_bands(k,bench);int rc=benchmark(k,bench);kernel_destroy(k);free(intel);free(ours);free(stress);flint_cleanup_master();return rc;}
