#define main s53w_base_main
#include "bench_sine_53_wide_intel.c"
#undef main
#include <mpfr.h>

#define REDN2 4096
static double *pih2,*pil2;

static int redtab2_init(void){
    pih2=al64((size_t)REDN2*sizeof(double));pil2=al64((size_t)REDN2*sizeof(double));if(!pih2||!pil2)return 0;
    mpfr_t pi,t;mpfr_init2(pi,256);mpfr_init2(t,256);mpfr_const_pi(pi,MPFR_RNDN);
    for(unsigned q=0;q<REDN2;q++){mpfr_mul_ui(t,pi,q,MPFR_RNDN);double h=mpfr_get_d(t,MPFR_RNDN);pih2[q]=h;mpfr_sub_d(t,t,h,MPFR_RNDN);pil2[q]=mpfr_get_d(t,MPFR_RNDN);}mpfr_clear(t);mpfr_clear(pi);return 1;
}
static void redtab2_clear(void){free(pil2);free(pih2);pil2=pih2=NULL;}
static inline void twos2(double a,double b,double*h,double*l){double s=a+b,bv=s-a,av=s-bv,br=b-bv,ar=a-av;*h=s;*l=ar+br;}

static inline double scalar2(const s53w_kernel*k,double x){
    double ax=fabs(x);int64_t q=(int64_t)nearbyint(ax*INVPI);if(q<0)q=0;if(q>=REDN2)q=REDN2-1;
    /* q is nearest integer to |x|/pi.  For q>=1, ax and pih2[q] are within
       a factor 2, hence ax-pih2[q] is exact by Sterbenz; q=0 is trivial. */
    double s=ax-pih2[q],rh,rl;twos2(s,-pil2[q],&rh,&rl);int rn=(rh<0.0)||(rh==0.0&&rl<0.0);if(rn){rh=-rh;rl=-rl;}
    double r=rh+rl;long a=lround(r*KGRID);if(a<0)a=0;if(a>=LUTN)a=LUTN-1;double aa=(double)a*INVK;
    /* aa is nearest 1/4096 anchor to rh, so rh-aa is likewise exact. */
    double d=rh-aa;
    double p=k->tab[(size_t)k->deg*LUTN+(size_t)a],dp=0.0;
    for(int j=k->deg-1;j>=0;j--){dp=fma(dp,d,p);p=fma(p,d,k->tab[(size_t)j*LUTN+(size_t)a]);}
    p=fma(rl,dp,p);if(signbit(x)^(int)(q&1)^rn)p=-p;return p;
}

#if defined(__x86_64__) || defined(__i386__)
#define V2 __attribute__((target("avx512f,avx512dq,fma")))
V2 static inline void twos2v(__m512d a,__m512d b,__m512d*h,__m512d*l){__m512d s=_mm512_add_pd(a,b),bv=_mm512_sub_pd(s,a),av=_mm512_sub_pd(s,bv),br=_mm512_sub_pd(b,bv),ar=_mm512_sub_pd(a,av);*h=s;*l=_mm512_add_pd(ar,br);}
V2 static void vector2(const s53w_kernel*k,const double*x,double*y,size_t n){
    const __m512d Z=_mm512_setzero_pd(),VIPI=_mm512_set1_pd(INVPI),VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK);
    const __m512i ONE=_mm512_set1_epi64(1),IZ=_mm512_setzero_si512(),ABSM=_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff));
    for(size_t i=0;i<n;i+=8){unsigned rem=(unsigned)(n-i);__mmask8 active=(__mmask8)(rem>=8?0xffu:((1u<<rem)-1u));
        __m512d vx=_mm512_maskz_loadu_pd(active,x+i);__mmask8 minput=(__mmask8)(_mm512_cmp_pd_mask(vx,Z,_CMP_LT_OQ)&active);
        __m512d ax=_mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(vx),ABSM));
        __m512d qf=_mm512_roundscale_pd(_mm512_mul_pd(ax,VIPI),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);__m512i qi=_mm512_cvttpd_epi64(qf);
        __m512d ph=_mm512_i64gather_pd(qi,pih2,8),pl=_mm512_i64gather_pd(qi,pil2,8);
        __m512d s=_mm512_sub_pd(ax,ph),rh,rl;twos2v(s,_mm512_sub_pd(Z,pl),&rh,&rl);
        __mmask8 rn=_mm512_cmp_pd_mask(rh,Z,_CMP_LT_OQ);__mmask8 rz=_mm512_cmp_pd_mask(rh,Z,_CMP_EQ_OQ);rn|=(__mmask8)(rz&_mm512_cmp_pd_mask(rl,Z,_CMP_LT_OQ));rn&=active;
        rh=_mm512_mask_sub_pd(rh,rn,Z,rh);rl=_mm512_mask_sub_pd(rl,rn,Z,rl);
        __m512d r=_mm512_add_pd(rh,rl);__m512d jd=_mm512_roundscale_pd(_mm512_mul_pd(r,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);__m512i ji=_mm512_cvttpd_epi64(jd);
        __m512d d=_mm512_sub_pd(rh,_mm512_mul_pd(jd,VIK));
        __m512d p=_mm512_i64gather_pd(ji,k->tab+(size_t)k->deg*LUTN,8),dp=Z;
        for(int j=k->deg-1;j>=0;j--){dp=_mm512_fmadd_pd(dp,d,p);__m512d c=_mm512_i64gather_pd(ji,k->tab+(size_t)j*LUTN,8);p=_mm512_fmadd_pd(p,d,c);}
        p=_mm512_fmadd_pd(rl,dp,p);__mmask8 odd=_mm512_cmpneq_epi64_mask(_mm512_and_epi64(qi,ONE),IZ);__mmask8 sg=(__mmask8)((minput^odd^rn)&active);p=_mm512_mask_sub_pd(p,sg,Z,p);_mm512_mask_storeu_pd(y+i,active,p);
    }
}
#endif
static int hav2(void){
#if defined(__x86_64__) || defined(__i386__)
return __builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("avx512dq")&&__builtin_cpu_supports("fma");
#else
return 0;
#endif
}
static void eval2(const s53w_kernel*k,const double*x,double*y,size_t n){
#if defined(__x86_64__) || defined(__i386__)
if(hav2()){vector2(k,x,y,n);return;}
#endif
for(size_t i=0;i<n;i++)y[i]=scalar2(k,x[i]);}

static int verify2(const char*tag,const s53w_kernel*k,const double*x,int n){double*o=al64((size_t)n*sizeof(double)),*in=al64((size_t)n*sizeof(double));if(!o||!in)return 0;eval2(k,x,o,n);vmdSin(n,x,in,VML_HA);arb_t ax,ay;arf_t lo,hi;arb_init(ax);arb_init(ay);arf_init(lo);arf_init(hi);int uq=0,oe=0,o1=0,ie=0,i1=0;uint64_t om=0,im=0;for(int i=0;i<n;i++){arb_set_d(ax,x[i]);arb_sin(ay,ax,256);arb_get_lbound_arf(lo,ay,256);arb_get_ubound_arf(hi,ay,256);double a=arf_get_d(lo,ARF_RND_NEAR),b=arf_get_d(hi,ARF_RND_NEAR);if(dbits(a)!=dbits(b))continue;uq++;uint64_t uo=ulpd(o[i],a),ui=ulpd(in[i],a);if(!uo)oe++;if(uo<=1)o1++;if(uo>om)om=uo;if(!ui)ie++;if(ui<=1)i1++;if(ui>im)im=ui;}printf("S53F2_VERIFY tag=%s cases=%d unique_ref=%d ours_exact=%d ours_le1ulp=%d ours_max_ulp=%lu intel_exact=%d intel_le1ulp=%d intel_max_ulp=%lu reference=Arb256\n",tag,n,uq,oe,o1,(unsigned long)om,ie,i1,(unsigned long)im);arf_clear(hi);arf_clear(lo);arb_clear(ay);arb_clear(ax);free(in);free(o);return uq==n&&om<=1;}
static void bands2(const s53w_kernel*k,const double*x){double o[CASES],in[CASES];eval2(k,x,o,CASES);vmdSin(CASES,x,in,VML_HA);arb_t ax,ay;arf_t lo,hi;arb_init(ax);arb_init(ay);arf_init(lo);arf_init(hi);for(int b=0;b<3;b++){int oe=0,o1=0,ie=0,i1=0;uint64_t om=0,im=0;for(int j=0;j<50;j++){int i=50*b+j;arb_set_d(ax,x[i]);arb_sin(ay,ax,256);arb_get_lbound_arf(lo,ay,256);arb_get_ubound_arf(hi,ay,256);double a=arf_get_d(lo,ARF_RND_NEAR),z=arf_get_d(hi,ARF_RND_NEAR);if(dbits(a)!=dbits(z))continue;uint64_t uo=ulpd(o[i],a),ui=ulpd(in[i],a);if(!uo)oe++;if(uo<=1)o1++;if(uo>om)om=uo;if(!ui)ie++;if(ui<=1)i1++;if(ui>im)im=ui;}const char*bn=b==0?"0_to_1":(b==1?"1_to_500":"1000_to_10000");printf("S53F2_BAND band=%s ours_exact=%d ours_le1ulp=%d ours_max_ulp=%lu intel_exact=%d intel_le1ulp=%d intel_max_ulp=%lu\n",bn,oe,o1,(unsigned long)om,ie,i1,(unsigned long)im);}arf_clear(hi);arf_clear(lo);arb_clear(ay);arb_clear(ax);}
static uint64_t run2(const s53w_kernel*k,const double*x,int rounds,volatile double*s){double y[CASES];uint64_t t=now_ns();for(int r=0;r<rounds;r++)eval2(k,x,y,CASES);t=now_ns()-t;*s+=y[CASES-1];return t;}
static int bench2(const s53w_kernel*k,const double*x){if(!hav2()){printf("S53F2_SKIP_TIMING reason=no_avx512\n");return 0;}volatile double s=0;run2(k,x,5000,&s);run_intel(x,5000,&s);double a[TRIALS],b[TRIALS],calls=(double)ROUNDS*CASES;for(int t=0;t<TRIALS;t++){uint64_t u,v;if(t&1){v=run_intel(x,ROUNDS,&s);u=run2(k,x,ROUNDS,&s);}else{u=run2(k,x,ROUNDS,&s);v=run_intel(x,ROUNDS,&s);}a[t]=(double)u/calls;b[t]=(double)v/calls;printf("S53F2_TRIAL trial=%d ours_ns=%.6f intel_ns=%.6f intel_over_ours=%.6fx\n",t+1,a[t],b[t],b[t]/a[t]);}qsort(a,TRIALS,sizeof(double),cmpd);qsort(b,TRIALS,sizeof(double),cmpd);double om=a[TRIALS/2],im=b[TRIALS/2];printf("S53F2_RESULT terms=%d degree=%d cases=150 ours_e2e_ns_per_input=%.6f intel_ha_ns_per_input=%.6f ours_over_intel=%.6fx intel_over_ours=%.6fx throughput_advantage_pct=%.3f reduction=AVX512_table_DD_sterbenz derivative=single_pass_horner masked_tail=1 reduction_table_build_excluded=1 all_raw_input_work_included=1 formula=unchanged_Mode5_secant_spine accuracy_contract=le1ulp sink=%.17g\n",k->terms,k->deg,om,im,om/im,im/om,(im/om-1.0)*100.0,(double)s);return 0;}

int main(void){int cpu=pin();mkl_set_num_threads_local(1);printf("S53F2_DOMAIN cpu_pin=%d target=binary64_53bit cases=150 bands=0_to_1,1_to_500,1000_to_10000 signs=25pos_25neg_each intel=oneMKL_vmdSin_VML_HA formula=unchanged_Mode5_secant_spine\n",cpu);if(!redtab2_init())return 2;s53w_kernel*k=kernel_create(2);if(!k)return 3;double x[CASES],v[CASES];make_bench(x);eval2(k,x,v,CASES);int same=0;for(int i=0;i<CASES;i++)if(dbits(v[i])==dbits(scalar2(k,x[i])))same++;printf("S53F2_VECTOR_SCALAR bit_identical=%d/150\n",same);if(same!=CASES){kernel_destroy(k);redtab2_clear();return 4;}if(!verify2("requested150",k,x,CASES)){kernel_destroy(k);redtab2_clear();return 5;}bands2(k,x);double*st=al64(STRESS*sizeof(double));if(st){make_stress(st);int ok=verify2("adversarial_stress",k,st,STRESS);printf("S53F2_STRESS pass=%d contractual=0\n",ok);free(st);}int rc=bench2(k,x);kernel_destroy(k);redtab2_clear();flint_cleanup_master();return rc;}
