from pathlib import Path
import sys

base=Path(sys.argv[1]).read_text()
harness_out=Path(sys.argv[2])
impl_out=Path(sys.argv[3])

# Harness stays ordinary Apple ARM64/NEON.  Only the isolated implementation TU
# is compiled with SME enabled, avoiding accidental non-streaming SVE codegen on
# Apple chips where SVE exists only inside SME streaming mode.
start=base.index('static inline void opt_cos53_eval(')
end=base.index('\nclass Adaptive2', start)
wrapper=r'''extern "C" size_t pluto_sme_eval_raw(const double*,double*,size_t,uint32_t*);
static inline void opt_cos53_eval(const double* __restrict x,double* __restrict y,size_t n)
{
    thread_local uint32_t* bad_idx=nullptr;
    thread_local size_t bad_cap=0;
    if (__builtin_expect(bad_cap<n,0)) {
        size_t nc=1;
        while(nc<n) nc<<=1;
        void* p=nullptr;
        if(posix_memalign(&p,64,nc*sizeof(uint32_t))!=0) std::abort();
        std::free(bad_idx);
        bad_idx=(uint32_t*)p;
        bad_cap=nc;
    }
    size_t bad_n=pluto_sme_eval_raw(x,y,n,bad_idx);
    for(size_t k=0;k<bad_n;k++) {
        size_t j=bad_idx[k];
        y[j]=std::cos(x[j]);
    }
}
'''
harness=base[:start]+wrapper+base[end:]
harness_out.write_text(harness)

impl=r'''#include <arm_sve.h>
#include <cstddef>
#include <cstdint>
#include "apple_cos53_coeff_aos.h"

static constexpr double INVPI = 0x1.45f306dc9c883p-2;
static constexpr double KGRID = 1280.0;
static constexpr double NINVK_HI = -0x1.999999999999ap-11;
static constexpr double NINVK_LO = 0x1.999999999999ap-65;
static constexpr double PI_P1 = 0x1.921fb54442000p+1;
static constexpr double PI_P2 = 0x1.a308d313198a3p-40;
static constexpr double MH = -0x1.ffffff92c5f94p-2;
static constexpr double M6 = -0x1.5555551eb851fp-3;

static size_t pluto_sme_stream(const double* __restrict x,double* __restrict y,size_t n,uint32_t* bad_idx) __arm_streaming;
static size_t pluto_sme_stream(const double* __restrict x,double* __restrict y,size_t n,uint32_t* bad_idx) __arm_streaming
{
    // Exact frozen PLUTO no-P3 mathematics, widened to streaming SVE/SME.
    const svfloat64_t magic = svdup_n_f64(0x1p52);
    const svuint64_t signmask = svdup_n_u64(UINT64_C(0x8000000000000000));
    const svuint64_t jmask = svdup_n_u64((UINT64_C(1)<<52)-1);
    const size_t vl = svcntd();
    alignas(64) uint64_t jt[32];
    alignas(64) double c0t[32], c1t[32];
    size_t bad_n=0;

    for (size_t i=0; i<n; i+=vl) {
        svbool_t pg = svwhilelt_b64((uint64_t)i,(uint64_t)n);
        const size_t lanes = (n-i < vl) ? (n-i) : vl;
        svfloat64_t xv = svld1_f64(pg,x+i);
        svfloat64_t ax = svabs_f64_x(pg,xv);

        svfloat64_t qscaled = svmul_n_f64_x(pg,ax,INVPI);
        svfloat64_t qmagic = svadd_f64_x(pg,qscaled,magic);
        svfloat64_t qd = svsub_f64_x(pg,qmagic,magic);
        svuint64_t qbits = svreinterpret_u64_f64(qmagic);

        svfloat64_t qp1 = svmul_n_f64_x(pg,qd,PI_P1);
        svfloat64_t t = svsub_f64_x(pg,ax,qp1);
        svfloat64_t rh = svmla_n_f64_x(pg,t,qd,-PI_P2);
        svfloat64_t d = svsub_f64_x(pg,t,rh);
        svfloat64_t rl = svmla_n_f64_x(pg,d,qd,-PI_P2);

        svuint64_t rhbits = svreinterpret_u64_f64(rh);
        svuint64_t rsign = svand_u64_x(pg,rhbits,signmask);
        svfloat64_t ah = svreinterpret_f64_u64(svbic_u64_x(pg,rhbits,signmask));
        svfloat64_t al = svreinterpret_f64_u64(sveor_u64_x(pg,svreinterpret_u64_f64(rl),rsign));

        svfloat64_t jscaled = svmul_n_f64_x(pg,ah,KGRID);
        svfloat64_t jmagic = svadd_f64_x(pg,jscaled,magic);
        svfloat64_t jd = svsub_f64_x(pg,jmagic,magic);
        svuint64_t qjbits = svreinterpret_u64_f64(jmagic);
        svuint64_t ji = svand_u64_x(pg,qjbits,jmask);

        svfloat64_t delta = svmla_n_f64_x(pg,ah,jd,NINVK_HI);
        delta = svmla_n_f64_x(pg,delta,jd,NINVK_LO);
        delta = svadd_f64_x(pg,delta,al);

        // AppleClang currently disallows arbitrary SVE gather intrinsics in a
        // streaming function.  Keep the exact AoS LUT and scalarize only these
        // coefficient addresses; all arithmetic is still full-width SSVE.
        svst1_u64(pg,jt,ji);
        for(size_t k=0;k<lanes;k++) {
            c0t[k]=opt_cos53_coeff_aos[2*jt[k]];
            c1t[k]=opt_cos53_coeff_aos[2*jt[k]+1];
        }
        svfloat64_t c0 = svld1_f64(pg,c0t);
        svfloat64_t c1 = svld1_f64(pg,c1t);

        svfloat64_t c2 = svmul_n_f64_x(pg,c0,MH);
        svfloat64_t c3 = svmul_n_f64_x(pg,c1,M6);
        svfloat64_t p = svmla_f64_x(pg,c2,c3,delta);
        p = svmla_f64_x(pg,c1,p,delta);
        p = svmla_f64_x(pg,c0,p,delta);

        svuint64_t parity = svand_n_u64_x(pg,qbits,1);
        svuint64_t outsign = svlsl_n_u64_x(pg,parity,63);
        p = svreinterpret_f64_u64(sveor_u64_x(pg,svreinterpret_u64_f64(p),outsign));
        svst1_f64(pg,y+i,p);

        for(size_t k=0;k<lanes;k++)
            if(__builtin_expect(jt[k]>=2009,0)) bad_idx[bad_n++]=(uint32_t)(i+k);
    }
    return bad_n;
}

// Non-streaming ABI entry.  Compiler emits the SMSTART/SMSTOP transition here,
// keeping every caller and the benchmark harness in ordinary ARM64 mode.
extern "C" size_t pluto_sme_eval_raw(const double* x,double* y,size_t n,uint32_t* bad_idx)
{
    return pluto_sme_stream(x,y,n,bad_idx);
}
'''
impl_out.write_text(impl)
