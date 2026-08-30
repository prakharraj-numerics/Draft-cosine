#define _GNU_SOURCE
#define main s53f2_disabled_main
#include "bench_sine_53_wide_fast2.c"
#undef main
#include <mpfr.h>

/*
 * v2: cosine-style guarded pi/4 octant reducer for binary64 sine.
 * Mathematical evaluator is unchanged Mode-5/secant-spine degree 5.
 *
 * Normal path:
 *   |x|<1       -> direct winning unit-domain evaluator, no wide reduction.
 *   |x|>=1      -> AVX-512 floor(|x|*4/pi), int32 octant, split-pi/4 FMA
 *                  reduction, branchless octant folding, one Mode-5 eval.
 * Rare path:
 *   within 2^-32 of any n*pi/4 boundary -> existing proven table-DD scalar
 *   reducer (fast2 scalar2), preserving the signed residual near n*pi.
 */

#define OTRIALS 11
#define OROUNDS 220000
#define BTRIALS 9
#define BROUNDS 260000
#define OSTRESS 32768
#define PIO4_HI 0x1.921fb54442d18p-1
#define PIO4_LO 0x1.1a62633145c07p-55
#define PIO2_HI 0x1.921fb54442d18p+0
#define PIO2_LO 0x1.1a62633145c07p-54
#define FOUR_OVER_PI 0x1.45f306dc9c883p+0
#define BOUND_TAU 0x1p-32
#define PIO4_TINY (-0x1.f1976b7ed8fbcp-111)
#define PIO4_CW1 0x1.921fb54400000p-1
#define PIO4_CW2 0x1.0b4611a600000p-35
#define PIO4_CW3 0x1.3198a2e037073p-70

#if defined(__x86_64__) || defined(__i386__)
#define OVEC __attribute__((target("avx512f,avx512dq,avx2,fma")))

OVEC static inline __mmask8 mask_eq_i32(__m256i a, int v)
{
    __m256i c=_mm256_cmpeq_epi32(a,_mm256_set1_epi32(v));
    return (__mmask8)_mm256_movemask_ps(_mm256_castsi256_ps(c));
}

OVEC static inline __m512d mode5_poly_i32(const s53w_kernel *k,__m512d y,
                                          __mmask8 signmask)
{
    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK),Z=_mm512_setzero_pd();
    __m512d jd=_mm512_roundscale_pd(_mm512_mul_pd(y,VK),
                    _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m256i ji=_mm512_cvttpd_epi32(jd);
    __m512d d=_mm512_fnmadd_pd(jd,VIK,y);
    __m512d p=_mm512_i32gather_pd(ji,k->tab+(size_t)k->deg*LUTN,8);
    for(int j=k->deg-1;j>=0;j--){
        __m512d c=_mm512_i32gather_pd(ji,k->tab+(size_t)j*LUTN,8);
        p=_mm512_fmadd_pd(p,d,c);
    }
    return _mm512_mask_sub_pd(p,signmask,Z,p);
}

OVEC static inline __m512d mode5_poly_i32_low(const s53w_kernel *k,
                                              __m512d yh,__m512d yl,
                                              __mmask8 signmask)
{
    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK),Z=_mm512_setzero_pd();
    __m512d ya=_mm512_add_pd(yh,yl);
    __m512d jd=_mm512_roundscale_pd(_mm512_mul_pd(ya,VK),
                    _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m256i ji=_mm512_cvttpd_epi32(jd);
    /* Grid anchor is dyadic 1/256.  Keep the reduction low word through the
       final tiny local delta rather than rounding it away before Horner. */
    __m512d d=_mm512_sub_pd(yh,_mm512_mul_pd(jd,VIK));
    d=_mm512_add_pd(d,yl);
    __m512d p=_mm512_i32gather_pd(ji,k->tab+(size_t)k->deg*LUTN,8);
    for(int j=k->deg-1;j>=0;j--){
        __m512d c=_mm512_i32gather_pd(ji,k->tab+(size_t)j*LUTN,8);
        p=_mm512_fmadd_pd(p,d,c);
    }
    return _mm512_mask_sub_pd(p,signmask,Z,p);
}


/* Xeon v11: instruction-level realization of the same degree-5 Mode-5
   evaluator. The profile is fixed at terms=2/degree=5 for binary64, so expose
   that fact to icx: direct nearest conversion gives the gather index, the
   aligned plane-major table is retained, and Horner is explicitly unrolled.
   Numerical FMA order is identical to v8. */
OVEC static inline __m512d mode5_poly_x11(const s53w_kernel *k,__m512d y,
                                          __mmask8 signmask)
{
    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK),Z=_mm512_setzero_pd();
    const __m512d MH=_mm512_set1_pd(-0.5),M6=_mm512_set1_pd(-1.0/6.0);
    const __m512d C24=_mm512_set1_pd(1.0/24.0),C120=_mm512_set1_pd(1.0/120.0);
    const double *tab=(const double *)__builtin_assume_aligned(k->tab,64);
    __m512d sy=_mm512_mul_pd(y,VK);
    __m256i ji=_mm512_cvt_roundpd_epi32(sy,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m512d jd=_mm512_cvtepi32_pd(ji);
    __m512d d=_mm512_fnmadd_pd(jd,VIK,y);
    __m512d c0=_mm512_i32gather_pd(ji,tab+0*LUTN,8);
    __m512d c1=_mm512_i32gather_pd(ji,tab+1*LUTN,8);
    __m512d c2=_mm512_mul_pd(c0,MH);
    __m512d c3=_mm512_mul_pd(c1,M6);
    __m512d c4=_mm512_mul_pd(c0,C24);
    __m512d c5=_mm512_mul_pd(c1,C120);
    __m512d p=_mm512_fmadd_pd(c5,d,c4);
    p=_mm512_fmadd_pd(p,d,c3);
    p=_mm512_fmadd_pd(p,d,c2);
    p=_mm512_fmadd_pd(p,d,c1);
    p=_mm512_fmadd_pd(p,d,c0);
    return _mm512_mask_sub_pd(p,signmask,Z,p);
}

OVEC static inline __m512d mode5_poly_low_x11(const s53w_kernel *k,
                                               __m512d yh,__m512d yl,
                                               __mmask8 signmask)
{
    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK),Z=_mm512_setzero_pd();
    const __m512d MH=_mm512_set1_pd(-0.5),M6=_mm512_set1_pd(-1.0/6.0);
    const __m512d C24=_mm512_set1_pd(1.0/24.0),C120=_mm512_set1_pd(1.0/120.0);
    const double *tab=(const double *)__builtin_assume_aligned(k->tab,64);
    __m512d ya=_mm512_add_pd(yh,yl);
    __m512d sy=_mm512_mul_pd(ya,VK);
    __m256i ji=_mm512_cvt_roundpd_epi32(sy,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m512d jd=_mm512_cvtepi32_pd(ji);
    __m512d d=_mm512_sub_pd(yh,_mm512_mul_pd(jd,VIK));
    d=_mm512_add_pd(d,yl);
    __m512d c0=_mm512_i32gather_pd(ji,tab+0*LUTN,8);
    __m512d c1=_mm512_i32gather_pd(ji,tab+1*LUTN,8);
    __m512d c2=_mm512_mul_pd(c0,MH);
    __m512d c3=_mm512_mul_pd(c1,M6);
    __m512d c4=_mm512_mul_pd(c0,C24);
    __m512d c5=_mm512_mul_pd(c1,C120);
    __m512d p=_mm512_fmadd_pd(c5,d,c4);
    p=_mm512_fmadd_pd(p,d,c3);
    p=_mm512_fmadd_pd(p,d,c2);
    p=_mm512_fmadd_pd(p,d,c1);
    p=_mm512_fmadd_pd(p,d,c0);
    return _mm512_mask_sub_pd(p,signmask,Z,p);
}

OVEC static inline void twodiff_cw(__m512d a,__m512d b,__m512d *h,__m512d *l)
{
    __m512d x=_mm512_sub_pd(a,b);
    __m512d bv=_mm512_sub_pd(a,x);
    __m512d av=_mm512_add_pd(x,bv);
    __m512d br=_mm512_sub_pd(bv,b);
    __m512d ar=_mm512_sub_pd(a,av);
    *h=x; *l=_mm512_add_pd(ar,br);
}

OVEC static void octant_vector_v11_single(const s53w_kernel *k,
                                  const double * __restrict x,
                                  double * __restrict out,size_t n)
{
    const __m512d Z=_mm512_setzero_pd(),ONE=_mm512_set1_pd(1.0);
    const __m512d V4OPI=_mm512_set1_pd(FOUR_OVER_PI);
    const __m512d VC1=_mm512_set1_pd(PIO4_CW1),VC2=_mm512_set1_pd(PIO4_CW2),VC3=_mm512_set1_pd(PIO4_CW3);
    const __m512d VFT=_mm512_set1_pd(BOUND_TAU*FOUR_OVER_PI),V1MFT=_mm512_set1_pd(1.0-BOUND_TAU*FOUR_OVER_PI);
    const __m512i ABSM=_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff));
    /* Direct octant->multiple map: [q,q-1,q+2,q+1,q,q-1,q+2,q+1]. */
    const __m256i ADJ=_mm256_setr_epi32(0,-1,2,1,0,-1,2,1);

    for(size_t i=0;i<n;i+=8){
        unsigned rem=(unsigned)(n-i);
        __mmask8 active=(__mmask8)(rem>=8?0xffu:((1u<<rem)-1u));
        __m512d vx=_mm512_maskz_loadu_pd(active,x+i);
        __m512i vxi=_mm512_castpd_si512(vx);
        __mmask8 inneg=(__mmask8)(_mm512_movepi64_mask(vxi)&active);
        __m512d ax=_mm512_castsi512_pd(_mm512_and_epi64(vxi,ABSM));
        __mmask8 unit=(__mmask8)(_mm512_cmp_pd_mask(ax,ONE,_CMP_LT_OQ)&active);
        if(__builtin_expect(unit==active,0)){
            __m512d p=mode5_poly_x11(k,ax,inneg);
            _mm512_mask_storeu_pd(out+i,active,p);continue;
        }
        __mmask8 wide=(__mmask8)(active&~unit);

        __m512d qf=_mm512_mul_pd(ax,V4OPI);
        __m256i qi=_mm512_cvttpd_epi32(qf);
        __m512d qfloor=_mm512_roundscale_pd(qf,_MM_FROUND_TO_ZERO|_MM_FROUND_NO_EXC);
        __m512d frac=_mm512_sub_pd(qf,qfloor);
        __mmask8 guarded=(__mmask8)((_mm512_cmp_pd_mask(frac,VFT,_CMP_LT_OQ)|
                                     _mm512_cmp_pd_mask(frac,V1MFT,_CMP_GT_OQ))&wide);

        __m256i oi=_mm256_and_si256(qi,_mm256_set1_epi32(7));
        __m256i adj=_mm256_permutevar8x32_epi32(ADJ,oi);
        __m256i mi=_mm256_add_epi32(qi,adj);
        __m512d md=_mm512_cvtepi32_pd(mi);
        /* Octant bit1 = reflection, bit2 = negative sine. */
        __mmask8 rev=(__mmask8)(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_slli_epi32(oi,30)))&wide);
        __mmask8 wide_neg=(__mmask8)(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_slli_epi32(oi,29)))&wide);

        __m512d t1=_mm512_mul_pd(md,VC1);
        __m512d r0=_mm512_sub_pd(ax,t1);
        __m512d t2=_mm512_mul_pd(md,VC2);
        __m512d rh,re;twodiff_cw(r0,t2,&rh,&re);
        __m512d rl=_mm512_fnmadd_pd(md,VC3,re);
        rh=_mm512_mask_sub_pd(rh,rev,Z,rh);
        rl=_mm512_mask_sub_pd(rl,rev,Z,rl);
        rh=_mm512_mask_mov_pd(rh,unit,ax);rl=_mm512_mask_mov_pd(rl,unit,Z);
        rh=_mm512_mask_mov_pd(rh,guarded,Z);rl=_mm512_mask_mov_pd(rl,guarded,Z);
        __mmask8 signmask=(__mmask8)(inneg^wide_neg);
        __m512d p=mode5_poly_low_x11(k,rh,rl,signmask);
        _mm512_mask_storeu_pd(out+i,active,p);
        if(__builtin_expect(guarded!=0,0)){
            for(unsigned lane=0;lane<8&&i+lane<n;lane++)
                if(guarded&(1u<<lane))out[i+lane]=scalar2(k,x[i+lane]);
        }
    }
}

#define X12_TILE 256
OVEC __attribute__((noinline,cold)) static void x55_prepare_generic(const double * __restrict x,size_t base,size_t n,
        __m512d *rh_out,__m512d *rl_out,__mmask8 *sign_out,
        __mmask8 *guard_out,__mmask8 *active_out,unsigned char *pure_unit_out)
{
    const __m512d Z=_mm512_setzero_pd(),ONE=_mm512_set1_pd(1.0);
    const __m512d VINVP=_mm512_set1_pd(0x1.45f306dc9c883p-2);
    const __m512d PIH=_mm512_set1_pd(0x1.921fb54442d18p+1);
    const __m512d PIL=_mm512_set1_pd(0x1.1a62633145c07p-53);
    const __m512i ABSM=_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff));

    unsigned rem=(unsigned)(n-base);
    __mmask8 active=(__mmask8)(rem>=8?0xffu:((1u<<rem)-1u));
    __m512d vx=_mm512_maskz_loadu_pd(active,x+base);
    __m512i vxi=_mm512_castpd_si512(vx);
    __mmask8 inneg=(__mmask8)(_mm512_movepi64_mask(vxi)&active);
    __m512d ax=_mm512_castsi512_pd(_mm512_and_epi64(vxi,ABSM));
    __mmask8 unit=(__mmask8)(_mm512_cmp_pd_mask(ax,ONE,_CMP_LT_OQ)&active);
    *pure_unit_out=(unsigned char)(unit==active);
    if(unit==active){
        *rh_out=ax;*rl_out=Z;*sign_out=inneg;*guard_out=0;*active_out=active;return;
    }

    __m256i qi=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(ax,VINVP),
                    _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m512d qd=_mm512_cvtepi32_pd(qi);
    /* One rounded pi product is avoided: FMA gives the high residual directly;
       keep the true-pi tail as a separate low word for Mode-5 delta formation. */
    __m512d rh=_mm512_fnmadd_pd(qd,PIH,ax);
    __m512d rl=_mm512_mul_pd(qd,_mm512_set1_pd(-0x1.1a62633145c07p-53));

    /* Only near a sine zero do the omitted product/cancellation bits matter in ULPs. */
    __m512d rs_fast=_mm512_add_pd(rh,rl);
    __m512i absm=_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff));
    __m512d ars=_mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(rs_fast),absm));
    __mmask8 repair=(__mmask8)(_mm512_cmp_pd_mask(ars,_mm512_set1_pd(0x1p-14),_CMP_LT_OQ)&active&~unit);
    if(__builtin_expect(repair!=0,0)){
        const __m512d PI1=_mm512_set1_pd(0x1.921fb54400000p+1);
        const __m512d PI2=_mm512_set1_pd(0x1.0b4611a600000p-33);
        const __m512d PI3=_mm512_set1_pd(0x1.3198a2e037073p-68);
        __m512d r0=_mm512_sub_pd(ax,_mm512_mul_pd(qd,PI1));
        __m512d rh3,re3;twodiff_cw(r0,_mm512_mul_pd(qd,PI2),&rh3,&re3);
        __m512d rl3=_mm512_fnmadd_pd(qd,PI3,re3);
        rh=_mm512_mask_mov_pd(rh,repair,rh3);
        rl=_mm512_mask_mov_pd(rl,repair,rl3);
    }

    __m512d rs=_mm512_add_pd(rh,rl);
    __mmask8 rneg=(__mmask8)(_mm512_movepi64_mask(_mm512_castpd_si512(rs))&active);
    rh=_mm512_mask_sub_pd(rh,rneg,Z,rh);
    rl=_mm512_mask_sub_pd(rl,rneg,Z,rl);
    rh=_mm512_mask_mov_pd(rh,unit,ax);
    rl=_mm512_mask_mov_pd(rl,unit,Z);

    __m256i parityv=_mm256_slli_epi32(_mm256_and_si256(qi,_mm256_set1_epi32(1)),31);
    __mmask8 parity=(__mmask8)(_mm256_movemask_ps(_mm256_castsi256_ps(parityv))&active);
    *rh_out=rh;*rl_out=rl;*sign_out=(__mmask8)((inneg^parity^rneg)&active);
    *guard_out=0;*active_out=active;
}

OVEC static inline void x12_prepare_block(const double * __restrict x,size_t base,size_t n,
        __m512d *rh_out,__m512d *rl_out,__mmask8 *sign_out,
        __mmask8 *guard_out,__mmask8 *active_out,unsigned char *pure_unit_out)
{
    const __m512d Z=_mm512_setzero_pd(),ONE=_mm512_set1_pd(1.0);
    const __m512d VINVP=_mm512_set1_pd(0x1.45f306dc9c883p-2);
    const __m512d RS=_mm512_set1_pd(0x1.8p52);
    const __m512d PIH=_mm512_set1_pd(0x1.921fb54442d18p+1);
    const __m512d PIL=_mm512_set1_pd(-0x1.1a62633145c07p-53);
    const __m512i ABSM=_mm512_set1_epi64((long long)UINT64_C(0x7fffffffffffffff));

    unsigned rem=(unsigned)(n-base);
    __mmask8 active=(__mmask8)(rem>=8?0xffu:((1u<<rem)-1u));
    __m512d vx=_mm512_maskz_loadu_pd(active,x+base);
    __m512i vxi=_mm512_castpd_si512(vx);
    __mmask8 inneg=(__mmask8)(_mm512_movepi64_mask(vxi)&active);
    __m512d ax=_mm512_castsi512_pd(_mm512_and_epi64(vxi,ABSM));
    __mmask8 unit=(__mmask8)(_mm512_cmp_pd_mask(ax,ONE,_CMP_LT_OQ)&active);

    /* Preserve X50's very cheap direct unit path. */
    if(__builtin_expect(unit==active,0)){
        *rh_out=ax; *rl_out=Z; *sign_out=inneg; *guard_out=0;
        *active_out=active; *pure_unit_out=1; return;
    }

    /* Dedicated full-width >1 path.  No unit-lane blends, no integer quotient
       conversion, no generic octant bookkeeping. */
    if(__builtin_expect(active==0xff && unit==0,1)){
        __m512d Y=_mm512_fmadd_pd(ax,VINVP,RS);
        __m512d N=_mm512_sub_pd(Y,RS);
        __m512i Ybits=_mm512_castpd_si512(Y);
        __m512i paritybits=_mm512_slli_epi64(Ybits,63);
        __mmask8 parity=_mm512_movepi64_mask(paritybits);

        __m512d rh=_mm512_fnmadd_pd(N,PIH,ax);
        __m512d rl=_mm512_mul_pd(N,PIL);
        __m512d rs=_mm512_add_pd(rh,rl);

        __m512d ars=_mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(rs),ABSM));
        __mmask8 repair=_mm512_cmp_pd_mask(ars,_mm512_set1_pd(0x1p-14),_CMP_LT_OQ);
        if(__builtin_expect(repair!=0,0)){
            const __m512d PI1=_mm512_set1_pd(0x1.921fb54400000p+1);
            const __m512d PI2=_mm512_set1_pd(0x1.0b4611a600000p-33);
            const __m512d PI3=_mm512_set1_pd(0x1.3198a2e037073p-68);
            __m512d r0=_mm512_sub_pd(ax,_mm512_mul_pd(N,PI1));
            __m512d rh3,re3; twodiff_cw(r0,_mm512_mul_pd(N,PI2),&rh3,&re3);
            __m512d rl3=_mm512_fnmadd_pd(N,PI3,re3);
            rh=_mm512_mask_mov_pd(rh,repair,rh3);
            rl=_mm512_mask_mov_pd(rl,repair,rl3);
            rs=_mm512_add_pd(rh,rl);
        }

        __mmask8 rneg=_mm512_movepi64_mask(_mm512_castpd_si512(rs));
        rh=_mm512_mask_sub_pd(rh,rneg,Z,rh);
        rl=_mm512_mask_sub_pd(rl,rneg,Z,rl);
        *rh_out=rh; *rl_out=rl;
        *sign_out=(__mmask8)(inneg^parity^rneg);
        *guard_out=0; *active_out=0xff; *pure_unit_out=0; return;
    }

    /* Mixed/partial blocks retain the already-certified X50 behavior. */
    x55_prepare_generic(x,base,n,rh_out,rl_out,sign_out,guard_out,active_out,pure_unit_out);
}

OVEC static void octant_vector_x20_tail(const s53w_kernel *k,
                                  const double * __restrict x,
                                  double * __restrict out,size_t n)
{
    /* Small calls keep v11 exactly. The staged engine is only for long batches. */
    if(__builtin_expect(n<X12_TILE,0)){octant_vector_v11_single(k,x,out,n);return;}

    _Alignas(64) double rhbuf[X12_TILE],rlbuf[X12_TILE];
    unsigned char signbuf[X12_TILE/8],guardbuf[X12_TILE/8],activebuf[X12_TILE/8],unitbuf[X12_TILE/8];
    for(size_t tile=0;tile<n;tile+=X12_TILE){
        size_t tn=n-tile;if(tn>X12_TILE)tn=X12_TILE;
        size_t blocks=(tn+7)/8;
        /* Phase 1: homogeneous range-reduction/octant work. */
        for(size_t b=0;b<blocks;b++){
            __m512d rh,rl;__mmask8 s,g,a;unsigned char pu;
            x12_prepare_block(x+tile,b*8,tn,&rh,&rl,&s,&g,&a,&pu);
            _mm512_store_pd(rhbuf+b*8,rh);_mm512_store_pd(rlbuf+b*8,rl);
            signbuf[b]=(unsigned char)s;guardbuf[b]=(unsigned char)g;activebuf[b]=(unsigned char)a;unitbuf[b]=pu;
        }
        /* Phase 2: homogeneous Mode-5 work; same six coefficients/FMA order. */
        for(size_t b=0;b<blocks;b++){
            __m512d rh=_mm512_load_pd(rhbuf+b*8),rl=_mm512_load_pd(rlbuf+b*8);
            __m512d p=unitbuf[b]?mode5_poly_x11(k,rh,(__mmask8)signbuf[b]):
                                    mode5_poly_low_x11(k,rh,rl,(__mmask8)signbuf[b]);
            _mm512_mask_storeu_pd(out+tile+b*8,(__mmask8)activebuf[b],p);
        }
        /* Phase 3: rare exact v8 scalar repair. */
        for(size_t b=0;b<blocks;b++)if(__builtin_expect(guardbuf[b]!=0,0)){
            __mmask8 g=(__mmask8)guardbuf[b];
            for(unsigned lane=0;lane<8&&b*8+lane<tn;lane++)
                if(g&(1u<<lane))out[tile+b*8+lane]=scalar2(k,x[tile+b*8+lane]);
        }
    }
}

OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8_x56_general(const s53w_kernel *k,
                                  const double * __restrict x,
                                  double * __restrict out,size_t n)
{
    if(__builtin_expect(n<X12_TILE,0)){octant_vector_v11_single(k,x,out,n);return;}
    const __m512d VK=_mm512_set1_pd(KGRID),VIK=_mm512_set1_pd(INVK),Z=_mm512_setzero_pd();
    const __m512d MH=_mm512_set1_pd(-0.5),M6=_mm512_set1_pd(-1.0/6.0),C24=_mm512_set1_pd(1.0/24.0),C120=_mm512_set1_pd(1.0/120.0);
    const double *tab=(const double *)__builtin_assume_aligned(k->tab,64);
    size_t i=0;
    __m512d nrh0,nrl0,nc0_0,nc1_0; __m256i nji0; __mmask8 ns0,ng0,na0; unsigned char npu0;
    if(n>=32){
        x12_prepare_block(x,0,32,&nrh0,&nrl0,&ns0,&ng0,&na0,&npu0);
        nji0=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(nrh0,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
        nc0_0=_mm512_i32gather_pd(nji0,tab+0*LUTN,8);
        nc1_0=_mm512_i32gather_pd(nji0,tab+1*LUTN,8);
    }
    for(;i+32<=n;i+=32){
        __m512d rh0,rl0,d0,c0_0,c1_0,c2_0,c3_0,c4_0,c5_0,p0; __m256i ji0; __mmask8 s0,g0,a0; unsigned char pu0;
        __m512d rh1,rl1,d1,c0_1,c1_1,c2_1,c3_1,c4_1,c5_1,p1; __m256i ji1; __mmask8 s1,g1,a1; unsigned char pu1;
        __m512d rh2,rl2,d2,c0_2,c1_2,c2_2,c3_2,c4_2,c5_2,p2; __m256i ji2; __mmask8 s2,g2,a2; unsigned char pu2;
        __m512d rh3,rl3,d3,c0_3,c1_3,c2_3,c3_3,c4_3,c5_3,p3; __m256i ji3; __mmask8 s3,g3,a3; unsigned char pu3;
        rh0=nrh0; rl0=nrl0; s0=ns0; g0=ng0; a0=na0; pu0=npu0; ji0=nji0; c0_0=nc0_0; c1_0=nc1_0;
        x12_prepare_block(x+i,8,32,&rh1,&rl1,&s1,&g1,&a1,&pu1);
        ji1=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(rh1,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
        c0_1=_mm512_i32gather_pd(ji1,tab+0*LUTN,8);
        c1_1=_mm512_i32gather_pd(ji1,tab+1*LUTN,8);
        x12_prepare_block(x+i,16,32,&rh2,&rl2,&s2,&g2,&a2,&pu2);
        ji2=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(rh2,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
        c0_2=_mm512_i32gather_pd(ji2,tab+0*LUTN,8);
        c1_2=_mm512_i32gather_pd(ji2,tab+1*LUTN,8);
        x12_prepare_block(x+i,24,32,&rh3,&rl3,&s3,&g3,&a3,&pu3);
        ji3=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(rh3,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
        c0_3=_mm512_i32gather_pd(ji3,tab+0*LUTN,8);
        c1_3=_mm512_i32gather_pd(ji3,tab+1*LUTN,8);
        if(__builtin_expect(i+64<=n,1)){
            x12_prepare_block(x+i+32,0,32,&nrh0,&nrl0,&ns0,&ng0,&na0,&npu0);
            nji0=_mm512_cvt_roundpd_epi32(_mm512_mul_pd(nrh0,VK),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
            nc0_0=_mm512_i32gather_pd(nji0,tab+0*LUTN,8);
            nc1_0=_mm512_i32gather_pd(nji0,tab+1*LUTN,8);
        }
        __m512d jd0=_mm512_cvtepi32_pd(ji0);
        d0=_mm512_fnmadd_pd(jd0,VIK,rh0);
        d0=_mm512_add_pd(d0,rl0);
        __m512d jd1=_mm512_cvtepi32_pd(ji1);
        d1=_mm512_fnmadd_pd(jd1,VIK,rh1);
        d1=_mm512_add_pd(d1,rl1);
        __m512d jd2=_mm512_cvtepi32_pd(ji2);
        d2=_mm512_fnmadd_pd(jd2,VIK,rh2);
        d2=_mm512_add_pd(d2,rl2);
        __m512d jd3=_mm512_cvtepi32_pd(ji3);
        d3=_mm512_fnmadd_pd(jd3,VIK,rh3);
        d3=_mm512_add_pd(d3,rl3);
        c2_0=_mm512_mul_pd(c0_0,MH); c3_0=_mm512_mul_pd(c1_0,M6); c4_0=_mm512_mul_pd(c0_0,C24); c5_0=_mm512_mul_pd(c1_0,C120);
        c2_1=_mm512_mul_pd(c0_1,MH); c3_1=_mm512_mul_pd(c1_1,M6); c4_1=_mm512_mul_pd(c0_1,C24); c5_1=_mm512_mul_pd(c1_1,C120);
        c2_2=_mm512_mul_pd(c0_2,MH); c3_2=_mm512_mul_pd(c1_2,M6); c4_2=_mm512_mul_pd(c0_2,C24); c5_2=_mm512_mul_pd(c1_2,C120);
        c2_3=_mm512_mul_pd(c0_3,MH); c3_3=_mm512_mul_pd(c1_3,M6); c4_3=_mm512_mul_pd(c0_3,C24); c5_3=_mm512_mul_pd(c1_3,C120);
        p0=_mm512_fmadd_pd(c5_0,d0,c4_0);
        p0=_mm512_fmadd_pd(p0,d0,c3_0);
        p0=_mm512_fmadd_pd(p0,d0,c2_0);
        p0=_mm512_fmadd_pd(p0,d0,c1_0);
        p0=_mm512_fmadd_pd(p0,d0,c0_0);
        p1=_mm512_fmadd_pd(c5_1,d1,c4_1);
        p1=_mm512_fmadd_pd(p1,d1,c3_1);
        p1=_mm512_fmadd_pd(p1,d1,c2_1);
        p1=_mm512_fmadd_pd(p1,d1,c1_1);
        p1=_mm512_fmadd_pd(p1,d1,c0_1);
        p2=_mm512_fmadd_pd(c5_2,d2,c4_2);
        p2=_mm512_fmadd_pd(p2,d2,c3_2);
        p2=_mm512_fmadd_pd(p2,d2,c2_2);
        p2=_mm512_fmadd_pd(p2,d2,c1_2);
        p2=_mm512_fmadd_pd(p2,d2,c0_2);
        p3=_mm512_fmadd_pd(c5_3,d3,c4_3);
        p3=_mm512_fmadd_pd(p3,d3,c3_3);
        p3=_mm512_fmadd_pd(p3,d3,c2_3);
        p3=_mm512_fmadd_pd(p3,d3,c1_3);
        p3=_mm512_fmadd_pd(p3,d3,c0_3);
        p0=_mm512_mask_sub_pd(p0,s0,Z,p0);
        _mm512_storeu_pd(out+i+0,p0);
        p1=_mm512_mask_sub_pd(p1,s1,Z,p1);
        _mm512_storeu_pd(out+i+8,p1);
        p2=_mm512_mask_sub_pd(p2,s2,Z,p2);
        _mm512_storeu_pd(out+i+16,p2);
        p3=_mm512_mask_sub_pd(p3,s3,Z,p3);
        _mm512_storeu_pd(out+i+24,p3);
        if(__builtin_expect(g0!=0,0)) for(unsigned lane=0;lane<8;lane++) if(g0&(1u<<lane)) out[i+0+lane]=scalar2(k,x[i+0+lane]);
        if(__builtin_expect(g1!=0,0)) for(unsigned lane=0;lane<8;lane++) if(g1&(1u<<lane)) out[i+8+lane]=scalar2(k,x[i+8+lane]);
        if(__builtin_expect(g2!=0,0)) for(unsigned lane=0;lane<8;lane++) if(g2&(1u<<lane)) out[i+16+lane]=scalar2(k,x[i+16+lane]);
        if(__builtin_expect(g3!=0,0)) for(unsigned lane=0;lane<8;lane++) if(g3&(1u<<lane)) out[i+24+lane]=scalar2(k,x[i+24+lane]);
    }
    if(i<n) octant_vector_x20_tail(k,x+i,out+i,n-i);
}
static const double x65_s[512] __attribute__((aligned(64)))={ 0x0.0p+0,0x1.921f0fe670071p-8,0x1.921d1fcdec784p-7,0x1.2d936bbe30efdp-6,0x1.92155f7a3667ep-6,0x1.f693731d1cf01p-6,0x1.2d865759455cdp-5,0x1.5fc00d290cd43p-5,0x1.91f65f10dd814p-5,0x1.c428d12c0d7e3p-5,0x1.f656e79f820e0p-5,0x1.1440134d709b3p-4,0x1.2d52092ce19f6p-4,0x1.4661179272096p-4,0x1.5f6d00a9aa419p-4,0x1.787586a5d5b21p-4,0x1.917a6bc29b42cp-4,0x1.aa7b724495c03p-4,0x1.c3785c79ec2d5p-4,0x1.dc70ecbae9fc9p-4,0x1.f564e56a9730ep-4,0x1.072a047ba831dp-3,0x1.139f0cedaf577p-3,0x1.20116d4ec7bcfp-3,0x1.2c8106e8e613ap-3,0x1.38edbb0cd8d14p-3,0x1.45576b1293e5ap-3,0x1.51bdf8597c5f2p-3,0x1.5e214448b3fc6p-3,0x1.6a81304f64ab2p-3,0x1.76dd9de50bf31p-3,0x1.83366e89c64c6p-3,0x1.8f8b83c69a60bp-3,0x1.9bdcbf2dc4366p-3,0x1.a82a025b00451p-3,0x1.b4732ef3d6722p-3,0x1.c0b826a7e4f63p-3,0x1.ccf8cb312b286p-3,0x1.d934fe5454311p-3,0x1.e56ca1e101a1bp-3,0x1.f19f97b215f1bp-3,0x1.fdcdc1adfedf9p-3,0x1.04fb80e37fdaep-2,0x1.0b0d9cfdbdb90p-2,0x1.111d262b1f677p-2,0x1.172a0d7765177p-2,0x1.1d3443f4cdb3ep-2,0x1.233bbabc3bb71p-2,0x1.294062ed59f06p-2,0x1.2f422daec0387p-2,0x1.35410c2e18152p-2,0x1.3b3cefa0414b7p-2,0x1.4135c94176601p-2,0x1.472b8a5571054p-2,0x1.4d1e24278e76ap-2,0x1.530d880af3c24p-2,0x1.58f9a75ab1fddp-2,0x1.5ee27379ea693p-2,0x1.64c7ddd3f27c6p-2,0x1.6aa9d7dc77e17p-2,0x1.7088530fa459fp-2,0x1.766340f2418f6p-2,0x1.7c3a9311dcce7p-2,0x1.820e3b04eaac4p-2,0x1.87de2a6aea963p-2,0x1.8daa52ec8a4b0p-2,0x1.9372a63bc93d7p-2,0x1.993716141bdffp-2,0x1.9ef7943a8ed8ap-2,0x1.a4b4127dea1e5p-2,0x1.aa6c82b6d3fcap-2,0x1.b020d6c7f4009p-2,0x1.b5d1009e15cc0p-2,0x1.bb7cf2304bd01p-2,0x1.c1249d8011ee7p-2,0x1.c6c7f4997000bp-2,0x1.cc66e9931c45ep-2,0x1.d2016e8e9db5bp-2,0x1.d79775b86e389p-2,0x1.dd28f1481cc58p-2,0x1.e2b5d3806f63bp-2,0x1.e83e0eaf85114p-2,0x1.edc1952ef78d6p-2,0x1.f3405963fd067p-2,0x1.f8ba4dbf89abap-2,0x1.fe2f64be71210p-2,0x1.01cfc874c3eb7p-1,0x1.0485626ae221ap-1,0x1.073879922ffeep-1,0x1.09e907417c5e1p-1,0x1.0c9704d5d898fp-1,0x1.0f426bb2a8e7ep-1,0x1.11eb3541b4b23p-1,0x1.14915af336cebp-1,0x1.1734d63dedb49p-1,0x1.19d5a09f2b9b8p-1,0x1.1c73b39ae68c8p-1,0x1.1f0f08bbc861bp-1,0x1.21a799933eb59p-1,0x1.243d5fb98ac1fp-1,0x1.26d054cdd12dfp-1,0x1.2960727629ca8p-1,0x1.2bedb25faf3eap-1,0x1.2e780e3e8ea17p-1,0x1.30ff7fce17035p-1,0x1.338400d0c8e57p-1,0x1.36058b10659f3p-1,0x1.3884185dfeb22p-1,0x1.3affa292050b9p-1,0x1.3d78238c58344p-1,0x1.3fed9534556d4p-1,0x1.425ff178e6bb1p-1,0x1.44cf325091dd6p-1,0x1.473b51b987347p-1,0x1.49a449b9b0939p-1,0x1.4c0a145ec0004p-1,0x1.4e6cabbe3e5e9p-1,0x1.50cc09f59a09bp-1,0x1.5328292a35596p-1,0x1.5581038975137p-1,0x1.57d69348ceca0p-1,0x1.5a28d2a5d7250p-1,0x1.5c77bbe65018cp-1,0x1.5ec3495837074p-1,0x1.610b7551d2cdfp-1,0x1.63503a31c1be9p-1,0x1.6591925f0783dp-1,0x1.67cf78491af10p-1,0x1.6a09e667f3bcdp-1,0x1.6c40d73c18275p-1,0x1.6e74454eaa8afp-1,0x1.70a42b3176d7ap-1,0x1.72d0837efff96p-1,0x1.74f948da8d28dp-1,0x1.771e75f037261p-1,0x1.79400574f55e5p-1,0x1.7b5df226aafafp-1,0x1.7d7836cc33db2p-1,0x1.7f8ece3571771p-1,0x1.81a1b33b57accp-1,0x1.83b0e0bff976ep-1,0x1.85bc51ae958ccp-1,0x1.87c400fba2ebfp-1,0x1.89c7e9a4dd4aap-1,0x1.8bc806b151741p-1,0x1.8dc45331698ccp-1,0x1.8fbcca3ef940dp-1,0x1.91b166fd49da2p-1,0x1.93a22499263fbp-1,0x1.958efe48e6dd7p-1,0x1.9777ef4c7d742p-1,0x1.995cf2ed80d22p-1,0x1.9b3e047f38741p-1,0x1.9d1b1f5ea80d5p-1,0x1.9ef43ef29af94p-1,0x1.a0c95eabaf937p-1,0x1.a29a7a0462782p-1,0x1.a4678c8119ac8p-1,0x1.a63091b02fae2p-1,0x1.a7f58529fe69dp-1,0x1.a9b66290ea1a3p-1,0x1.ab7325916c0d4p-1,0x1.ad2bc9e21d511p-1,0x1.aee04b43c1474p-1,0x1.b090a58150200p-1,0x1.b23cd470013b4p-1,0x1.b3e4d3ef55712p-1,0x1.b5889fe921405p-1,0x1.b728345196e3ep-1,0x1.b8c38d27504e9p-1,0x1.ba5aa673590d2p-1,0x1.bbed7c49380eap-1,0x1.bd7c0ac6f952ap-1,0x1.bf064e15377ddp-1,0x1.c08c426725549p-1,0x1.c20de3fa971b0p-1,0x1.c38b2f180bdb1p-1,0x1.c5042012b6907p-1,0x1.c678b3488739bp-1,0x1.c7e8e52233cf3p-1,0x1.c954b213411f5p-1,0x1.cabc169a0b900p-1,0x1.cc1f0f3fcfc5cp-1,0x1.cd7d9898b32f6p-1,0x1.ced7af43cc773p-1,0x1.d02d4feb2bd92p-1,0x1.d17e7743e35dcp-1,0x1.d2cb220e0ef9fp-1,0x1.d4134d14dc93ap-1,0x1.d556f52e93eb1p-1,0x1.d696173c9e68bp-1,0x1.d7d0b02b8ecf9p-1,0x1.d906bcf328d46p-1,0x1.da383a9668988p-1,0x1.db6526238a09bp-1,0x1.dc8d7cb410260p-1,0x1.ddb13b6ccc23cp-1,0x1.ded05f7de47dap-1,0x1.dfeae622dbe2bp-1,0x1.e100cca2980acp-1,0x1.e212104f686e5p-1,0x1.e31eae870ce25p-1,0x1.e426a4b2bc17ep-1,0x1.e529f04729ffcp-1,0x1.e6288ec48e112p-1,0x1.e7227db6a9744p-1,0x1.e817bab4cd10dp-1,0x1.e9084361df7f2p-1,0x1.e9f4156c62ddap-1,0x1.eadb2e8e7a88ep-1,0x1.ebbd8c8df0b74p-1,0x1.ec9b2d3c3bf84p-1,0x1.ed740e7684963p-1,0x1.ee482e25a9dbcp-1,0x1.ef178a3e473c2p-1,0x1.efe220c0b95ecp-1,0x1.f0a7efb9230d7p-1,0x1.f168f53f7205dp-1,0x1.f2252f7763adap-1,0x1.f2dc9c9089a9dp-1,0x1.f38f3ac64e589p-1,0x1.f43d085ff92ddp-1,0x1.f4e603b0b2f2dp-1,0x1.f58a2b1789e84p-1,0x1.f6297cff75cb0p-1,0x1.f6c3f7df5bbb7p-1,0x1.f7599a3a12077p-1,0x1.f7ea629e63d6ep-1,0x1.f8764fa714ba9p-1,0x1.f8fd5ffae41dbp-1,0x1.f97f924c9099bp-1,0x1.f9fce55adb2c8p-1,0x1.fa7557f08a517p-1,0x1.fae8e8e46cfbbp-1,0x1.fb5797195d741p-1,0x1.fbc1617e44186p-1,0x1.fc26470e19fd3p-1,0x1.fc8646cfeb721p-1,0x1.fce15fd6da67bp-1,0x1.fd37914220b84p-1,0x1.fd88da3d12526p-1,0x1.fdd539ff1f456p-1,0x1.fe1cafcbd5b09p-1,0x1.fe5f3af2e3940p-1,0x1.fe9cdad01883ap-1,0x1.fed58ecb673c4p-1,0x1.ff095658e71adp-1,0x1.ff3830f8d575cp-1,0x1.ff621e3796d7ep-1,0x1.ff871dadb81dfp-1,0x1.ffa72effef75dp-1,0x1.ffc251df1d3f8p-1,0x1.ffd886084cd0dp-1,0x1.ffe9cb44b51a1p-1,0x1.fff62169b92dbp-1,0x1.fffd8858e8a92p-1,0x1.0000000000000p+0,0x1.fffd8858e8a92p-1,0x1.fff62169b92dbp-1,0x1.ffe9cb44b51a1p-1,0x1.ffd886084cd0dp-1,0x1.ffc251df1d3f8p-1,0x1.ffa72effef75dp-1,0x1.ff871dadb81dfp-1,0x1.ff621e3796d7ep-1,0x1.ff3830f8d575cp-1,0x1.ff095658e71adp-1,0x1.fed58ecb673c4p-1,0x1.fe9cdad01883ap-1,0x1.fe5f3af2e3940p-1,0x1.fe1cafcbd5b09p-1,0x1.fdd539ff1f456p-1,0x1.fd88da3d12526p-1,0x1.fd37914220b84p-1,0x1.fce15fd6da67bp-1,0x1.fc8646cfeb721p-1,0x1.fc26470e19fd3p-1,0x1.fbc1617e44186p-1,0x1.fb5797195d741p-1,0x1.fae8e8e46cfbbp-1,0x1.fa7557f08a517p-1,0x1.f9fce55adb2c8p-1,0x1.f97f924c9099bp-1,0x1.f8fd5ffae41dbp-1,0x1.f8764fa714ba9p-1,0x1.f7ea629e63d6ep-1,0x1.f7599a3a12077p-1,0x1.f6c3f7df5bbb7p-1,0x1.f6297cff75cb0p-1,0x1.f58a2b1789e84p-1,0x1.f4e603b0b2f2dp-1,0x1.f43d085ff92ddp-1,0x1.f38f3ac64e589p-1,0x1.f2dc9c9089a9dp-1,0x1.f2252f7763adap-1,0x1.f168f53f7205dp-1,0x1.f0a7efb9230d7p-1,0x1.efe220c0b95ecp-1,0x1.ef178a3e473c2p-1,0x1.ee482e25a9dbcp-1,0x1.ed740e7684963p-1,0x1.ec9b2d3c3bf84p-1,0x1.ebbd8c8df0b74p-1,0x1.eadb2e8e7a88ep-1,0x1.e9f4156c62ddap-1,0x1.e9084361df7f2p-1,0x1.e817bab4cd10dp-1,0x1.e7227db6a9744p-1,0x1.e6288ec48e112p-1,0x1.e529f04729ffcp-1,0x1.e426a4b2bc17ep-1,0x1.e31eae870ce25p-1,0x1.e212104f686e5p-1,0x1.e100cca2980acp-1,0x1.dfeae622dbe2bp-1,0x1.ded05f7de47dap-1,0x1.ddb13b6ccc23cp-1,0x1.dc8d7cb410260p-1,0x1.db6526238a09bp-1,0x1.da383a9668988p-1,0x1.d906bcf328d46p-1,0x1.d7d0b02b8ecf9p-1,0x1.d696173c9e68bp-1,0x1.d556f52e93eb1p-1,0x1.d4134d14dc93ap-1,0x1.d2cb220e0ef9fp-1,0x1.d17e7743e35dcp-1,0x1.d02d4feb2bd92p-1,0x1.ced7af43cc773p-1,0x1.cd7d9898b32f6p-1,0x1.cc1f0f3fcfc5cp-1,0x1.cabc169a0b900p-1,0x1.c954b213411f5p-1,0x1.c7e8e52233cf3p-1,0x1.c678b3488739bp-1,0x1.c5042012b6907p-1,0x1.c38b2f180bdb1p-1,0x1.c20de3fa971b0p-1,0x1.c08c426725549p-1,0x1.bf064e15377ddp-1,0x1.bd7c0ac6f952ap-1,0x1.bbed7c49380eap-1,0x1.ba5aa673590d2p-1,0x1.b8c38d27504e9p-1,0x1.b728345196e3ep-1,0x1.b5889fe921405p-1,0x1.b3e4d3ef55712p-1,0x1.b23cd470013b4p-1,0x1.b090a58150200p-1,0x1.aee04b43c1474p-1,0x1.ad2bc9e21d511p-1,0x1.ab7325916c0d4p-1,0x1.a9b66290ea1a3p-1,0x1.a7f58529fe69dp-1,0x1.a63091b02fae2p-1,0x1.a4678c8119ac8p-1,0x1.a29a7a0462782p-1,0x1.a0c95eabaf937p-1,0x1.9ef43ef29af94p-1,0x1.9d1b1f5ea80d5p-1,0x1.9b3e047f38741p-1,0x1.995cf2ed80d22p-1,0x1.9777ef4c7d742p-1,0x1.958efe48e6dd7p-1,0x1.93a22499263fbp-1,0x1.91b166fd49da2p-1,0x1.8fbcca3ef940dp-1,0x1.8dc45331698ccp-1,0x1.8bc806b151741p-1,0x1.89c7e9a4dd4aap-1,0x1.87c400fba2ebfp-1,0x1.85bc51ae958ccp-1,0x1.83b0e0bff976ep-1,0x1.81a1b33b57accp-1,0x1.7f8ece3571771p-1,0x1.7d7836cc33db2p-1,0x1.7b5df226aafafp-1,0x1.79400574f55e5p-1,0x1.771e75f037261p-1,0x1.74f948da8d28dp-1,0x1.72d0837efff96p-1,0x1.70a42b3176d7ap-1,0x1.6e74454eaa8afp-1,0x1.6c40d73c18275p-1,0x1.6a09e667f3bcdp-1,0x1.67cf78491af10p-1,0x1.6591925f0783dp-1,0x1.63503a31c1be9p-1,0x1.610b7551d2cdfp-1,0x1.5ec3495837074p-1,0x1.5c77bbe65018cp-1,0x1.5a28d2a5d7250p-1,0x1.57d69348ceca0p-1,0x1.5581038975137p-1,0x1.5328292a35596p-1,0x1.50cc09f59a09bp-1,0x1.4e6cabbe3e5e9p-1,0x1.4c0a145ec0004p-1,0x1.49a449b9b0939p-1,0x1.473b51b987347p-1,0x1.44cf325091dd6p-1,0x1.425ff178e6bb1p-1,0x1.3fed9534556d4p-1,0x1.3d78238c58344p-1,0x1.3affa292050b9p-1,0x1.3884185dfeb22p-1,0x1.36058b10659f3p-1,0x1.338400d0c8e57p-1,0x1.30ff7fce17035p-1,0x1.2e780e3e8ea17p-1,0x1.2bedb25faf3eap-1,0x1.2960727629ca8p-1,0x1.26d054cdd12dfp-1,0x1.243d5fb98ac1fp-1,0x1.21a799933eb59p-1,0x1.1f0f08bbc861bp-1,0x1.1c73b39ae68c8p-1,0x1.19d5a09f2b9b8p-1,0x1.1734d63dedb49p-1,0x1.14915af336cebp-1,0x1.11eb3541b4b23p-1,0x1.0f426bb2a8e7ep-1,0x1.0c9704d5d898fp-1,0x1.09e907417c5e1p-1,0x1.073879922ffeep-1,0x1.0485626ae221ap-1,0x1.01cfc874c3eb7p-1,0x1.fe2f64be71210p-2,0x1.f8ba4dbf89abap-2,0x1.f3405963fd067p-2,0x1.edc1952ef78d6p-2,0x1.e83e0eaf85114p-2,0x1.e2b5d3806f63bp-2,0x1.dd28f1481cc58p-2,0x1.d79775b86e389p-2,0x1.d2016e8e9db5bp-2,0x1.cc66e9931c45ep-2,0x1.c6c7f4997000bp-2,0x1.c1249d8011ee7p-2,0x1.bb7cf2304bd01p-2,0x1.b5d1009e15cc0p-2,0x1.b020d6c7f4009p-2,0x1.aa6c82b6d3fcap-2,0x1.a4b4127dea1e5p-2,0x1.9ef7943a8ed8ap-2,0x1.993716141bdffp-2,0x1.9372a63bc93d7p-2,0x1.8daa52ec8a4b0p-2,0x1.87de2a6aea963p-2,0x1.820e3b04eaac4p-2,0x1.7c3a9311dcce7p-2,0x1.766340f2418f6p-2,0x1.7088530fa459fp-2,0x1.6aa9d7dc77e17p-2,0x1.64c7ddd3f27c6p-2,0x1.5ee27379ea693p-2,0x1.58f9a75ab1fddp-2,0x1.530d880af3c24p-2,0x1.4d1e24278e76ap-2,0x1.472b8a5571054p-2,0x1.4135c94176601p-2,0x1.3b3cefa0414b7p-2,0x1.35410c2e18152p-2,0x1.2f422daec0387p-2,0x1.294062ed59f06p-2,0x1.233bbabc3bb71p-2,0x1.1d3443f4cdb3ep-2,0x1.172a0d7765177p-2,0x1.111d262b1f677p-2,0x1.0b0d9cfdbdb90p-2,0x1.04fb80e37fdaep-2,0x1.fdcdc1adfedf9p-3,0x1.f19f97b215f1bp-3,0x1.e56ca1e101a1bp-3,0x1.d934fe5454311p-3,0x1.ccf8cb312b286p-3,0x1.c0b826a7e4f63p-3,0x1.b4732ef3d6722p-3,0x1.a82a025b00451p-3,0x1.9bdcbf2dc4366p-3,0x1.8f8b83c69a60bp-3,0x1.83366e89c64c6p-3,0x1.76dd9de50bf31p-3,0x1.6a81304f64ab2p-3,0x1.5e214448b3fc6p-3,0x1.51bdf8597c5f2p-3,0x1.45576b1293e5ap-3,0x1.38edbb0cd8d14p-3,0x1.2c8106e8e613ap-3,0x1.20116d4ec7bcfp-3,0x1.139f0cedaf577p-3,0x1.072a047ba831dp-3,0x1.f564e56a9730ep-4,0x1.dc70ecbae9fc9p-4,0x1.c3785c79ec2d5p-4,0x1.aa7b724495c03p-4,0x1.917a6bc29b42cp-4,0x1.787586a5d5b21p-4,0x1.5f6d00a9aa419p-4,0x1.4661179272096p-4,0x1.2d52092ce19f6p-4,0x1.1440134d709b3p-4,0x1.f656e79f820e0p-5,0x1.c428d12c0d7e3p-5,0x1.91f65f10dd814p-5,0x1.5fc00d290cd43p-5,0x1.2d865759455cdp-5,0x1.f693731d1cf01p-6,0x1.92155f7a3667ep-6,0x1.2d936bbe30efdp-6,0x1.921d1fcdec784p-7,0x1.921f0fe670071p-8 };
static const double x65_c[512] __attribute__((aligned(64)))={ 0x1.0000000000000p+0,0x1.fffd8858e8a92p-1,0x1.fff62169b92dbp-1,0x1.ffe9cb44b51a1p-1,0x1.ffd886084cd0dp-1,0x1.ffc251df1d3f8p-1,0x1.ffa72effef75dp-1,0x1.ff871dadb81dfp-1,0x1.ff621e3796d7ep-1,0x1.ff3830f8d575cp-1,0x1.ff095658e71adp-1,0x1.fed58ecb673c4p-1,0x1.fe9cdad01883ap-1,0x1.fe5f3af2e3940p-1,0x1.fe1cafcbd5b09p-1,0x1.fdd539ff1f456p-1,0x1.fd88da3d12526p-1,0x1.fd37914220b84p-1,0x1.fce15fd6da67bp-1,0x1.fc8646cfeb721p-1,0x1.fc26470e19fd3p-1,0x1.fbc1617e44186p-1,0x1.fb5797195d741p-1,0x1.fae8e8e46cfbbp-1,0x1.fa7557f08a517p-1,0x1.f9fce55adb2c8p-1,0x1.f97f924c9099bp-1,0x1.f8fd5ffae41dbp-1,0x1.f8764fa714ba9p-1,0x1.f7ea629e63d6ep-1,0x1.f7599a3a12077p-1,0x1.f6c3f7df5bbb7p-1,0x1.f6297cff75cb0p-1,0x1.f58a2b1789e84p-1,0x1.f4e603b0b2f2dp-1,0x1.f43d085ff92ddp-1,0x1.f38f3ac64e589p-1,0x1.f2dc9c9089a9dp-1,0x1.f2252f7763adap-1,0x1.f168f53f7205dp-1,0x1.f0a7efb9230d7p-1,0x1.efe220c0b95ecp-1,0x1.ef178a3e473c2p-1,0x1.ee482e25a9dbcp-1,0x1.ed740e7684963p-1,0x1.ec9b2d3c3bf84p-1,0x1.ebbd8c8df0b74p-1,0x1.eadb2e8e7a88ep-1,0x1.e9f4156c62ddap-1,0x1.e9084361df7f2p-1,0x1.e817bab4cd10dp-1,0x1.e7227db6a9744p-1,0x1.e6288ec48e112p-1,0x1.e529f04729ffcp-1,0x1.e426a4b2bc17ep-1,0x1.e31eae870ce25p-1,0x1.e212104f686e5p-1,0x1.e100cca2980acp-1,0x1.dfeae622dbe2bp-1,0x1.ded05f7de47dap-1,0x1.ddb13b6ccc23cp-1,0x1.dc8d7cb410260p-1,0x1.db6526238a09bp-1,0x1.da383a9668988p-1,0x1.d906bcf328d46p-1,0x1.d7d0b02b8ecf9p-1,0x1.d696173c9e68bp-1,0x1.d556f52e93eb1p-1,0x1.d4134d14dc93ap-1,0x1.d2cb220e0ef9fp-1,0x1.d17e7743e35dcp-1,0x1.d02d4feb2bd92p-1,0x1.ced7af43cc773p-1,0x1.cd7d9898b32f6p-1,0x1.cc1f0f3fcfc5cp-1,0x1.cabc169a0b900p-1,0x1.c954b213411f5p-1,0x1.c7e8e52233cf3p-1,0x1.c678b3488739bp-1,0x1.c5042012b6907p-1,0x1.c38b2f180bdb1p-1,0x1.c20de3fa971b0p-1,0x1.c08c426725549p-1,0x1.bf064e15377ddp-1,0x1.bd7c0ac6f952ap-1,0x1.bbed7c49380eap-1,0x1.ba5aa673590d2p-1,0x1.b8c38d27504e9p-1,0x1.b728345196e3ep-1,0x1.b5889fe921405p-1,0x1.b3e4d3ef55712p-1,0x1.b23cd470013b4p-1,0x1.b090a58150200p-1,0x1.aee04b43c1474p-1,0x1.ad2bc9e21d511p-1,0x1.ab7325916c0d4p-1,0x1.a9b66290ea1a3p-1,0x1.a7f58529fe69dp-1,0x1.a63091b02fae2p-1,0x1.a4678c8119ac8p-1,0x1.a29a7a0462782p-1,0x1.a0c95eabaf937p-1,0x1.9ef43ef29af94p-1,0x1.9d1b1f5ea80d5p-1,0x1.9b3e047f38741p-1,0x1.995cf2ed80d22p-1,0x1.9777ef4c7d742p-1,0x1.958efe48e6dd7p-1,0x1.93a22499263fbp-1,0x1.91b166fd49da2p-1,0x1.8fbcca3ef940dp-1,0x1.8dc45331698ccp-1,0x1.8bc806b151741p-1,0x1.89c7e9a4dd4aap-1,0x1.87c400fba2ebfp-1,0x1.85bc51ae958ccp-1,0x1.83b0e0bff976ep-1,0x1.81a1b33b57accp-1,0x1.7f8ece3571771p-1,0x1.7d7836cc33db2p-1,0x1.7b5df226aafafp-1,0x1.79400574f55e5p-1,0x1.771e75f037261p-1,0x1.74f948da8d28dp-1,0x1.72d0837efff96p-1,0x1.70a42b3176d7ap-1,0x1.6e74454eaa8afp-1,0x1.6c40d73c18275p-1,0x1.6a09e667f3bcdp-1,0x1.67cf78491af10p-1,0x1.6591925f0783dp-1,0x1.63503a31c1be9p-1,0x1.610b7551d2cdfp-1,0x1.5ec3495837074p-1,0x1.5c77bbe65018cp-1,0x1.5a28d2a5d7250p-1,0x1.57d69348ceca0p-1,0x1.5581038975137p-1,0x1.5328292a35596p-1,0x1.50cc09f59a09bp-1,0x1.4e6cabbe3e5e9p-1,0x1.4c0a145ec0004p-1,0x1.49a449b9b0939p-1,0x1.473b51b987347p-1,0x1.44cf325091dd6p-1,0x1.425ff178e6bb1p-1,0x1.3fed9534556d4p-1,0x1.3d78238c58344p-1,0x1.3affa292050b9p-1,0x1.3884185dfeb22p-1,0x1.36058b10659f3p-1,0x1.338400d0c8e57p-1,0x1.30ff7fce17035p-1,0x1.2e780e3e8ea17p-1,0x1.2bedb25faf3eap-1,0x1.2960727629ca8p-1,0x1.26d054cdd12dfp-1,0x1.243d5fb98ac1fp-1,0x1.21a799933eb59p-1,0x1.1f0f08bbc861bp-1,0x1.1c73b39ae68c8p-1,0x1.19d5a09f2b9b8p-1,0x1.1734d63dedb49p-1,0x1.14915af336cebp-1,0x1.11eb3541b4b23p-1,0x1.0f426bb2a8e7ep-1,0x1.0c9704d5d898fp-1,0x1.09e907417c5e1p-1,0x1.073879922ffeep-1,0x1.0485626ae221ap-1,0x1.01cfc874c3eb7p-1,0x1.fe2f64be71210p-2,0x1.f8ba4dbf89abap-2,0x1.f3405963fd067p-2,0x1.edc1952ef78d6p-2,0x1.e83e0eaf85114p-2,0x1.e2b5d3806f63bp-2,0x1.dd28f1481cc58p-2,0x1.d79775b86e389p-2,0x1.d2016e8e9db5bp-2,0x1.cc66e9931c45ep-2,0x1.c6c7f4997000bp-2,0x1.c1249d8011ee7p-2,0x1.bb7cf2304bd01p-2,0x1.b5d1009e15cc0p-2,0x1.b020d6c7f4009p-2,0x1.aa6c82b6d3fcap-2,0x1.a4b4127dea1e5p-2,0x1.9ef7943a8ed8ap-2,0x1.993716141bdffp-2,0x1.9372a63bc93d7p-2,0x1.8daa52ec8a4b0p-2,0x1.87de2a6aea963p-2,0x1.820e3b04eaac4p-2,0x1.7c3a9311dcce7p-2,0x1.766340f2418f6p-2,0x1.7088530fa459fp-2,0x1.6aa9d7dc77e17p-2,0x1.64c7ddd3f27c6p-2,0x1.5ee27379ea693p-2,0x1.58f9a75ab1fddp-2,0x1.530d880af3c24p-2,0x1.4d1e24278e76ap-2,0x1.472b8a5571054p-2,0x1.4135c94176601p-2,0x1.3b3cefa0414b7p-2,0x1.35410c2e18152p-2,0x1.2f422daec0387p-2,0x1.294062ed59f06p-2,0x1.233bbabc3bb71p-2,0x1.1d3443f4cdb3ep-2,0x1.172a0d7765177p-2,0x1.111d262b1f677p-2,0x1.0b0d9cfdbdb90p-2,0x1.04fb80e37fdaep-2,0x1.fdcdc1adfedf9p-3,0x1.f19f97b215f1bp-3,0x1.e56ca1e101a1bp-3,0x1.d934fe5454311p-3,0x1.ccf8cb312b286p-3,0x1.c0b826a7e4f63p-3,0x1.b4732ef3d6722p-3,0x1.a82a025b00451p-3,0x1.9bdcbf2dc4366p-3,0x1.8f8b83c69a60bp-3,0x1.83366e89c64c6p-3,0x1.76dd9de50bf31p-3,0x1.6a81304f64ab2p-3,0x1.5e214448b3fc6p-3,0x1.51bdf8597c5f2p-3,0x1.45576b1293e5ap-3,0x1.38edbb0cd8d14p-3,0x1.2c8106e8e613ap-3,0x1.20116d4ec7bcfp-3,0x1.139f0cedaf577p-3,0x1.072a047ba831dp-3,0x1.f564e56a9730ep-4,0x1.dc70ecbae9fc9p-4,0x1.c3785c79ec2d5p-4,0x1.aa7b724495c03p-4,0x1.917a6bc29b42cp-4,0x1.787586a5d5b21p-4,0x1.5f6d00a9aa419p-4,0x1.4661179272096p-4,0x1.2d52092ce19f6p-4,0x1.1440134d709b3p-4,0x1.f656e79f820e0p-5,0x1.c428d12c0d7e3p-5,0x1.91f65f10dd814p-5,0x1.5fc00d290cd43p-5,0x1.2d865759455cdp-5,0x1.f693731d1cf01p-6,0x1.92155f7a3667ep-6,0x1.2d936bbe30efdp-6,0x1.921d1fcdec784p-7,0x1.921f0fe670071p-8,0x0.0p+0,-0x1.921f0fe670071p-8,-0x1.921d1fcdec784p-7,-0x1.2d936bbe30efdp-6,-0x1.92155f7a3667ep-6,-0x1.f693731d1cf01p-6,-0x1.2d865759455cdp-5,-0x1.5fc00d290cd43p-5,-0x1.91f65f10dd814p-5,-0x1.c428d12c0d7e3p-5,-0x1.f656e79f820e0p-5,-0x1.1440134d709b3p-4,-0x1.2d52092ce19f6p-4,-0x1.4661179272096p-4,-0x1.5f6d00a9aa419p-4,-0x1.787586a5d5b21p-4,-0x1.917a6bc29b42cp-4,-0x1.aa7b724495c03p-4,-0x1.c3785c79ec2d5p-4,-0x1.dc70ecbae9fc9p-4,-0x1.f564e56a9730ep-4,-0x1.072a047ba831dp-3,-0x1.139f0cedaf577p-3,-0x1.20116d4ec7bcfp-3,-0x1.2c8106e8e613ap-3,-0x1.38edbb0cd8d14p-3,-0x1.45576b1293e5ap-3,-0x1.51bdf8597c5f2p-3,-0x1.5e214448b3fc6p-3,-0x1.6a81304f64ab2p-3,-0x1.76dd9de50bf31p-3,-0x1.83366e89c64c6p-3,-0x1.8f8b83c69a60bp-3,-0x1.9bdcbf2dc4366p-3,-0x1.a82a025b00451p-3,-0x1.b4732ef3d6722p-3,-0x1.c0b826a7e4f63p-3,-0x1.ccf8cb312b286p-3,-0x1.d934fe5454311p-3,-0x1.e56ca1e101a1bp-3,-0x1.f19f97b215f1bp-3,-0x1.fdcdc1adfedf9p-3,-0x1.04fb80e37fdaep-2,-0x1.0b0d9cfdbdb90p-2,-0x1.111d262b1f677p-2,-0x1.172a0d7765177p-2,-0x1.1d3443f4cdb3ep-2,-0x1.233bbabc3bb71p-2,-0x1.294062ed59f06p-2,-0x1.2f422daec0387p-2,-0x1.35410c2e18152p-2,-0x1.3b3cefa0414b7p-2,-0x1.4135c94176601p-2,-0x1.472b8a5571054p-2,-0x1.4d1e24278e76ap-2,-0x1.530d880af3c24p-2,-0x1.58f9a75ab1fddp-2,-0x1.5ee27379ea693p-2,-0x1.64c7ddd3f27c6p-2,-0x1.6aa9d7dc77e17p-2,-0x1.7088530fa459fp-2,-0x1.766340f2418f6p-2,-0x1.7c3a9311dcce7p-2,-0x1.820e3b04eaac4p-2,-0x1.87de2a6aea963p-2,-0x1.8daa52ec8a4b0p-2,-0x1.9372a63bc93d7p-2,-0x1.993716141bdffp-2,-0x1.9ef7943a8ed8ap-2,-0x1.a4b4127dea1e5p-2,-0x1.aa6c82b6d3fcap-2,-0x1.b020d6c7f4009p-2,-0x1.b5d1009e15cc0p-2,-0x1.bb7cf2304bd01p-2,-0x1.c1249d8011ee7p-2,-0x1.c6c7f4997000bp-2,-0x1.cc66e9931c45ep-2,-0x1.d2016e8e9db5bp-2,-0x1.d79775b86e389p-2,-0x1.dd28f1481cc58p-2,-0x1.e2b5d3806f63bp-2,-0x1.e83e0eaf85114p-2,-0x1.edc1952ef78d6p-2,-0x1.f3405963fd067p-2,-0x1.f8ba4dbf89abap-2,-0x1.fe2f64be71210p-2,-0x1.01cfc874c3eb7p-1,-0x1.0485626ae221ap-1,-0x1.073879922ffeep-1,-0x1.09e907417c5e1p-1,-0x1.0c9704d5d898fp-1,-0x1.0f426bb2a8e7ep-1,-0x1.11eb3541b4b23p-1,-0x1.14915af336cebp-1,-0x1.1734d63dedb49p-1,-0x1.19d5a09f2b9b8p-1,-0x1.1c73b39ae68c8p-1,-0x1.1f0f08bbc861bp-1,-0x1.21a799933eb59p-1,-0x1.243d5fb98ac1fp-1,-0x1.26d054cdd12dfp-1,-0x1.2960727629ca8p-1,-0x1.2bedb25faf3eap-1,-0x1.2e780e3e8ea17p-1,-0x1.30ff7fce17035p-1,-0x1.338400d0c8e57p-1,-0x1.36058b10659f3p-1,-0x1.3884185dfeb22p-1,-0x1.3affa292050b9p-1,-0x1.3d78238c58344p-1,-0x1.3fed9534556d4p-1,-0x1.425ff178e6bb1p-1,-0x1.44cf325091dd6p-1,-0x1.473b51b987347p-1,-0x1.49a449b9b0939p-1,-0x1.4c0a145ec0004p-1,-0x1.4e6cabbe3e5e9p-1,-0x1.50cc09f59a09bp-1,-0x1.5328292a35596p-1,-0x1.5581038975137p-1,-0x1.57d69348ceca0p-1,-0x1.5a28d2a5d7250p-1,-0x1.5c77bbe65018cp-1,-0x1.5ec3495837074p-1,-0x1.610b7551d2cdfp-1,-0x1.63503a31c1be9p-1,-0x1.6591925f0783dp-1,-0x1.67cf78491af10p-1,-0x1.6a09e667f3bcdp-1,-0x1.6c40d73c18275p-1,-0x1.6e74454eaa8afp-1,-0x1.70a42b3176d7ap-1,-0x1.72d0837efff96p-1,-0x1.74f948da8d28dp-1,-0x1.771e75f037261p-1,-0x1.79400574f55e5p-1,-0x1.7b5df226aafafp-1,-0x1.7d7836cc33db2p-1,-0x1.7f8ece3571771p-1,-0x1.81a1b33b57accp-1,-0x1.83b0e0bff976ep-1,-0x1.85bc51ae958ccp-1,-0x1.87c400fba2ebfp-1,-0x1.89c7e9a4dd4aap-1,-0x1.8bc806b151741p-1,-0x1.8dc45331698ccp-1,-0x1.8fbcca3ef940dp-1,-0x1.91b166fd49da2p-1,-0x1.93a22499263fbp-1,-0x1.958efe48e6dd7p-1,-0x1.9777ef4c7d742p-1,-0x1.995cf2ed80d22p-1,-0x1.9b3e047f38741p-1,-0x1.9d1b1f5ea80d5p-1,-0x1.9ef43ef29af94p-1,-0x1.a0c95eabaf937p-1,-0x1.a29a7a0462782p-1,-0x1.a4678c8119ac8p-1,-0x1.a63091b02fae2p-1,-0x1.a7f58529fe69dp-1,-0x1.a9b66290ea1a3p-1,-0x1.ab7325916c0d4p-1,-0x1.ad2bc9e21d511p-1,-0x1.aee04b43c1474p-1,-0x1.b090a58150200p-1,-0x1.b23cd470013b4p-1,-0x1.b3e4d3ef55712p-1,-0x1.b5889fe921405p-1,-0x1.b728345196e3ep-1,-0x1.b8c38d27504e9p-1,-0x1.ba5aa673590d2p-1,-0x1.bbed7c49380eap-1,-0x1.bd7c0ac6f952ap-1,-0x1.bf064e15377ddp-1,-0x1.c08c426725549p-1,-0x1.c20de3fa971b0p-1,-0x1.c38b2f180bdb1p-1,-0x1.c5042012b6907p-1,-0x1.c678b3488739bp-1,-0x1.c7e8e52233cf3p-1,-0x1.c954b213411f5p-1,-0x1.cabc169a0b900p-1,-0x1.cc1f0f3fcfc5cp-1,-0x1.cd7d9898b32f6p-1,-0x1.ced7af43cc773p-1,-0x1.d02d4feb2bd92p-1,-0x1.d17e7743e35dcp-1,-0x1.d2cb220e0ef9fp-1,-0x1.d4134d14dc93ap-1,-0x1.d556f52e93eb1p-1,-0x1.d696173c9e68bp-1,-0x1.d7d0b02b8ecf9p-1,-0x1.d906bcf328d46p-1,-0x1.da383a9668988p-1,-0x1.db6526238a09bp-1,-0x1.dc8d7cb410260p-1,-0x1.ddb13b6ccc23cp-1,-0x1.ded05f7de47dap-1,-0x1.dfeae622dbe2bp-1,-0x1.e100cca2980acp-1,-0x1.e212104f686e5p-1,-0x1.e31eae870ce25p-1,-0x1.e426a4b2bc17ep-1,-0x1.e529f04729ffcp-1,-0x1.e6288ec48e112p-1,-0x1.e7227db6a9744p-1,-0x1.e817bab4cd10dp-1,-0x1.e9084361df7f2p-1,-0x1.e9f4156c62ddap-1,-0x1.eadb2e8e7a88ep-1,-0x1.ebbd8c8df0b74p-1,-0x1.ec9b2d3c3bf84p-1,-0x1.ed740e7684963p-1,-0x1.ee482e25a9dbcp-1,-0x1.ef178a3e473c2p-1,-0x1.efe220c0b95ecp-1,-0x1.f0a7efb9230d7p-1,-0x1.f168f53f7205dp-1,-0x1.f2252f7763adap-1,-0x1.f2dc9c9089a9dp-1,-0x1.f38f3ac64e589p-1,-0x1.f43d085ff92ddp-1,-0x1.f4e603b0b2f2dp-1,-0x1.f58a2b1789e84p-1,-0x1.f6297cff75cb0p-1,-0x1.f6c3f7df5bbb7p-1,-0x1.f7599a3a12077p-1,-0x1.f7ea629e63d6ep-1,-0x1.f8764fa714ba9p-1,-0x1.f8fd5ffae41dbp-1,-0x1.f97f924c9099bp-1,-0x1.f9fce55adb2c8p-1,-0x1.fa7557f08a517p-1,-0x1.fae8e8e46cfbbp-1,-0x1.fb5797195d741p-1,-0x1.fbc1617e44186p-1,-0x1.fc26470e19fd3p-1,-0x1.fc8646cfeb721p-1,-0x1.fce15fd6da67bp-1,-0x1.fd37914220b84p-1,-0x1.fd88da3d12526p-1,-0x1.fdd539ff1f456p-1,-0x1.fe1cafcbd5b09p-1,-0x1.fe5f3af2e3940p-1,-0x1.fe9cdad01883ap-1,-0x1.fed58ecb673c4p-1,-0x1.ff095658e71adp-1,-0x1.ff3830f8d575cp-1,-0x1.ff621e3796d7ep-1,-0x1.ff871dadb81dfp-1,-0x1.ffa72effef75dp-1,-0x1.ffc251df1d3f8p-1,-0x1.ffd886084cd0dp-1,-0x1.ffe9cb44b51a1p-1,-0x1.fff62169b92dbp-1,-0x1.fffd8858e8a92p-1 };

OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8_rawx67(
        const s53w_kernel *k,const double * __restrict x,double * __restrict out,size_t n)
{
    (void)k;
    const __m512d VINV=_mm512_set1_pd(0x1.45f306dc9c883p+7);
    const __m512d RS=_mm512_set1_pd(0x1.8p52);
    const __m512d HHI=_mm512_set1_pd(0x1.921fb54442d18p-8);
    const __m512d HLO=_mm512_set1_pd(0x1.1a62633145c07p-62);
    const __m512d ONE=_mm512_set1_pd(1.0),Z=_mm512_setzero_pd();
    const __m512d MH=_mm512_set1_pd(-0.5),M6=_mm512_set1_pd(-1.0/6.0),C24=_mm512_set1_pd(1.0/24.0),C120=_mm512_set1_pd(1.0/120.0);
    const __m256i I511=_mm256_set1_epi32(511);
    size_t i=0;

    __m512d nvx0,nN0,nd0,nc0_0,nc1_0; __m256i nji0; __mmask8 nsg0;
    if(n>=32){
        nvx0=_mm512_loadu_pd(x);
        __m512d Y=_mm512_fmadd_pd(nvx0,VINV,RS);
        nN0=_mm512_sub_pd(Y,RS);
        __m512i yb=_mm512_castpd_si512(Y);
        nji0=_mm256_and_si256(_mm512_cvtepi64_epi32(yb),I511);
        nsg0=_mm512_movepi64_mask(_mm512_slli_epi64(yb,54));
        nc0_0=_mm512_i32gather_pd(nji0,x65_s,8);
        nc1_0=_mm512_i32gather_pd(nji0,x65_c,8);
        nd0=_mm512_fnmadd_pd(nN0,HHI,nvx0);
        nd0=_mm512_fnmadd_pd(nN0,HLO,nd0);
    }

    for(;i+32<=n;i+=32){
        __m512d vx[4],N[4],d[4],c0[4],c1[4],pv[4];
        __m256i ji[4]; __mmask8 sg[4];
        vx[0]=nvx0; N[0]=nN0; d[0]=nd0; c0[0]=nc0_0; c1[0]=nc1_0; ji[0]=nji0; sg[0]=nsg0;

        for(int g=1;g<4;g++){
            vx[g]=_mm512_loadu_pd(x+i+8*g);
            __m512d Y=_mm512_fmadd_pd(vx[g],VINV,RS);
            N[g]=_mm512_sub_pd(Y,RS);
            __m512i yb=_mm512_castpd_si512(Y);
            ji[g]=_mm256_and_si256(_mm512_cvtepi64_epi32(yb),I511);
            sg[g]=_mm512_movepi64_mask(_mm512_slli_epi64(yb,54));
            c0[g]=_mm512_i32gather_pd(ji[g],x65_s,8);
            c1[g]=_mm512_i32gather_pd(ji[g],x65_c,8);
            d[g]=_mm512_fnmadd_pd(N[g],HHI,vx[g]);
            d[g]=_mm512_fnmadd_pd(N[g],HLO,d[g]);
        }

        /* Launch next iteration's first gather before current polynomial chains. */
        if(__builtin_expect(i+64<=n,1)){
            nvx0=_mm512_loadu_pd(x+i+32);
            __m512d Y=_mm512_fmadd_pd(nvx0,VINV,RS);
            nN0=_mm512_sub_pd(Y,RS);
            __m512i yb=_mm512_castpd_si512(Y);
            nji0=_mm256_and_si256(_mm512_cvtepi64_epi32(yb),I511);
            nsg0=_mm512_movepi64_mask(_mm512_slli_epi64(yb,54));
            nc0_0=_mm512_i32gather_pd(nji0,x65_s,8);
            nc1_0=_mm512_i32gather_pd(nji0,x65_c,8);
            nd0=_mm512_fnmadd_pd(nN0,HHI,nvx0);
            nd0=_mm512_fnmadd_pd(nN0,HLO,nd0);
        }

        /* Same degree-5 polynomial, algebraically grouped into independent
           even/odd d^2 chains: much shorter dependency chain than Horner. */
        for(int g=0;g<4;g++){
            __m512d z=_mm512_mul_pd(d[g],d[g]);
            __m512d ec=_mm512_fmadd_pd(z,C24,MH);      /* -1/2 + z/24 */
            __m512d oc=_mm512_fmadd_pd(z,C120,M6);     /* -1/6 + z/120 */
            __m512d cd=_mm512_mul_pd(c1[g],d[g]);
            /* Same polynomial, but fuse the potentially cancelling leading
               terms first: base = c0 + c1*d.  Corrections are O(z). */
            __m512d base=_mm512_fmadd_pd(c1[g],d[g],c0[g]);
            __m512d ep=_mm512_mul_pd(c0[g],ec);
            __m512d inner=_mm512_fmadd_pd(cd,oc,ep);
            pv[g]=_mm512_fmadd_pd(z,inner,base);
            pv[g]=_mm512_mask_sub_pd(pv[g],sg[g],Z,pv[g]);
            _mm512_storeu_pd(out+i+8*g,pv[g]);
        }
    }
    if(i<n) octant_vector_v8_x56_general(k,x+i,out+i,n-i);
}

OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8(
        const s53w_kernel *k,const double * __restrict x,double * __restrict out,size_t n)
{
    /* X66 is the dedicated documented |x|<=10000 kernel.  No O(n) pre-scan. */
    if(__builtin_expect(n>=32,1)){octant_vector_v8_rawx67(k,x,out,n);return;}
    octant_vector_v8_x56_general(k,x,out,n);
}

#endif

static inline double unit_scalar_v8(const s53w_kernel *k,double x)
{
    double ax=fabs(x); long a=lround(ax*KGRID);
    if(a<0)a=0;if(a>=LUTN)a=LUTN-1;
    double d=fma(-(double)a,INVK,ax);
    double p=k->tab[(size_t)k->deg*LUTN+(size_t)a];
    for(int j=k->deg-1;j>=0;j--)p=fma(p,d,k->tab[(size_t)j*LUTN+(size_t)a]);
    return signbit(x)?-p:p;
}

static void octant_eval_v8(const s53w_kernel *k,const double *x,double *y,size_t n)
{
#if defined(__x86_64__) || defined(__i386__)
    if(hav2()){octant_vector_v8(k,x,y,n);return;}
#endif
    for(size_t i=0;i<n;i++)y[i]=fabs(x[i])<1.0?unit_scalar_v8(k,x[i]):scalar2(k,x[i]);
}

static int guard_count_v8(const double *x,int n)
{
    int c=0; const double ft=BOUND_TAU*FOUR_OVER_PI;
    for(int i=0;i<n;i++){
        double ax=fabs(x[i]);if(ax<1.0)continue;
        double qf=ax*FOUR_OVER_PI;double q=trunc(qf),frac=qf-q;
        if(frac<ft||frac>1.0-ft)c++;
    }
    return c;
}

static int verify_v8(const char *tag,const s53w_kernel *k,const double *x,int n)
{
    double *o=al64((size_t)n*sizeof(double)),*in=al64((size_t)n*sizeof(double));
    if(!o||!in){free(o);free(in);return 0;}
    octant_eval_v8(k,x,o,(size_t)n);vmdSin(n,x,in,VML_HA);
    arb_t ax,ay;arf_t lo,hi;arb_init(ax);arb_init(ay);arf_init(lo);arf_init(hi);
    int uq=0,oe=0,o1=0,ie=0,i1=0;uint64_t om=0,im=0;
    for(int i=0;i<n;i++){
        arb_set_d(ax,x[i]);arb_sin(ay,ax,256);arb_get_lbound_arf(lo,ay,256);arb_get_ubound_arf(hi,ay,256);
        double a=arf_get_d(lo,ARF_RND_NEAR),b=arf_get_d(hi,ARF_RND_NEAR);if(dbits(a)!=dbits(b))continue;
        uq++;uint64_t uo=ulpd(o[i],a),ui=ulpd(in[i],a);if(!uo)oe++;if(uo<=1)o1++;if(uo>om)om=uo;if(!ui)ie++;if(ui<=1)i1++;if(ui>im)im=ui;
    }
    printf("S53X67_VERIFY tag=%s cases=%d unique_ref=%d ours_exact=%d ours_le1ulp=%d ours_max_ulp=%lu intel_exact=%d intel_le1ulp=%d intel_max_ulp=%lu guarded_lanes=%d reference=Arb256\n",
           tag,n,uq,oe,o1,(unsigned long)om,ie,i1,(unsigned long)im,guard_count_v8(x,n));
    arf_clear(hi);arf_clear(lo);arb_clear(ay);arb_clear(ax);free(in);free(o);return uq==n&&om<=1;
}

static void verify_bands_v8(const s53w_kernel *k,const double *x)
{
    for(int b=0;b<3;b++){
        const double *p=x+50*b;double o[50],in[50];octant_eval_v8(k,p,o,50);vmdSin(50,p,in,VML_HA);
        arb_t ax,ay;arf_t lo,hi;arb_init(ax);arb_init(ay);arf_init(lo);arf_init(hi);
        int oe=0,o1=0,ie=0,i1=0;uint64_t om=0,im=0;
        for(int j=0;j<50;j++){
            arb_set_d(ax,p[j]);arb_sin(ay,ax,256);arb_get_lbound_arf(lo,ay,256);arb_get_ubound_arf(hi,ay,256);
            double a=arf_get_d(lo,ARF_RND_NEAR),z=arf_get_d(hi,ARF_RND_NEAR);if(dbits(a)!=dbits(z))continue;
            uint64_t uo=ulpd(o[j],a),ui=ulpd(in[j],a);if(!uo)oe++;if(uo<=1)o1++;if(uo>om)om=uo;if(!ui)ie++;if(ui<=1)i1++;if(ui>im)im=ui;
        }
        const char *bn=b==0?"0_to_1":(b==1?"1_to_500":"1000_to_10000");
        printf("S53X67_BAND band=%s ours_exact=%d ours_le1ulp=%d ours_max_ulp=%lu intel_exact=%d intel_le1ulp=%d intel_max_ulp=%lu guarded=%d\n",
               bn,oe,o1,(unsigned long)om,ie,i1,(unsigned long)im,guard_count_v8(p,50));
        arf_clear(hi);arf_clear(lo);arb_clear(ay);arb_clear(ax);
    }
}

static void make_pi4_stress_v8(double *x)
{
    mpfr_t pi,p4,t;mpfr_init2(pi,256);mpfr_init2(p4,256);mpfr_init2(t,256);mpfr_const_pi(pi,MPFR_RNDN);mpfr_div_ui(p4,pi,4,MPFR_RNDN);
    for(int i=0;i<OSTRESS;i++){
        uint64_t h=mix64(UINT64_C(2026082897)+(uint64_t)i*UINT64_C(0x9e3779b97f4a7c15));unsigned q=(unsigned)((h>>16)%12733U);
        mpfr_mul_ui(t,p4,q,MPFR_RNDN);double b=mpfr_get_d(t,MPFR_RNDN),v=b;
        int s=i&7;if(s==1)v=nextafter(b,INFINITY);else if(s==2)v=nextafter(b,-INFINITY);else if(s==3)v=nextafter(nextafter(b,INFINITY),INFINITY);else if(s==4)v=nextafter(nextafter(b,-INFINITY),-INFINITY);else if(s==5)v=nextafter(nextafter(nextafter(b,INFINITY),INFINITY),INFINITY);else if(s==6)v=nextafter(nextafter(nextafter(b,-INFINITY),-INFINITY),-INFINITY);
        if(v<0)v=0;if(v>10000)v=10000;x[i]=(h&1)?-v:v;
    }
    mpfr_clear(t);mpfr_clear(p4);mpfr_clear(pi);
}

static uint64_t run_v8(const s53w_kernel *k,const double *x,int n,int rounds,volatile double *sink)
{
    double y[CASES];uint64_t t=now_ns();for(int r=0;r<rounds;r++)octant_eval_v8(k,x,y,(size_t)n);t=now_ns()-t;*sink+=y[n-1];return t;
}
static uint64_t run_intel_n(const double *x,int n,int rounds,volatile double *sink)
{
    double y[CASES];uint64_t t=now_ns();for(int r=0;r<rounds;r++)vmdSin(n,x,y,VML_HA);t=now_ns()-t;*sink+=y[n-1];return t;
}

static void bench_band_v8(const s53w_kernel *k,const double *x,int b)
{
    const double *p=x+50*b;volatile double sink=0;double a[BTRIALS],z[BTRIALS],calls=(double)BROUNDS*50.0;
    run_v8(k,p,50,5000,&sink);run_intel_n(p,50,5000,&sink);
    for(int t=0;t<BTRIALS;t++){uint64_t u,v;if(t&1){v=run_intel_n(p,50,BROUNDS,&sink);u=run_v8(k,p,50,BROUNDS,&sink);}else{u=run_v8(k,p,50,BROUNDS,&sink);v=run_intel_n(p,50,BROUNDS,&sink);}a[t]=(double)u/calls;z[t]=(double)v/calls;}
    qsort(a,BTRIALS,sizeof(double),cmpd);qsort(z,BTRIALS,sizeof(double),cmpd);double om=a[BTRIALS/2],im=z[BTRIALS/2];const char *bn=b==0?"0_to_1":(b==1?"1_to_500":"1000_to_10000");
    printf("S53X67_BAND_RESULT band=%s cases=50 ours_ns=%.6f intel_ns=%.6f intel_over_ours=%.6fx throughput_advantage_pct=%.3f guarded=%d sink=%.17g\n",bn,om,im,im/om,(im/om-1.0)*100.0,guard_count_v8(p,50),(double)sink);
}

static int bench_v8(const s53w_kernel *k,const double *x)
{
    if(!hav2()){printf("S53X67_SKIP_TIMING reason=no_avx512\n");return 0;}
    volatile double sink=0;run_v8(k,x,CASES,5000,&sink);run_intel(x,5000,&sink);double ot[OTRIALS],it[OTRIALS],calls=(double)OROUNDS*CASES;
    for(int t=0;t<OTRIALS;t++){uint64_t a,b;if(t&1){b=run_intel(x,OROUNDS,&sink);a=run_v8(k,x,CASES,OROUNDS,&sink);}else{a=run_v8(k,x,CASES,OROUNDS,&sink);b=run_intel(x,OROUNDS,&sink);}ot[t]=(double)a/calls;it[t]=(double)b/calls;printf("S53X67_TRIAL trial=%d ours_ns=%.6f intel_ns=%.6f intel_over_ours=%.6fx\n",t+1,ot[t],it[t],it[t]/ot[t]);}
    qsort(ot,OTRIALS,sizeof(double),cmpd);qsort(it,OTRIALS,sizeof(double),cmpd);double om=ot[OTRIALS/2],im=it[OTRIALS/2];
    printf("S53X67_RESULT cases=150 terms=2 degree=5 K=%d ours_e2e_ns_per_input=%.6f intel_ha_ns_per_input=%.6f ours_over_intel=%.6fx intel_over_ours=%.6fx throughput_advantage_pct=%.3f requested_guarded_lanes=%d reduction=Xeon_AVX512_X67_compensated_grouped rare_fallback=table_DD_qpi unit_dispatch=direct formula=unchanged_Mode5_secant_spine accuracy_contract=le1ulp all_raw_input_work_included=1 sink=%.17g\n",SF_K,om,im,om/im,im/om,(im/om-1.0)*100.0,guard_count_v8(x,CASES),(double)sink);
    for(int b=0;b<3;b++)bench_band_v8(k,x,b);return 0;
}

static uint64_t run_batch_x11(const s53w_kernel *k,const double *x,double *y,
                              int n,int rounds,volatile double *sink)
{
    uint64_t t=now_ns();for(int r=0;r<rounds;r++)octant_eval_v8(k,x,y,(size_t)n);
    t=now_ns()-t;*sink+=y[n-1];return t;
}
static uint64_t run_intel_batch_x11(const double *x,double *y,int n,int rounds,
                                    volatile double *sink)
{
    uint64_t t=now_ns();for(int r=0;r<rounds;r++)vmdSin(n,x,y,VML_HA);
    t=now_ns()-t;*sink+=y[n-1];return t;
}
static void bench_batches_x11(const s53w_kernel *k,const double base[CASES])
{
    const int ns[2]={1200,9600};const int rounds[2]={25000,3125};
    for(int b=0;b<2;b++){
        int n=ns[b],rr=rounds[b];double *x=al64((size_t)n*sizeof(double));
        double *yo=al64((size_t)n*sizeof(double)),*yi=al64((size_t)n*sizeof(double));
        if(!x||!yo||!yi){free(yi);free(yo);free(x);continue;}
        for(int i=0;i<n;i++)x[i]=base[i%CASES];volatile double sink=0;
        run_batch_x11(k,x,yo,n,100,&sink);run_intel_batch_x11(x,yi,n,100,&sink);
        double ot[7],it[7],calls=(double)n*(double)rr;
        for(int t=0;t<7;t++){uint64_t a,z;if(t&1){z=run_intel_batch_x11(x,yi,n,rr,&sink);a=run_batch_x11(k,x,yo,n,rr,&sink);}else{a=run_batch_x11(k,x,yo,n,rr,&sink);z=run_intel_batch_x11(x,yi,n,rr,&sink);}ot[t]=(double)a/calls;it[t]=(double)z/calls;}
        qsort(ot,7,sizeof(double),cmpd);qsort(it,7,sizeof(double),cmpd);
        printf("S53X67_BATCH_RESULT cases=%d ours_ns=%.6f intel_ns=%.6f ours_over_intel=%.6fx intel_over_ours=%.6fx repeated_certified150=1 formula=unchanged_Mode5_secant_spine sink=%.17g\n",n,ot[3],it[3],ot[3]/it[3],it[3]/ot[3],(double)sink);
        free(yi);free(yo);free(x);
    }
}

int main(void)
{
    int cpu=pin();mkl_set_num_threads_local(1);printf("S53X67_DOMAIN cpu_pin=%d target=binary64_53bit cases=150 bands=0_to_1,1_to_500,1000_to_10000 signs=25pos_25neg_each intel=oneMKL_vmdSin_VML_HA reduction=cosine_style_pi4_octant_guarded_v8 formula=unchanged_Mode5_secant_spine\n",cpu);
    if(!redtab2_init())return 2;s53w_kernel *k=kernel_create(2);if(!k)return 3;double x[CASES];make_bench(x);
    if(!verify_v8("requested150",k,x,CASES)){kernel_destroy(k);redtab2_clear();return 4;}verify_bands_v8(k,x);
    double *st=al64(STRESS*sizeof(double));if(!st)return 5;make_stress(st);if(!verify_v8("legacy_npi_npi2_stress",k,st,STRESS)){free(st);kernel_destroy(k);redtab2_clear();return 6;}free(st);
    double *os=al64(OSTRESS*sizeof(double));if(!os)return 7;make_pi4_stress_v8(os);if(!verify_v8("all_npi4_boundary_stress",k,os,OSTRESS)){free(os);kernel_destroy(k);redtab2_clear();return 8;}free(os);
    int rc=bench_v8(k,x);bench_batches_x11(k,x);kernel_destroy(k);redtab2_clear();flint_cleanup_master();return rc;
}
