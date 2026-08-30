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
OVEC static inline void x12_prepare_block(const double * __restrict x,size_t base,size_t n,
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

OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8(const s53w_kernel *k,
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
    printf("S53X50_VERIFY tag=%s cases=%d unique_ref=%d ours_exact=%d ours_le1ulp=%d ours_max_ulp=%lu intel_exact=%d intel_le1ulp=%d intel_max_ulp=%lu guarded_lanes=%d reference=Arb256\n",
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
        printf("S53X50_BAND band=%s ours_exact=%d ours_le1ulp=%d ours_max_ulp=%lu intel_exact=%d intel_le1ulp=%d intel_max_ulp=%lu guarded=%d\n",
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
    printf("S53X50_BAND_RESULT band=%s cases=50 ours_ns=%.6f intel_ns=%.6f intel_over_ours=%.6fx throughput_advantage_pct=%.3f guarded=%d sink=%.17g\n",bn,om,im,im/om,(im/om-1.0)*100.0,guard_count_v8(p,50),(double)sink);
}

static int bench_v8(const s53w_kernel *k,const double *x)
{
    if(!hav2()){printf("S53X50_SKIP_TIMING reason=no_avx512\n");return 0;}
    volatile double sink=0;run_v8(k,x,CASES,5000,&sink);run_intel(x,5000,&sink);double ot[OTRIALS],it[OTRIALS],calls=(double)OROUNDS*CASES;
    for(int t=0;t<OTRIALS;t++){uint64_t a,b;if(t&1){b=run_intel(x,OROUNDS,&sink);a=run_v8(k,x,CASES,OROUNDS,&sink);}else{a=run_v8(k,x,CASES,OROUNDS,&sink);b=run_intel(x,OROUNDS,&sink);}ot[t]=(double)a/calls;it[t]=(double)b/calls;printf("S53X50_TRIAL trial=%d ours_ns=%.6f intel_ns=%.6f intel_over_ours=%.6fx\n",t+1,ot[t],it[t],it[t]/ot[t]);}
    qsort(ot,OTRIALS,sizeof(double),cmpd);qsort(it,OTRIALS,sizeof(double),cmpd);double om=ot[OTRIALS/2],im=it[OTRIALS/2];
    printf("S53X50_RESULT cases=150 terms=2 degree=5 K=%d ours_e2e_ns_per_input=%.6f intel_ha_ns_per_input=%.6f ours_over_intel=%.6fx intel_over_ours=%.6fx throughput_advantage_pct=%.3f requested_guarded_lanes=%d reduction=Xeon_AVX512_X50_cross_iteration_lookahead rare_fallback=table_DD_qpi unit_dispatch=direct formula=unchanged_Mode5_secant_spine accuracy_contract=le1ulp all_raw_input_work_included=1 sink=%.17g\n",SF_K,om,im,om/im,im/om,(im/om-1.0)*100.0,guard_count_v8(x,CASES),(double)sink);
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
        printf("S53X50_BATCH_RESULT cases=%d ours_ns=%.6f intel_ns=%.6f ours_over_intel=%.6fx intel_over_ours=%.6fx repeated_certified150=1 formula=unchanged_Mode5_secant_spine sink=%.17g\n",n,ot[3],it[3],ot[3]/it[3],it[3]/ot[3],(double)sink);
        free(yi);free(yo);free(x);
    }
}

int main(void)
{
    int cpu=pin();mkl_set_num_threads_local(1);printf("S53X50_DOMAIN cpu_pin=%d target=binary64_53bit cases=150 bands=0_to_1,1_to_500,1000_to_10000 signs=25pos_25neg_each intel=oneMKL_vmdSin_VML_HA reduction=cosine_style_pi4_octant_guarded_v8 formula=unchanged_Mode5_secant_spine\n",cpu);
    if(!redtab2_init())return 2;s53w_kernel *k=kernel_create(2);if(!k)return 3;double x[CASES];make_bench(x);
    if(!verify_v8("requested150",k,x,CASES)){kernel_destroy(k);redtab2_clear();return 4;}verify_bands_v8(k,x);
    double *st=al64(STRESS*sizeof(double));if(!st)return 5;make_stress(st);if(!verify_v8("legacy_npi_npi2_stress",k,st,STRESS)){free(st);kernel_destroy(k);redtab2_clear();return 6;}free(st);
    double *os=al64(OSTRESS*sizeof(double));if(!os)return 7;make_pi4_stress_v8(os);if(!verify_v8("all_npi4_boundary_stress",k,os,OSTRESS)){free(os);kernel_destroy(k);redtab2_clear();return 8;}free(os);
    int rc=bench_v8(k,x);bench_batches_x11(k,x);kernel_destroy(k);redtab2_clear();flint_cleanup_master();return rc;
}
