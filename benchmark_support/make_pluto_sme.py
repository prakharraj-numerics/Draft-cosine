from pathlib import Path
import sys

src=Path(sys.argv[1]).read_text()
src=src.replace('#include <arm_neon.h>\n', '#include <arm_neon.h>\n#include <arm_sve.h>\n', 1)
start=src.index('static inline void opt_cos53_eval(')
end=src.index('\nclass Adaptive2', start)

sme=r'''static void opt_cos53_eval(const double* __restrict x,double* __restrict y,size_t n) __arm_streaming;
static void opt_cos53_eval(const double* __restrict x,double* __restrict y,size_t n) __arm_streaming
{
    // PLUTO-SME: same K1280/no-P3 range reduction, LUT, degree-3 polynomial,
    // parity/sign repair and root-only scalar fallback as frozen PLUTO.
    const svfloat64_t magic = svdup_n_f64(0x1p52);
    const svuint64_t signmask = svdup_n_u64(UINT64_C(0x8000000000000000));
    const svuint64_t jmask = svdup_n_u64((UINT64_C(1)<<52)-1);
    const size_t vl = svcntd();
    alignas(64) uint64_t jt[32];
    alignas(64) double c0t[32], c1t[32];

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
        svuint64_t ahbits = svbic_u64_x(pg,rhbits,signmask);
        svfloat64_t ah = svreinterpret_f64_u64(ahbits);
        svuint64_t albits = sveor_u64_x(pg,svreinterpret_u64_f64(rl),rsign);
        svfloat64_t al = svreinterpret_f64_u64(albits);

        svfloat64_t jscaled = svmul_n_f64_x(pg,ah,KGRID);
        svfloat64_t jmagic = svadd_f64_x(pg,jscaled,magic);
        svfloat64_t jd = svsub_f64_x(pg,jmagic,magic);
        svuint64_t jbits = svreinterpret_u64_f64(jmagic);
        svuint64_t ji = svand_u64_x(pg,jbits,jmask);

        svfloat64_t delta = svmla_n_f64_x(pg,ah,jd,NINVK_HI);
        delta = svmla_n_f64_x(pg,delta,jd,NINVK_LO);
        delta = svadd_f64_x(pg,delta,al);

        // Streaming SVE on AppleClang does not expose arbitrary memory gather.
        // Spill the 8 LUT indices, perform the same AoS scalar loads, then reload
        // contiguous coefficient vectors. All arithmetic remains 512-bit SSVE.
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

        // Frozen PLUTO root repair: only affected lanes fall back to scalar cos.
        svbool_t bad = svcmpge_n_u64(pg,ji,2009);
        if (__builtin_expect(svptest_any(pg,bad),0)) {
            for(size_t k=0;k<lanes;k++) if(jt[k]>=2009) y[i+k]=std::cos(x[i+k]);
        }
    }
}
'''

src=src[:start]+sme+src[end:]
Path(sys.argv[2]).write_text(src)
