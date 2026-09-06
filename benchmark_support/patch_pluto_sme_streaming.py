from pathlib import Path
import sys

p=Path(sys.argv[1])
s=p.read_text()

# Real SME/ZA experiment. Keep PLUTO reduction, LUT and repair semantics frozen;
# only map the hot polynomial evaluation onto the M4 FP64 ZA matrix engine.
s=s.replace('#include <arm_sve.h>\n', '#include <arm_sve.h>\n#include <arm_sme.h>\n', 1)

old_sig='''__arm_locally_streaming static size_t pluto_sme_core(const double* __restrict x,double* __restrict y,size_t n,uint64_t* __restrict repairs){'''
new_sig='''__arm_locally_streaming static size_t pluto_sme_core(const double* __restrict x,double* __restrict y,size_t n,uint64_t* __restrict repairs) __arm_new("za") {'''
assert old_sig in s
s=s.replace(old_sig,new_sig,1)

# M4 streaming vector length is 64 bytes = 8 FP64 lanes.  SME_FA64 is absent on
# this runner, so retain exact scalar AoS LUT staging instead of wide gathers.
old='''    size_t nr=0;\n    for(size_t i=0;i<n;i+=svcntd()){'''
new='''    size_t nr=0;\n    alignas(64) uint64_t ji_tmp[8];\n    alignas(64) uint64_t c0_bits[8];\n    alignas(64) uint64_t c1_bits[8];\n    alignas(64) uint64_t qbits_tmp[8];\n    alignas(64) double delta_tmp[8];\n    alignas(64) double za_rows[8][8];\n    alignas(64) double p_tmp[8];\n    const uint64_t* coeff_bits=reinterpret_cast<const uint64_t*>(opt_cos53_coeff_aos);\n    for(size_t i=0;i<n;i+=8){'''
assert old in s
s=s.replace(old,new,1)

old='''        svuint64_t idx2=svlsl_n_u64_x(pg,jidx,1); svfloat64_t c0=svld1_gather_u64index_f64(pg,opt_cos53_coeff_aos,idx2); svfloat64_t c1=svld1_gather_u64index_f64(pg,opt_cos53_coeff_aos,svadd_n_u64_x(pg,idx2,1));\n'''
new='''        svst1_u64(pg,ji_tmp,jidx);\n        svst1_u64(pg,qbits_tmp,qbits);\n        svst1_f64(pg,delta_tmp,delta);\n        const uint64_t lanes=(n-i<8)?(uint64_t)(n-i):UINT64_C(8);\n        for(uint64_t l=0;l<lanes;l++) {\n            const uint64_t j=ji_tmp[l];\n            const uint64_t* q=coeff_bits + 2*j;\n            c0_bits[l]=q[0];\n            c1_bits[l]=q[1];\n            if(__builtin_expect(j>=2009,0)) repairs[nr++]=i+l;\n        }\n        svbool_t pg2=svwhilelt_b64((uint64_t)i,(uint64_t)n);\n        svfloat64_t c0=svreinterpret_f64_u64(svld1_u64(pg2,c0_bits));\n        svfloat64_t c1=svreinterpret_f64_u64(svld1_u64(pg2,c1_bits));\n        qbits=svld1_u64(pg2,qbits_tmp);\n        delta=svld1_f64(pg2,delta_tmp);\n'''
assert old in s
s=s.replace(old,new,1)

# Same cubic PLUTO polynomial, regrouped into two coefficient/basis products:
#   p = c0*(1 + MH*d^2) + c1*(d + M6*d^3)
# The two products are accumulated by genuine FP64 FMOPA into ZA.  We extract
# only the diagonal, which corresponds exactly to independent input lanes.
# Accuracy is not assumed: the existing <=2 ULP MPFR gate decides whether this
# regrouping is acceptable.  If not, the next iteration will preserve Horner
# rounding while still executing its FMAs through ZA.
old='''        svfloat64_t c2=svmul_n_f64_x(pg,c0,MH); svfloat64_t c3=svmul_n_f64_x(pg,c1,M6); svfloat64_t p=svmla_f64_x(pg,c2,c3,delta); p=svmla_f64_x(pg,c1,p,delta); p=svmla_f64_x(pg,c0,p,delta);\n        svuint64_t parity=svand_u64_x(pg,qbits,oneu); svuint64_t outsign=svlsl_n_u64_x(pg,parity,63); p=svreinterpretq_f64_u64(sveor_u64_x(pg,svreinterpret_u64_f64(p),outsign)); svst1_f64(pg,y+i,p);\n'''
# Some source revisions use svreinterpret_f64_u64 rather than the accidental q-form.
if old not in s:
    old='''        svfloat64_t c2=svmul_n_f64_x(pg,c0,MH); svfloat64_t c3=svmul_n_f64_x(pg,c1,M6); svfloat64_t p=svmla_f64_x(pg,c2,c3,delta); p=svmla_f64_x(pg,c1,p,delta); p=svmla_f64_x(pg,c0,p,delta);\n        svuint64_t parity=svand_u64_x(pg,qbits,oneu); svuint64_t outsign=svlsl_n_u64_x(pg,parity,63); p=svreinterpret_f64_u64(sveor_u64_x(pg,svreinterpret_u64_f64(p),outsign)); svst1_f64(pg,y+i,p);\n'''
new='''        svfloat64_t d2=svmul_f64_x(pg2,delta,delta);\n        svfloat64_t basis0=svmla_n_f64_x(pg2,svdup_n_f64(1.0),d2,MH);\n        svfloat64_t d3=svmul_f64_x(pg2,d2,delta);\n        svfloat64_t basis1=svmla_n_f64_x(pg2,delta,d3,M6);\n        svzero_za();\n        svmopa_za64_f64_m(0,pg2,pg2,c0,basis0);\n        svmopa_za64_f64_m(0,pg2,pg2,c1,basis1);\n        for(uint64_t r=0;r<lanes;r++) svst1_hor_za64(0,(uint32_t)r,pg2,za_rows[r]);\n        for(uint64_t l=0;l<lanes;l++) p_tmp[l]=za_rows[l][l];\n        svfloat64_t p=svld1_f64(pg2,p_tmp);\n        svuint64_t parity=svand_u64_x(pg2,qbits,oneu); svuint64_t outsign=svlsl_n_u64_x(pg2,parity,63); p=svreinterpret_f64_u64(sveor_u64_x(pg2,svreinterpret_u64_f64(p),outsign)); svst1_f64(pg2,y+i,p);\n'''
assert old in s
s=s.replace(old,new,1)

# Repair detection is already fused into scalar LUT staging.
old='''        svbool_t rp=svcmpge_n_u64(pg,jidx,2009); uint64_t cnt=svcntp_b64(pg,rp);\n        if(cnt){svuint64_t lane=svindex_u64((uint64_t)i,1); svuint64_t packed=svcompact_u64(rp,lane); svbool_t cp=svwhilelt_b64((uint64_t)0,cnt); svst1_u64(cp,repairs+nr,packed); nr+=cnt;}\n'''
assert old in s
s=s.replace(old,'',1)

s=s.replace('// SME integration: the same PLUTO arithmetic in SME streaming mode. On M4 Pro\n// the streaming vector length is 512 bits, so this processes 8 binary64 lanes.\n// ZA is deliberately not used: PLUTO is lane-wise with indexed LUT loads, so\n// forcing an outer-product/tile formulation would change the computation rather\n// than accelerate it. SSVE is the elementwise execution engine supplied by SME.',
'''// SME/ZA integration: exact frozen PLUTO reduction and LUT machinery, with the\n// polynomial batch mapped to the M4 FP64 ZA matrix engine via FMOPA.''')

p.write_text(s)
