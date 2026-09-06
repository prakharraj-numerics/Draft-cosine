from pathlib import Path
import sys

p=Path(sys.argv[1])
s=p.read_text()

# Fast SME mapping: preserve frozen PLUTO reduction/LUT/Horner exactly, enter
# streaming mode once for the whole batch, use 512-bit SSVE for lane-wise math,
# and avoid the pathological indexed SVE gathers with exact scalar AoS staging.
old_sig='''__arm_locally_streaming static size_t pluto_sme_core(const double* __restrict x,double* __restrict y,size_t n,uint64_t* __restrict repairs){'''
new_sig='''extern "C" size_t pluto_sme_stream(const double* __restrict x,double* __restrict y,size_t n,uint64_t* __restrict repairs) __arm_streaming;\nextern "C" size_t pluto_sme_stream(const double* __restrict x,double* __restrict y,size_t n,uint64_t* __restrict repairs) __arm_streaming {'''
assert old_sig in s
s=s.replace(old_sig,new_sig,1)

old='''    size_t nr=0;\n    for(size_t i=0;i<n;i+=svcntd()){'''
new='''    size_t nr=0;\n    alignas(64) uint64_t ji_tmp[8];\n    alignas(64) uint64_t c0_bits[8];\n    alignas(64) uint64_t c1_bits[8];\n    alignas(64) uint64_t qbits_tmp[8];\n    alignas(64) double delta_tmp[8];\n    const uint64_t* coeff_bits=reinterpret_cast<const uint64_t*>(opt_cos53_coeff_aos);\n    for(size_t i=0;i<n;i+=8){'''
assert old in s
s=s.replace(old,new,1)

old='''        svuint64_t idx2=svlsl_n_u64_x(pg,jidx,1); svfloat64_t c0=svld1_gather_u64index_f64(pg,opt_cos53_coeff_aos,idx2); svfloat64_t c1=svld1_gather_u64index_f64(pg,opt_cos53_coeff_aos,svadd_n_u64_x(pg,idx2,1));\n'''
new='''        svst1_u64(pg,ji_tmp,jidx);\n        svst1_u64(pg,qbits_tmp,qbits);\n        svst1_f64(pg,delta_tmp,delta);\n        const uint64_t lanes=(n-i<8)?(uint64_t)(n-i):UINT64_C(8);\n        for(uint64_t l=0;l<lanes;l++) {\n            const uint64_t j=ji_tmp[l];\n            const uint64_t* q=coeff_bits + 2*j;\n            c0_bits[l]=q[0];\n            c1_bits[l]=q[1];\n            if(__builtin_expect(j>=2009,0)) repairs[nr++]=i+l;\n        }\n        svbool_t pg2=svwhilelt_b64((uint64_t)i,(uint64_t)n);\n        svfloat64_t c0=svreinterpret_f64_u64(svld1_u64(pg2,c0_bits));\n        svfloat64_t c1=svreinterpret_f64_u64(svld1_u64(pg2,c1_bits));\n        qbits=svld1_u64(pg2,qbits_tmp);\n        delta=svld1_f64(pg2,delta_tmp);\n'''
assert old in s
s=s.replace(old,new,1)

old='''        svfloat64_t c2=svmul_n_f64_x(pg,c0,MH); svfloat64_t c3=svmul_n_f64_x(pg,c1,M6); svfloat64_t p=svmla_f64_x(pg,c2,c3,delta); p=svmla_f64_x(pg,c1,p,delta); p=svmla_f64_x(pg,c0,p,delta);\n        svuint64_t parity=svand_u64_x(pg,qbits,oneu); svuint64_t outsign=svlsl_n_u64_x(pg,parity,63); p=svreinterpret_f64_u64(sveor_u64_x(pg,svreinterpret_u64_f64(p),outsign)); svst1_f64(pg,y+i,p);\n'''
new='''        svfloat64_t c2=svmul_n_f64_x(pg2,c0,MH); svfloat64_t c3=svmul_n_f64_x(pg2,c1,M6); svfloat64_t p=svmla_f64_x(pg2,c2,c3,delta); p=svmla_f64_x(pg2,c1,p,delta); p=svmla_f64_x(pg2,c0,p,delta);\n        svuint64_t parity=svand_u64_x(pg2,qbits,oneu); svuint64_t outsign=svlsl_n_u64_x(pg2,parity,63); p=svreinterpret_f64_u64(sveor_u64_x(pg2,svreinterpret_u64_f64(p),outsign)); svst1_f64(pg2,y+i,p);\n'''
assert old in s
s=s.replace(old,new,1)

old='''        svbool_t rp=svcmpge_n_u64(pg,jidx,2009); uint64_t cnt=svcntp_b64(pg,rp);\n        if(cnt){svuint64_t lane=svindex_u64((uint64_t)i,1); svuint64_t packed=svcompact_u64(rp,lane); svbool_t cp=svwhilelt_b64((uint64_t)0,cnt); svst1_u64(cp,repairs+nr,packed); nr+=cnt;}\n'''
assert old in s
s=s.replace(old,'',1)

needle='''static inline void pluto_sme(const double* __restrict x,double* __restrict y,size_t n){'''
replacement='''extern "C" size_t pluto_sme_eval_raw(const double*,double*,size_t,uint64_t*);\nstatic inline void pluto_sme(const double* __restrict x,double* __restrict y,size_t n){'''
assert needle in s
s=s.replace(needle,replacement,1)
s=s.replace('''    size_t nr=pluto_sme_core(x,y,n,repairs.data());''','''    size_t nr=pluto_sme_eval_raw(x,y,n,repairs.data());''',1)

s=s.replace('// SME integration: the same PLUTO arithmetic in SME streaming mode. On M4 Pro\n// the streaming vector length is 512 bits, so this processes 8 binary64 lanes.\n// ZA is deliberately not used: PLUTO is lane-wise with indexed LUT loads, so\n// forcing an outer-product/tile formulation would change the computation rather\n// than accelerate it. SSVE is the elementwise execution engine supplied by SME.',
'''// SME streaming integration: frozen PLUTO reduction/LUT/Horner, 8 FP64 lanes\n// per 64-byte streaming vector, exact scalar-staged LUT loads, no ZA diagonal tax.''')

p.write_text(s)
