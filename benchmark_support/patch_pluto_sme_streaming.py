from pathlib import Path
import sys

p=Path(sys.argv[1])
s=p.read_text()

# Namespace Apple SME ABI:
# ordinary C caller -> hand-written SMSTART wrapper -> __arm_streaming body.
old_sig='''__arm_locally_streaming static size_t pluto_sme_core(const double* __restrict x,double* __restrict y,size_t n,uint64_t* __restrict repairs){'''
new_sig='''extern "C" size_t pluto_sme_stream(const double* __restrict x,double* __restrict y,size_t n,uint64_t* __restrict repairs) __arm_streaming;\nextern "C" size_t pluto_sme_stream(const double* __restrict x,double* __restrict y,size_t n,uint64_t* __restrict repairs) __arm_streaming {'''
assert old_sig in s
s=s.replace(old_sig,new_sig,1)

# Namespace M4/M5 exposes 64-byte streaming vectors: 8 doubles.
old='''    size_t nr=0;\n    for(size_t i=0;i<n;i+=svcntd()){'''
new='''    size_t nr=0;\n    alignas(64) uint64_t ji_tmp[8];\n    alignas(64) uint64_t c0_bits[8];\n    alignas(64) uint64_t c1_bits[8];\n    const uint64_t* coeff_bits=reinterpret_cast<const uint64_t*>(opt_cos53_coeff_aos);\n    for(size_t i=0;i<n;i+=8){'''
assert old in s
s=s.replace(old,new,1)

# SME_FA64=0: stage exact AoS coefficients with scalar loads.
# Crucial SIGILL fix: do NOT keep the original predicate live across this scalar loop.
# AppleClang was spilling/reloading it as `ldr p0, [x8, #-1, mul vl]` (opcode 0x85bf1d00),
# which traps in this streaming execution path. Recreate ptrue after staging instead.
old='''        svuint64_t idx2=svlsl_n_u64_x(pg,jidx,1); svfloat64_t c0=svld1_gather_u64index_f64(pg,opt_cos53_coeff_aos,idx2); svfloat64_t c1=svld1_gather_u64index_f64(pg,opt_cos53_coeff_aos,svadd_n_u64_x(pg,idx2,1));\n'''
new='''        svst1_u64(pg,ji_tmp,jidx);\n        const uint64_t lanes=(n-i<8)?(uint64_t)(n-i):UINT64_C(8);\n        for(uint64_t l=0;l<lanes;l++) {\n            const uint64_t j=ji_tmp[l];\n            const uint64_t* q=coeff_bits + 2*j;\n            c0_bits[l]=q[0];\n            c1_bits[l]=q[1];\n            if(__builtin_expect(j>=2009,0)) repairs[nr++]=i+l;\n        }\n        svbool_t pg2=svptrue_b64();\n        svfloat64_t c0=svreinterpret_f64_u64(svld1_u64(pg2,c0_bits));\n        svfloat64_t c1=svreinterpret_f64_u64(svld1_u64(pg2,c1_bits));\n'''
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
old='''    size_t nr=pluto_sme_core(x,y,n,repairs.data());'''
new='''    size_t nr=pluto_sme_eval_raw(x,y,n,repairs.data());'''
assert old in s
s=s.replace(old,new,1)

s=s.replace('// SME integration: the same PLUTO arithmetic in SME streaming mode. On M4 Pro\n// the streaming vector length is 512 bits, so this processes 8 binary64 lanes.',
            '// SME integration: exact PLUTO arithmetic in 64-byte Apple Streaming SVE.\n// A hand-written wrapper enters streaming mode once per whole batch.')

p.write_text(s)
