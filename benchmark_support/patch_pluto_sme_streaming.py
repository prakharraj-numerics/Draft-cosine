from pathlib import Path
import sys

p=Path(sys.argv[1])
s=p.read_text()

# Reliable contract on the Namespace Apple runners:
# ordinary C caller -> hand-written SMSTART wrapper -> streaming-compatible body.
# The streaming-compatible body itself must not insert mode transitions and must
# only contain instructions valid with either PSTATE.SM value.
old_sig='''__arm_locally_streaming static size_t pluto_sme_core(const double* __restrict x,double* __restrict y,size_t n,uint64_t* __restrict repairs){'''
new_sig='''extern "C" size_t pluto_sme_stream(const double* __restrict x,double* __restrict y,size_t n,uint64_t* __restrict repairs) __arm_streaming_compatible;\nextern "C" size_t pluto_sme_stream(const double* __restrict x,double* __restrict y,size_t n,uint64_t* __restrict repairs) __arm_streaming_compatible {'''
assert old_sig in s
s=s.replace(old_sig,new_sig,1)

# Namespace M4/M5 currently exposes a 64-byte streaming vector length: 8 doubles.
# Keep only one 8-lane scratch set, aligned to one streaming vector, outside the
# loop. This avoids the previous oversized 3x256-byte stack frame and any stack
# probe/helper call that could force an internal streaming-mode transition.
old='''    size_t nr=0;\n    for(size_t i=0;i<n;i+=svcntd()){'''
new='''    size_t nr=0;\n    alignas(64) uint64_t ji_tmp[8];\n    alignas(64) uint64_t c0_bits[8];\n    alignas(64) uint64_t c1_bits[8];\n    const uint64_t* coeff_bits=reinterpret_cast<const uint64_t*>(opt_cos53_coeff_aos);\n    for(size_t i=0;i<n;i+=8){'''
assert old in s
s=s.replace(old,new,1)

# Indexed gathers are illegal in Streaming SVE on these hosts because SME_FA64=0.
# Preserve exact K1280 coefficient values by one fused scalar staging pass.  The
# active-lane count is computed scalarly (max 8), avoiding an unnecessary CNTP.
old='''        svuint64_t idx2=svlsl_n_u64_x(pg,jidx,1); svfloat64_t c0=svld1_gather_u64index_f64(pg,opt_cos53_coeff_aos,idx2); svfloat64_t c1=svld1_gather_u64index_f64(pg,opt_cos53_coeff_aos,svadd_n_u64_x(pg,idx2,1));\n'''
new='''        svst1_u64(pg,ji_tmp,jidx);\n        const uint64_t lanes=(n-i<8)?(uint64_t)(n-i):UINT64_C(8);\n        for(uint64_t l=0;l<lanes;l++) {\n            const uint64_t j=ji_tmp[l];\n            const uint64_t* q=coeff_bits + 2*j;\n            c0_bits[l]=q[0];\n            c1_bits[l]=q[1];\n            if(__builtin_expect(j>=2009,0)) repairs[nr++]=i+l;\n        }\n        svfloat64_t c0=svreinterpret_f64_u64(svld1_u64(pg,c0_bits));\n        svfloat64_t c1=svreinterpret_f64_u64(svld1_u64(pg,c1_bits));\n'''
assert old in s
s=s.replace(old,new,1)

old='''        svbool_t rp=svcmpge_n_u64(pg,jidx,2009); uint64_t cnt=svcntp_b64(pg,rp);\n        if(cnt){svuint64_t lane=svindex_u64((uint64_t)i,1); svuint64_t packed=svcompact_u64(rp,lane); svbool_t cp=svwhilelt_b64((uint64_t)0,cnt); svst1_u64(cp,repairs+nr,packed); nr+=cnt;}\n'''
assert old in s
s=s.replace(old,'',1)

# Ordinary ABI entry point. The external assembly wrapper is the *only* place
# that changes PSTATE.SM, once for the entire batch.
needle='''static inline void pluto_sme(const double* __restrict x,double* __restrict y,size_t n){'''
replacement='''extern "C" size_t pluto_sme_eval_raw(const double*,double*,size_t,uint64_t*);\nstatic inline void pluto_sme(const double* __restrict x,double* __restrict y,size_t n){'''
assert needle in s
s=s.replace(needle,replacement,1)
old='''    size_t nr=pluto_sme_core(x,y,n,repairs.data());'''
new='''    size_t nr=pluto_sme_eval_raw(x,y,n,repairs.data());'''
assert old in s
s=s.replace(old,new,1)

s=s.replace('// SME integration: the same PLUTO arithmetic in SME streaming mode. On M4 Pro\n// the streaming vector length is 512 bits, so this processes 8 binary64 lanes.',
            '// SME integration: exact PLUTO arithmetic in 64-byte Apple Streaming SVE.\n// A hand-written wrapper enters streaming mode exactly once per whole batch.')

p.write_text(s)
