from pathlib import Path
import sys
p=Path(sys.argv[1])
s=p.read_text()
old='''        svuint64_t idx2=svlsl_n_u64_x(pg,jidx,1); svfloat64_t c0=svld1_gather_u64index_f64(pg,opt_cos53_coeff_aos,idx2); svfloat64_t c1=svld1_gather_u64index_f64(pg,opt_cos53_coeff_aos,svadd_n_u64_x(pg,idx2,1));
'''
new='''        // AppleClang marks SVE indexed gathers as non-streaming-only.  Keep the
        // PLUTO K1280 lookup exactly intact by spilling the eight (or future-SVL)
        // indices, staging coefficient *bits* with scalar integer loads, then
        // bringing them back into streaming Z registers.  This preserves every
        // coefficient and every arithmetic operation while exposing the real cost
        // of PLUTO's indexed LUT under SME.
        alignas(256) uint64_t ji_tmp[32]{};
        alignas(256) uint64_t c0_bits[32]{};
        alignas(256) uint64_t c1_bits[32]{};
        svst1_u64(pg,ji_tmp,jidx);
        const uint64_t lanes=svcntp_b64(pg,pg);
        const uint64_t* coeff_bits=reinterpret_cast<const uint64_t*>(opt_cos53_coeff_aos);
        for(uint64_t l=0;l<lanes;l++) {
            const uint64_t j=ji_tmp[l];
            c0_bits[l]=coeff_bits[2*j];
            c1_bits[l]=coeff_bits[2*j+1];
        }
        svfloat64_t c0=svreinterpret_f64_u64(svld1_u64(pg,c0_bits));
        svfloat64_t c1=svreinterpret_f64_u64(svld1_u64(pg,c1_bits));
'''
assert old in s
s=s.replace(old,new,1)
old='''        svbool_t rp=svcmpge_n_u64(pg,jidx,2009); uint64_t cnt=svcntp_b64(pg,rp);
        if(cnt){svuint64_t lane=svindex_u64((uint64_t)i,1); svuint64_t packed=svcompact_u64(rp,lane); svbool_t cp=svwhilelt_b64((uint64_t)0,cnt); svst1_u64(cp,repairs+nr,packed); nr+=cnt;}
'''
new='''        for(uint64_t l=0;l<lanes;l++) if(__builtin_expect(ji_tmp[l]>=2009,0)) repairs[nr++]=i+l;
'''
assert old in s
s=s.replace(old,new,1)
s=s.replace('// SME integration: the same PLUTO arithmetic in SME streaming mode. On M4 Pro\n// the streaming vector length is 512 bits, so this processes 8 binary64 lanes.','// SME integration: the same PLUTO arithmetic in SME streaming mode on any\n// SME-capable Apple Silicon. The loop is vector-length agnostic via svcntd().')
p.write_text(s)
