from pathlib import Path
import sys
p=Path(sys.argv[1]); s=p.read_text()

insert=r'''

// ---- A-E bottleneck diagnostics (math constants/table unchanged) ----
struct DiagPrep{const double* key=nullptr; size_t n=0; std::vector<double> delta,c0,c1; std::vector<uint64_t> j,qbits;};
static thread_local std::vector<DiagPrep> diag_cache;
static DiagPrep& diag_prep(const double*x,size_t n){
    for(auto&z:diag_cache)if(z.key==x&&z.n==n)return z;
    diag_cache.emplace_back(); auto&z=diag_cache.back(); z.key=x;z.n=n;z.delta.resize(n);z.c0.resize(n);z.c1.resize(n);z.j.resize(n);z.qbits.resize(n);
    constexpr uint64_t JMASK=(UINT64_C(1)<<52)-1;
    for(size_t i=0;i<n;i++){
        double ax=std::fabs(x[i]); double qscaled=ax*INVPI, qmagic=qscaled+0x1p52, qd=qmagic-0x1p52; uint64_t qb;std::memcpy(&qb,&qmagic,8);z.qbits[i]=qb;
        double t=ax-qd*PI_P1; double rh=std::fma(qd,-PI_P2,t); double d=t-rh; double rl=std::fma(qd,-PI_P2,d);
        uint64_t rhu;std::memcpy(&rhu,&rh,8);uint64_t rs=rhu&UINT64_C(0x8000000000000000),ahu=rhu&~UINT64_C(0x8000000000000000),rlu;std::memcpy(&rlu,&rl,8);rlu^=rs;double ah,al;std::memcpy(&ah,&ahu,8);std::memcpy(&al,&rlu,8);
        double jm=ah*KGRID+0x1p52,jd=jm-0x1p52;uint64_t jb;std::memcpy(&jb,&jm,8);uint64_t j=jb&JMASK;z.j[i]=j;
        double de=std::fma(jd,NINVK_HI,ah);de=std::fma(jd,NINVK_LO,de);de+=al;z.delta[i]=de;
        uint64_t jj=j<2009?j:0; z.c0[i]=opt_cos53_coeff_aos[2*jj]; z.c1[i]=opt_cos53_coeff_aos[2*jj+1];
    } return z;
}

extern "C" void pluto_diag_B_stream(const double*x,double*y,size_t n) __arm_streaming;
extern "C" void pluto_diag_B_stream(const double*x,double*y,size_t n) __arm_streaming {
 const svuint64_t sm=svdup_u64(UINT64_C(0x8000000000000000)),one=svdup_u64(1),jm=svdup_u64((UINT64_C(1)<<52)-1);
 for(size_t i=0;i<n;i+=8){svbool_t pg=svwhilelt_b64((uint64_t)i,(uint64_t)n);svfloat64_t ax=svabs_f64_x(pg,svld1_f64(pg,x+i));svfloat64_t qmagic=svadd_n_f64_x(pg,svmul_n_f64_x(pg,ax,INVPI),0x1p52),qd=svsub_n_f64_x(pg,qmagic,0x1p52);svuint64_t qb=svreinterpret_u64_f64(qmagic);svfloat64_t t=svsub_f64_x(pg,ax,svmul_n_f64_x(pg,qd,PI_P1)),rh=svmla_n_f64_x(pg,t,qd,-PI_P2);svfloat64_t d=svsub_f64_x(pg,t,rh),rl=svmla_n_f64_x(pg,d,qd,-PI_P2);svuint64_t rhu=svreinterpret_u64_f64(rh),rs=svand_u64_x(pg,rhu,sm);svfloat64_t ah=svreinterpret_f64_u64(svbic_u64_x(pg,rhu,sm)),al=svreinterpret_f64_u64(sveor_u64_x(pg,svreinterpret_u64_f64(rl),rs));svfloat64_t jmagic=svadd_n_f64_x(pg,svmul_n_f64_x(pg,ah,KGRID),0x1p52),jd=svsub_n_f64_x(pg,jmagic,0x1p52);(void)svand_u64_x(pg,svreinterpret_u64_f64(jmagic),jm);svfloat64_t de=svmla_n_f64_x(pg,ah,jd,NINVK_HI);de=svmla_n_f64_x(pg,de,jd,NINVK_LO);de=svadd_f64_x(pg,de,al);svfloat64_t c0=svdup_n_f64(opt_cos53_coeff_aos[1000]),c1=svdup_n_f64(opt_cos53_coeff_aos[1001]);svfloat64_t c2=svmul_n_f64_x(pg,c0,MH),c3=svmul_n_f64_x(pg,c1,M6),v=svmla_f64_x(pg,c2,c3,de);v=svmla_f64_x(pg,c1,v,de);v=svmla_f64_x(pg,c0,v,de);svuint64_t par=svand_u64_x(pg,qb,one),os=svlsl_n_u64_x(pg,par,63);v=svreinterpret_f64_u64(sveor_u64_x(pg,svreinterpret_u64_f64(v),os));svst1_f64(pg,y+i,v);}
}

extern "C" void pluto_diag_C_stream(const double*x,double*y,size_t n,const double*de0,const double*c00,const double*c10,const uint64_t*qb0) __arm_streaming;
extern "C" void pluto_diag_C_stream(const double*x,double*y,size_t n,const double*de0,const double*c00,const double*c10,const uint64_t*qb0) __arm_streaming {
 (void)x; const svuint64_t one=svdup_u64(1);for(size_t i=0;i<n;i+=8){svbool_t pg=svwhilelt_b64((uint64_t)i,(uint64_t)n);svfloat64_t de=svld1_f64(pg,de0+i),c0=svld1_f64(pg,c00+i),c1=svld1_f64(pg,c10+i);svfloat64_t c2=svmul_n_f64_x(pg,c0,MH),c3=svmul_n_f64_x(pg,c1,M6),v=svmla_f64_x(pg,c2,c3,de);v=svmla_f64_x(pg,c1,v,de);v=svmla_f64_x(pg,c0,v,de);svuint64_t qb=svld1_u64(pg,qb0+i),os=svlsl_n_u64_x(pg,svand_u64_x(pg,qb,one),63);v=svreinterpret_f64_u64(sveor_u64_x(pg,svreinterpret_u64_f64(v),os));svst1_f64(pg,y+i,v);}}

extern "C" void pluto_diag_D_stream(const double*x,double*y,size_t n,const double*de0,const uint64_t*j0,const uint64_t*qb0) __arm_streaming;
extern "C" void pluto_diag_D_stream(const double*x,double*y,size_t n,const double*de0,const uint64_t*j0,const uint64_t*qb0) __arm_streaming {
 (void)x; const svuint64_t one=svdup_u64(1);alignas(64)double a0[8],a1[8];for(size_t i=0;i<n;i+=8){size_t lanes=std::min<size_t>(8,n-i);for(size_t l=0;l<lanes;l++){uint64_t j=j0[i+l];if(j>=2009)j=0;a0[l]=opt_cos53_coeff_aos[2*j];a1[l]=opt_cos53_coeff_aos[2*j+1];}svbool_t pg=svwhilelt_b64((uint64_t)i,(uint64_t)n);svfloat64_t de=svld1_f64(pg,de0+i),c0=svld1_f64(pg,a0),c1=svld1_f64(pg,a1);svfloat64_t c2=svmul_n_f64_x(pg,c0,MH),c3=svmul_n_f64_x(pg,c1,M6),v=svmla_f64_x(pg,c2,c3,de);v=svmla_f64_x(pg,c1,v,de);v=svmla_f64_x(pg,c0,v,de);svuint64_t qb=svld1_u64(pg,qb0+i),os=svlsl_n_u64_x(pg,svand_u64_x(pg,qb,one),63);v=svreinterpret_f64_u64(sveor_u64_x(pg,svreinterpret_u64_f64(v),os));svst1_f64(pg,y+i,v);}}

template<size_t BS> static void pluto_diag_E_stream_impl(const double*x,double*y,size_t n) __arm_streaming {
 const svuint64_t sm=svdup_u64(UINT64_C(0x8000000000000000)),one=svdup_u64(1),jmask=svdup_u64((UINT64_C(1)<<52)-1);alignas(64)uint64_t jj[BS],qbv[BS];alignas(64)double dd[BS],a0[BS],a1[BS];
 for(size_t base=0;base<n;base+=BS){size_t bn=std::min<size_t>(BS,n-base);for(size_t k=0;k<bn;k+=8){size_t i=base+k;svbool_t pg=svwhilelt_b64((uint64_t)k,(uint64_t)bn);svfloat64_t ax=svabs_f64_x(pg,svld1_f64(pg,x+i));svfloat64_t qm=svadd_n_f64_x(pg,svmul_n_f64_x(pg,ax,INVPI),0x1p52),qd=svsub_n_f64_x(pg,qm,0x1p52);svuint64_t qb=svreinterpret_u64_f64(qm);svfloat64_t t=svsub_f64_x(pg,ax,svmul_n_f64_x(pg,qd,PI_P1)),rh=svmla_n_f64_x(pg,t,qd,-PI_P2);svfloat64_t d=svsub_f64_x(pg,t,rh),rl=svmla_n_f64_x(pg,d,qd,-PI_P2);svuint64_t rhu=svreinterpret_u64_f64(rh),rs=svand_u64_x(pg,rhu,sm);svfloat64_t ah=svreinterpret_f64_u64(svbic_u64_x(pg,rhu,sm)),al=svreinterpret_f64_u64(sveor_u64_x(pg,svreinterpret_u64_f64(rl),rs));svfloat64_t jmg=svadd_n_f64_x(pg,svmul_n_f64_x(pg,ah,KGRID),0x1p52),jd=svsub_n_f64_x(pg,jmg,0x1p52);svuint64_t j=svand_u64_x(pg,svreinterpret_u64_f64(jmg),jmask);svfloat64_t de=svmla_n_f64_x(pg,ah,jd,NINVK_HI);de=svmla_n_f64_x(pg,de,jd,NINVK_LO);de=svadd_f64_x(pg,de,al);svst1_u64(pg,jj+k,j);svst1_u64(pg,qbv+k,qb);svst1_f64(pg,dd+k,de);}for(size_t k=0;k<bn;k++){uint64_t j=jj[k];if(j>=2009){a0[k]=1.0;a1[k]=0.0;}else{a0[k]=opt_cos53_coeff_aos[2*j];a1[k]=opt_cos53_coeff_aos[2*j+1];}}for(size_t k=0;k<bn;k+=8){svbool_t pg=svwhilelt_b64((uint64_t)k,(uint64_t)bn);svfloat64_t de=svld1_f64(pg,dd+k),c0=svld1_f64(pg,a0+k),c1=svld1_f64(pg,a1+k);svfloat64_t c2=svmul_n_f64_x(pg,c0,MH),c3=svmul_n_f64_x(pg,c1,M6),v=svmla_f64_x(pg,c2,c3,de);v=svmla_f64_x(pg,c1,v,de);v=svmla_f64_x(pg,c0,v,de);svuint64_t os=svlsl_n_u64_x(pg,svand_u64_x(pg,svld1_u64(pg,qbv+k),one),63);v=svreinterpret_f64_u64(sveor_u64_x(pg,svreinterpret_u64_f64(v),os));svst1_f64(pg,y+base+k,v);}for(size_t k=0;k<bn;k++)if(__builtin_expect(jj[k]>=2009,0))y[base+k]=std::cos(x[base+k]);}
}
extern "C" void pluto_diag_E64_stream(const double*x,double*y,size_t n) __arm_streaming; extern "C" void pluto_diag_E64_stream(const double*x,double*y,size_t n) __arm_streaming {pluto_diag_E_stream_impl<64>(x,y,n);}
extern "C" void pluto_diag_E128_stream(const double*x,double*y,size_t n) __arm_streaming; extern "C" void pluto_diag_E128_stream(const double*x,double*y,size_t n) __arm_streaming {pluto_diag_E_stream_impl<128>(x,y,n);}

extern "C" void pluto_diag_B_raw(const double*,double*,size_t);extern "C" void pluto_diag_C_raw(const double*,double*,size_t,const double*,const double*,const double*,const uint64_t*);extern "C" void pluto_diag_D_raw(const double*,double*,size_t,const double*,const uint64_t*,const uint64_t*);extern "C" void pluto_diag_E64_raw(const double*,double*,size_t);extern "C" void pluto_diag_E128_raw(const double*,double*,size_t);
static void pluto_B(const double*x,double*y,size_t n){pluto_diag_B_raw(x,y,n);}static void pluto_C(const double*x,double*y,size_t n){auto&z=diag_prep(x,n);pluto_diag_C_raw(x,y,n,z.delta.data(),z.c0.data(),z.c1.data(),z.qbits.data());}static void pluto_D(const double*x,double*y,size_t n){auto&z=diag_prep(x,n);pluto_diag_D_raw(x,y,n,z.delta.data(),z.j.data(),z.qbits.data());}static void pluto_E64(const double*x,double*y,size_t n){pluto_diag_E64_raw(x,y,n);}static void pluto_E128(const double*x,double*y,size_t n){pluto_diag_E128_raw(x,y,n);}
struct BRunner{void run(const double*x,double*y,size_t n){pluto_B(x,y,n);}};struct CRunner{void run(const double*x,double*y,size_t n){pluto_C(x,y,n);}};struct DRunner{void run(const double*x,double*y,size_t n){pluto_D(x,y,n);}};struct E64Runner{void run(const double*x,double*y,size_t n){pluto_E64(x,y,n);}};struct E128Runner{void run(const double*x,double*y,size_t n){pluto_E128(x,y,n);}};
'''
needle='static void fill_case(double*x,size_t n,int c)'
assert needle in s;s=s.replace(needle,insert+'\n'+needle,1)
old='if(mode=="neon"){NeonRunner r;return bench("neon",r,n);}if(mode=="sme"||mode=="A"){SmeRunner r;return bench("A",r,n);}if(mode=="B"){BRunner r;return bench("B",r,n);}if(mode=="C"){CRunner r;return bench("C",r,n);}if(mode=="D"){DRunner r;return bench("D",r,n);}if(mode=="E64"){E64Runner r;return bench("E64",r,n);}if(mode=="E128"){E128Runner r;return bench("E128",r,n);}if(mode=="apple"){AppleRunner r;return bench("apple",r,n);}return 3;'
if old not in s:
 old='if(mode=="neon"){NeonRunner r;return bench("neon",r,n);}if(mode=="sme"){SmeRunner r;return bench("sme",r,n);}if(mode=="apple"){AppleRunner r;return bench("apple",r,n);}return 3;'
new='if(mode=="neon"){NeonRunner r;return bench("neon",r,n);}if(mode=="sme"||mode=="A"){SmeRunner r;return bench("A",r,n);}if(mode=="B"){BRunner r;return bench("B",r,n);}if(mode=="C"){CRunner r;return bench("C",r,n);}if(mode=="D"){DRunner r;return bench("D",r,n);}if(mode=="E64"){E64Runner r;return bench("E64",r,n);}if(mode=="E128"){E128Runner r;return bench("E128",r,n);}if(mode=="apple"){AppleRunner r;return bench("apple",r,n);}return 3;'
assert old in s;s=s.replace(old,new,1)
old='score("neon","9600",x,pluto_neon);score("sme","9600",x,pluto_sme);';new='score("neon","9600",x,pluto_neon);score("sme","9600",x,pluto_sme);score("E64","9600",x,pluto_E64);score("E128","9600",x,pluto_E128);';assert old in s;s=s.replace(old,new,1)
old='score("sme","1m_stress",x,pluto_sme);return 0;';new='score("sme","1m_stress",x,pluto_sme);score("E64","1m_stress",x,pluto_E64);score("E128","1m_stress",x,pluto_E128);return 0;';assert old in s;s=s.replace(old,new,1)
p.write_text(s)
