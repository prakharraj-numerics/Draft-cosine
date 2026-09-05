#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
BASE_SCRIPT=benchmark_support/run_apple_cos53_direct_amx_vs_frozen_below30k.sh

if [[ "$MODE" == "build" ]]; then
  # Reuse the already-audited dependency setup, frozen K=2048 constants and
  # baseline runner. Then replace only the experimental AMX evaluator.
  bash "$BASE_SCRIPT" build

  python3 - <<'PY'
from pathlib import Path
p=Path('/tmp/base.cpp')
s=p.read_text()
start=s.index('static inline void cos53_eval_amx(')
end=s.index('\nclass AppleTwoCoreAMX', start)
new=r'''static inline void cos53_eval_amx(const double* x, double* y, size_t n)
{
    if (n == 0) return;

    // AMX-native attack:
    //   * retain the incumbent's accurate pi-multiple reduction,
    //   * DELETE the 3218-entry local cosine coefficient LUT,
    //   * approximate cos(r) globally on 0<=r<=pi/2 with a degree-9
    //     near-minimax polynomial in z=r^2,
    //   * keep polynomial coefficients resident in AMX Y registers,
    //   * use AMX GENLUT to broadcast a coefficient directly into Z,
    //   * use AMX VECFP f64 for every Horner FMA.
    // Thus AMX is doing the whole approximation spine, not three tail FMAs.

    alignas(64) static const double coeff0[8] = {
        0x1.0000000000000p+0,
       -0x1.0000000000000p-1,
        0x1.5555555555554p-5,
       -0x1.6c16c16c16b71p-10,
        0x1.a01a01a0132f7p-16,
       -0x1.27e4fb74f50a4p-22,
        0x1.1eed8dcee263bp-29,
       -0x1.93969c9c98d05p-37
    };
    alignas(64) static const double coeff1[8] = {
        0x1.ae432cc0bd5d4p-45,
       -0x1.5ca3b5d23666ap-53,
        0.0,0.0,0.0,0.0,0.0,0.0
    };
    // Eight packed u4 index vectors. Pattern k contains eight copies of k.
    // GENLUT mode 10 expands one selected coefficient to all eight f64 lanes.
    alignas(64) static const uint32_t patterns[16] = {
        0x00000000u,0x11111111u,0x22222222u,0x33333333u,
        0x44444444u,0x55555555u,0x66666666u,0x77777777u,
        0,0,0,0,0,0,0,0
    };

    alignas(64) double zin[8], pout[8], rhbuf[8], rlbuf[8];
    alignas(64) int64_t qbuf[8];

    const float64x2_t vinvpi=vdupq_n_f64(INVPI);
    const float64x2_t zero=vdupq_n_f64(0.0);

    AMX_SET();
    // Y0/Y1 = coefficient tables, X1 = packed broadcast-index patterns.
    AMX_LDY((uint64_t)coeff0 | (0ull<<56));
    AMX_LDY((uint64_t)coeff1 | (1ull<<56));
    AMX_LDX((uint64_t)patterns | (1ull<<56));

    auto set_coeff_z0 = [](int k) {
        const uint64_t table=(k<8)?0ull:1ull;
        const uint64_t lane=(uint64_t)(k&7);
        // mode10: any64/u4 lookup; table from Y; destination Z0;
        // source is packed index vector at X1 + 4*lane.
        const uint64_t op=(10ull<<53) | (1ull<<59) | (table<<60) |
                          (1ull<<26) | (64ull + 4ull*lane);
        AMX_GENLUT(op);
    };

    for(size_t base=0;base<n;base+=8) {
        const size_t lanes=std::min<size_t>(8,n-base);
        bool fallback=false;
        for(int g=0;g<4;g++) {
            const int o=2*g;
            double xx[2];
            for(int k=0;k<2;k++) {
                size_t at=base+(size_t)o+(size_t)k;
                xx[k]=(at<base+lanes)?x[at]:x[base+lanes-1];
            }
            float64x2_t ax=vabsq_f64(vld1q_f64(xx));
            int64x2_t qi=vcvtnq_s64_f64(vmulq_f64(ax,vinvpi));
            vst1q_s64(qbuf+o,qi);
            if((uint64_t)qbuf[o]>=APPLE_COS53_REDN || (uint64_t)qbuf[o+1]>=APPLE_COS53_REDN) {
                fallback=true; continue;
            }
            double pha[2]={apple_cos53_pih[qbuf[o]],apple_cos53_pih[qbuf[o+1]]};
            double pla[2]={apple_cos53_pil[qbuf[o]],apple_cos53_pil[qbuf[o+1]]};
            float64x2_t ss=vsubq_f64(ax,vld1q_f64(pha));
            float64x2_t bb=vnegq_f64(vld1q_f64(pla));
            float64x2_t rh=vaddq_f64(ss,bb);
            float64x2_t bv=vsubq_f64(rh,ss);
            float64x2_t av=vsubq_f64(rh,bv);
            float64x2_t br=vsubq_f64(bb,bv);
            float64x2_t ar=vsubq_f64(ss,av);
            float64x2_t rl=vaddq_f64(ar,br);
            float64x2_t rs=vaddq_f64(rh,rl);
            uint64x2_t neg=vcltq_f64(rs,zero);
            rh=vbslq_f64(neg,vnegq_f64(rh),rh);
            rl=vbslq_f64(neg,vnegq_f64(rl),rl);
            vst1q_f64(rhbuf+o,rh); vst1q_f64(rlbuf+o,rl);
            float64x2_t r=vaddq_f64(rh,rl);
            vst1q_f64(zin+o,vmulq_f64(r,r));
        }
        if(fallback) {
            AMX_CLR();
            // Only outside the intended <=10K benchmark/validation domain.
            for(size_t k=0;k<lanes;k++) y[base+k]=std::cos(x[base+k]);
            if(base+lanes<n) cos53_eval_amx(x+base+lanes,y+base+lanes,n-base-lanes);
            return;
        }

        // Y2 = z=r^2. Horner p(z) uses no coefficient memory traffic here.
        AMX_LDY((uint64_t)zin | (2ull<<56));
        set_coeff_z0(9);
        for(int k=8;k>=0;k--) {
            AMX_EXTRX(0);               // Z0 -> X0 (current p)
            set_coeff_z0(k);            // Z0 = broadcast(c[k]) via GENLUT
            AMX_VECFP((7ull<<42)|128ull); // Z0 += X0 * Y2, f64
        }
        AMX_STZ((uint64_t)pout);

        // Near zeros use the already-established root-centred representation.
        // This is rare (~0.06% for phase-uniform data) and avoids asking a
        // global absolute-error polynomial to deliver relative ULPs at cos=0.
        static constexpr double PIO2_HI=0x1.921fb54442d18p+0;
        static constexpr double PIO2_LO=0x1.1a62633145c07p-54;
        static constexpr double PIO2_TINY=-0x1.f1976b7ed8fbcp-110;
        for(size_t k=0;k<lanes;k++) {
            double mag=pout[k];
            if(std::fabs(mag)<1.0e-3) {
                double e=((PIO2_HI-rhbuf[k]) + (PIO2_LO-rlbuf[k])) + PIO2_TINY;
                double e2=e*e;
                double q=std::fma(-1.0/5040.0,e2,1.0/120.0);
                q=std::fma(q,e2,-1.0/6.0);
                mag=std::fma(e*e2,q,e);
            }
            uint64_t u; std::memcpy(&u,&mag,8);
            if(qbuf[k]&1) u^=UINT64_C(0x8000000000000000);
            std::memcpy(y+base+k,&u,8);
        }
    }
    AMX_CLR();
}
'''
s=s[:start]+new+s[end:]

vs=s.index('static int verify_amx()')
ve=s.index('\nstatic int bench_one',vs)
verify=r'''static int verify_amx() {
    const size_t N=96000;
    std::vector<double> x(N),a(N),b(N);
    const double lo[3]={0.0,1.0,1000.0}, hi[3]={1.0,500.0,10000.0};
    for(size_t i=0;i<N;i++) {
        int c=(int)(i%6), d=c/2;
        double u=unit52(mix64(UINT64_C(2026090607)+(uint64_t)i*UINT64_C(0x9e3779b97f4a7c15)));
        double v=std::fma(hi[d]-lo[d],u,lo[d]); x[i]=(c&1)?-v:v;
    }
    // Include the four previously observed near-root danger cases.
    if(N>=4) {
        x[0]=-387.98693342237215; x[1]=-2492.8537705956187;
        x[2]=-64.402892910711032; x[3]=9778.2068913742987;
    }
    DirectAMXRunner amx; FrozenBaselineRunner baseline;
    amx.run(x.data(),a.data(),N); baseline.run(x.data(),b.data(),N);

    mpfr_t z,r; mpfr_init2(z,256); mpfr_init2(r,256);
    uint64_t ma=0,mb=0; size_t ae=0,a1=0,a2=0,be=0,b1=0,b2=0,diff=0;
    for(size_t i=0;i<N;i++) {
        mpfr_set_d(z,x[i],MPFR_RNDN); mpfr_cos(r,z,MPFR_RNDN);
        double ref=mpfr_get_d(r,MPFR_RNDN);
        uint64_t ua=ulpd(a[i],ref), ub=ulpd(b[i],ref);
        ma=std::max(ma,ua); mb=std::max(mb,ub);
        ae+=ua==0; a1+=ua<=1; a2+=ua<=2;
        be+=ub==0; b1+=ub<=1; b2+=ub<=2;
        diff+=std::memcmp(&a[i],&b[i],8)!=0;
    }
    mpfr_clear(r); mpfr_clear(z);
    std::printf("APPLE_COS53_AMX_NATIVE_VERIFY cases=%zu amx_exact=%zu amx_le1=%zu amx_le2=%zu amx_maxulp=%llu baseline_exact=%zu baseline_le1=%zu baseline_le2=%zu baseline_maxulp=%llu bitdiff=%zu reference=MPFR256\n",
        N,ae,a1,a2,(unsigned long long)ma,be,b1,b2,(unsigned long long)mb,diff);
    return a2==N ? 0 : 9;
}
'''
s=s[:vs]+verify+s[ve:]
p.write_text(s)
PY

  MP="$(brew --prefix mpfr)"; GP="$(brew --prefix gmp)"
  clang++ -O3 -DNDEBUG -DAPPLE_COS53_VALIDATE_MPFR -std=c++20 -mcpu=native \
    -fno-fast-math -ffp-contract=off -fblocks \
    -I/tmp -I/tmp/highway -I/tmp/pthreadpool-install/include -I/tmp/corsix-amx \
    -I"$MP/include" -I"$GP/include" \
    /tmp/base.cpp /tmp/pthreadpool-install/lib/libpthreadpool.a \
    -L"$MP/lib" -L"$GP/lib" -lmpfr -lgmp -framework Accelerate -pthread -ldl \
    -o /tmp/apple_cos53_amx_native
  exit 0
fi

if [[ "$MODE" == "probe" ]]; then exec /tmp/apple_cos53_amx_native probe; fi
if [[ "$MODE" == "verify" ]]; then exec /tmp/apple_cos53_amx_native verify; fi
if [[ "$MODE" == "one" ]]; then
  [[ $# -eq 3 ]]
  exec /tmp/apple_cos53_amx_native "$2" "$3"
fi

echo "usage: $0 build | probe | verify | one baseline|amx N" >&2
exit 2
