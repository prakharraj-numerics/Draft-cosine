#pragma once

// Experimental Apple AMX-native COS53 backend.
// Design goal: use AMX for what the hardware is unusually good at rather than
// shipping only three existing Horner FMAs to AMX.
//
// 1. CPU/NEON performs the wide-range pi reduction.
// 2. Residual r in [0, pi/2] is divided into 8 coarse pieces.
// 3. AMX GENLUT performs the per-lane piece selection / coefficient gather.
// 4. NEON evaluates universal small-angle cos(t), sin(t) polynomials.
// 5. AMX VECFP combines cos(center)*cos(t) + sin(center)*sin(t).
//
// The last piece is centered exactly at pi/2, so near roots the result is
// sin(pi/2-r), avoiding the coefficient-cancellation pathology of a generic
// local cosine polynomial.

namespace apple_cos53_amx_genlut {

alignas(64) static constexpr double kThresholds[8] = {
    0x0.0p+0,
    0x1.921fb54442d18p-3,
    0x1.921fb54442d18p-2,
    0x1.2d97c7f3321d2p-1,
    0x1.921fb54442d18p-1,
    0x1.f6a7a2955385ep-1,
    0x1.2d97c7f3321d2p+0,
    0x1.5fdbbe9bba775p+0,
};

// Centers for pieces 0..6 are midpoints. Piece 7 is deliberately centered at
// pi/2 so its local identity is exactly cos(pi/2-t)=sin(t).
alignas(64) static constexpr double kCenterHi[8] = {
    0x1.921fb54442d18p-4,
    0x1.2d97c7f3321d2p-2,
    0x1.f6a7a2955385ep-2,
    0x1.5fdbbe9bba775p-1,
    0x1.c463abeccb2bbp-1,
    0x1.1475cc9eedf01p+0,
    0x1.46b9c347764a4p+0,
    0x1.921fb54442d18p+0,
};
alignas(64) static constexpr double kCenterLo[8] = {
    0x1.1a62633145c07p-58,
    0x1.a79394c9e8a0ap-57,
    0x1.60fafbfd97309p-56,
    0x1.ee2c2d963a10cp-56,
    0x1.3daeaf976e788p-55,
   -0x1.3ddc5bce200bbp-54,
   -0x1.1a900f67f753ap-54,
    0x1.1a62633145c07p-54,
};
static constexpr double kPio2Tiny = -0x1.f1976b7ed8fbcp-110;

alignas(64) static constexpr double kCosCenter[8] = {
    0x1.fd88da3d12526p-1,
    0x1.e9f4156c62ddap-1,
    0x1.c38b2f180bdb1p-1,
    0x1.8bc806b151741p-1,
    0x1.44cf325091dd6p-1,
    0x1.e2b5d3806f63bp-2,
    0x1.294062ed59f06p-2,
    0x0.0p+0,
};
alignas(64) static constexpr double kSinCenter[8] = {
    0x1.917a6bc29b42cp-4,
    0x1.294062ed59f06p-2,
    0x1.e2b5d3806f63bp-2,
    0x1.44cf325091dd6p-1,
    0x1.8bc806b151741p-1,
    0x1.c38b2f180bdb1p-1,
    0x1.e9f4156c62ddap-1,
    0x1.0000000000000p+0,
};
alignas(64) static constexpr double kZero[8] = {0,0,0,0,0,0,0,0};

static constexpr double kInvPieceWidth = 0x1.45f306dc9c883p+2; // 16/pi

static inline uint64_t ldst_reg(const void* p, unsigned reg) {
    return (uint64_t)p | ((uint64_t)reg << 56);
}

// GENLUT operands. f64 generate uses 4-bit indices with the high bit forced to
// zero on M1, i.e. exactly our eight pieces.
static constexpr uint64_t kGenIndex =
    (7ull << 60) | (1ull << 59) | (2ull << 53) | (1ull << 20) | 256ull;
// source X1 (byte offset 64), table Y5/Y6, f64/u4 lookup, dest X2/X3.
static constexpr uint64_t kLookupCos =
    (5ull << 60) | (1ull << 59) | (10ull << 53) | (2ull << 20) | 64ull;
static constexpr uint64_t kLookupSin =
    (6ull << 60) | (1ull << 59) | (10ull << 53) | (3ull << 20) | 64ull;
// VECFP f64: Z0 += X2*Y0, then Z0 += X3*Y1.
static constexpr uint64_t kVecCos = (7ull << 42) | (128ull << 10);
static constexpr uint64_t kVecSin = (7ull << 42) | (192ull << 10) | 64ull;

static inline void eval_small_cs(const double* t, double* cv, double* sv) {
    for (int g=0; g<4; ++g) {
        const int o=2*g;
        float64x2_t tv=vld1q_f64(t+o);
        float64x2_t z2=vmulq_f64(tv,tv);

        // cos(t), degree 10, |t| <= pi/16.
        float64x2_t pc=vdupq_n_f64(-1.0/3628800.0);
        pc=vfmaq_f64(vdupq_n_f64( 1.0/40320.0),   pc,z2);
        pc=vfmaq_f64(vdupq_n_f64(-1.0/720.0),     pc,z2);
        pc=vfmaq_f64(vdupq_n_f64( 1.0/24.0),      pc,z2);
        pc=vfmaq_f64(vdupq_n_f64(-0.5),           pc,z2);
        float64x2_t c=vfmaq_f64(vdupq_n_f64(1.0), pc,z2);

        // sin(t), degree 11.
        float64x2_t ps=vdupq_n_f64(-1.0/39916800.0);
        ps=vfmaq_f64(vdupq_n_f64( 1.0/362880.0), ps,z2);
        ps=vfmaq_f64(vdupq_n_f64(-1.0/5040.0),   ps,z2);
        ps=vfmaq_f64(vdupq_n_f64( 1.0/120.0),    ps,z2);
        ps=vfmaq_f64(vdupq_n_f64(-1.0/6.0),      ps,z2);
        float64x2_t s=vmulq_f64(tv,vfmaq_f64(vdupq_n_f64(1.0),ps,z2));

        vst1q_f64(cv+o,c);
        vst1q_f64(sv+o,s);
    }
}

// hw_generate=true uses GENLUT's f64 search instruction to derive the eight
// 4-bit indices. false supplies the same packed indices from the CPU, allowing
// us to measure whether GENLUT generation itself is useful or only its lookup.
static inline void eval(const double* x, double* y, size_t n, bool hw_generate) {
    if (!n) return;
    AMX_SET();

    // These three tables remain resident in AMX for the entire chunk.
    AMX_LDY(ldst_reg(kCosCenter,5));
    AMX_LDY(ldst_reg(kSinCenter,6));
    AMX_LDY(ldst_reg(kThresholds,7));

    alignas(64) double xin[8], rarr[8], rharr[8], rlarr[8];
    alignas(64) double tarr[8], carr[8], sarr[8], pout[8];
    alignas(64) uint8_t idxbuf[64];
    int64_t qbuf[8];

    const float64x2_t vinvpi=vdupq_n_f64(INVPI);
    const float64x2_t z=vdupq_n_f64(0.0);

    for (size_t base=0; base<n; base+=8) {
        const size_t lanes=std::min<size_t>(8,n-base);
        for(size_t k=0;k<lanes;k++) xin[k]=x[base+k];
        for(size_t k=lanes;k<8;k++) xin[k]=xin[lanes-1];

        bool fallback=false;
        for(int g=0;g<4;g++) {
            const int o=2*g;
            float64x2_t ax=vabsq_f64(vld1q_f64(xin+o));
            int64x2_t qi=vcvtnq_s64_f64(vmulq_f64(ax,vinvpi));
            vst1q_s64(qbuf+o,qi);
            if ((uint64_t)qbuf[o] >= APPLE_COS53_REDN ||
                (uint64_t)qbuf[o+1] >= APPLE_COS53_REDN) {
                fallback=true;
                continue;
            }
            double pha[2]={apple_cos53_pih[qbuf[o]],apple_cos53_pih[qbuf[o+1]]};
            double pla[2]={apple_cos53_pil[qbuf[o]],apple_cos53_pil[qbuf[o+1]]};
            float64x2_t ph=vld1q_f64(pha), pl=vld1q_f64(pla);
            float64x2_t ss=vsubq_f64(ax,ph);
            float64x2_t bb=vnegq_f64(pl);
            float64x2_t rh=vaddq_f64(ss,bb);
            float64x2_t bv=vsubq_f64(rh,ss);
            float64x2_t av=vsubq_f64(rh,bv);
            float64x2_t br=vsubq_f64(bb,bv);
            float64x2_t ar=vsubq_f64(ss,av);
            float64x2_t rl=vaddq_f64(ar,br);
            float64x2_t rs=vaddq_f64(rh,rl);
            uint64x2_t neg=vcltq_f64(rs,z);
            rh=vbslq_f64(neg,vnegq_f64(rh),rh);
            rl=vbslq_f64(neg,vnegq_f64(rl),rl);
            vst1q_f64(rharr+o,rh);
            vst1q_f64(rlarr+o,rl);
            vst1q_f64(rarr+o,vaddq_f64(rh,rl));
        }

        if(fallback) {
            AMX_CLR();
            for(size_t k=0;k<lanes;k++) y[base+k]=std::cos(x[base+k]);
            if(base+lanes<n) eval(x+base+lanes,y+base+lanes,n-base-lanes,hw_generate);
            return;
        }

        uint32_t packed=0;
        for(int k=0;k<8;k++) {
            int j=(int)(rarr[k]*kInvPieceWidth);
            if(j<0) j=0; else if(j>7) j=7;
            packed |= (uint32_t)j << (4*k);
            double t=(kCenterHi[j]-rharr[k]) + (kCenterLo[j]-rlarr[k]);
            if(j==7) t += kPio2Tiny;
            tarr[k]=t;
        }
        eval_small_cs(tarr,carr,sarr);

        if(hw_generate) {
            AMX_LDX(ldst_reg(rarr,4));
            AMX_GENLUT(kGenIndex); // -> packed f64 piece indices in X1
        } else {
            std::memset(idxbuf,0,sizeof(idxbuf));
            std::memcpy(idxbuf,&packed,sizeof(packed));
            AMX_LDX(ldst_reg(idxbuf,1));
        }

        AMX_GENLUT(kLookupCos); // X2 = cos(center[index])
        AMX_GENLUT(kLookupSin); // X3 = sin(center[index])
        AMX_LDY(ldst_reg(carr,0));
        AMX_LDY(ldst_reg(sarr,1));
        AMX_LDZ((uint64_t)kZero);
        AMX_VECFP(kVecCos);
        AMX_VECFP(kVecSin);
        AMX_STZ((uint64_t)pout);

        for(int g=0;g<4;g++) {
            const int o=2*g;
            uint64_t smv[2]={
                (qbuf[o]&1)?UINT64_C(0x8000000000000000):0,
                (qbuf[o+1]&1)?UINT64_C(0x8000000000000000):0
            };
            uint64x2_t pu=vreinterpretq_u64_f64(vld1q_f64(pout+o));
            pu=veorq_u64(pu,vld1q_u64(smv));
            vst1q_f64(pout+o,vreinterpretq_f64_u64(pu));
        }
        std::memcpy(y+base,pout,lanes*sizeof(double));
    }
    AMX_CLR();
}

class TwoCore {
    pthread_t th_{};
    dispatch_semaphore_t wake_;
    std::atomic<uint64_t> seq_{0},done_{0};
    std::atomic<bool> stop_{false};
    const double* x_=nullptr; double* y_=nullptr; size_t begin_=0,end_=0;
    bool hw_;
    static void* entry(void* p){((TwoCore*)p)->loop();return nullptr;}
    void loop(){
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0);
        uint64_t seen=0;
        for(;;){
            uint64_t s=seq_.load(std::memory_order_acquire);
            if(s!=seen){
                seen=s;
                if(stop_.load(std::memory_order_relaxed)) break;
                eval(x_+begin_,y_+begin_,end_-begin_,hw_);
                done_.store(seen,std::memory_order_release);
                for(int k=0;k<4000;k++){
                    uint64_t q=seq_.load(std::memory_order_acquire);
                    if(q!=seen) break;
                    __asm__ volatile("yield");
                }
                continue;
            }
            dispatch_semaphore_wait(wake_,DISPATCH_TIME_FOREVER);
        }
    }
public:
    explicit TwoCore(bool hw):wake_(dispatch_semaphore_create(0)),hw_(hw){
        pthread_create(&th_,nullptr,&TwoCore::entry,this);
    }
    ~TwoCore(){
        stop_.store(true,std::memory_order_relaxed);
        seq_.fetch_add(1,std::memory_order_release);
        dispatch_semaphore_signal(wake_);
        pthread_join(th_,nullptr);
    }
    void run(const double*x,double*y,size_t n){
        size_t mid=(n/2)&~(size_t)1;
        x_=x;y_=y;begin_=mid;end_=n;
        uint64_t s=seq_.fetch_add(1,std::memory_order_acq_rel)+1;
        dispatch_semaphore_signal(wake_);
        eval(x,y,mid,hw_);
        while(done_.load(std::memory_order_acquire)!=s) __asm__ volatile("yield");
    }
};

struct PoolCtx2 {const double*x;double*y;size_t n,mid;bool hw;};
static inline void pool_piece2(void* vp,size_t task){
    auto*c=static_cast<PoolCtx2*>(vp);
    size_t a=task?c->mid:0,b=task?c->n:c->mid;
    eval(c->x+a,c->y+a,b-a,c->hw);
}

class Runner {
    TwoCore tc_;
    pthreadpool_t pool_;
    bool hw_;
public:
    explicit Runner(bool hw):tc_(hw),pool_(pthreadpool_create(2)),hw_(hw){if(!pool_)std::abort();}
    ~Runner(){pthreadpool_destroy(pool_);}
    void run(const double*x,double*y,size_t n){
        if(n<5000){tc_.run(x,y,n);return;}
        size_t mid=(n/2)&~(size_t)1;
        PoolCtx2 c{x,y,n,mid,hw_};
        pthreadpool_parallelize_1d(pool_,pool_piece2,&c,2,0);
    }
};

} // namespace apple_cos53_amx_genlut
