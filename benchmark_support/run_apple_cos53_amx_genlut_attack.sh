#!/usr/bin/env bash
set -euo pipefail

[[ "$(uname -m)" == arm64 ]]
[[ "$(sysctl -n machdep.cpu.brand_string)" == *"Apple M1"* ]]
MODE="${1:-}"

if [[ "$MODE" == "build" ]]; then
  # Reuse the already-audited frozen-baseline reconstruction and AMX toolchain.
  bash benchmark_support/run_apple_cos53_direct_amx_vs_frozen_below30k.sh build

  python3 - <<'PY'
from pathlib import Path
p=Path('/tmp/base.cpp')
s=p.read_text()
needle='static uint64_t mix64(uint64_t x)'
assert needle in s
s=s.replace(needle,'#include "benchmark_support/apple_cos53_amx_genlut_poly.hpp"\n\n'+needle,1)

vs=s.index('static int verify_amx()')
ve=s.index('\n#ifdef APPLE_COS53_VALIDATE_MPFR',vs)
replacement=r'''static inline void acc_ulp(uint64_t u,uint64_t& mx,uint64_t& e,uint64_t& l1,uint64_t& l2,uint64_t& l3,uint64_t& g3) {
    mx=std::max(mx,u); if(u==0)e++; if(u<=1)l1++; if(u<=2)l2++; if(u<=3)l3++; else g3++;
}

static int verify_amx() {
    const size_t N=9600;
    uint64_t bg_m=0,bg_e=0,bg_1=0,bg_2=0,bg_3=0,bg_g3=0;
    uint64_t gg_m=0,gg_e=0,gg_1=0,gg_2=0,gg_3=0,gg_g3=0;
    uint64_t ig_m=0,ig_e=0,ig_1=0,ig_2=0,ig_3=0,ig_g3=0;
    size_t gen_idx_diff=0;
    mpfr_t z,r; mpfr_init2(z,256); mpfr_init2(r,256);

    FrozenBaselineRunner baseline;
    apple_cos53_amx_genlut::Runner gen(true), idx(false);
    for(int c=0;c<6;c++) {
        Buffers b(N);
        // Buffers fills each of its six arrays with a distinct sign/domain case;
        // select c so all six benchmark domains are covered.
        std::vector<double> bo(N),go(N),io(N);
        baseline.run(b.x[c],bo.data(),N);
        gen.run(b.x[c],go.data(),N);
        idx.run(b.x[c],io.data(),N);
        for(size_t i=0;i<N;i++) {
            mpfr_set_d(z,b.x[c][i],MPFR_RNDN); mpfr_cos(r,z,MPFR_RNDN);
            double ref=mpfr_get_d(r,MPFR_RNDN);
            acc_ulp(ulpd(bo[i],ref),bg_m,bg_e,bg_1,bg_2,bg_3,bg_g3);
            acc_ulp(ulpd(go[i],ref),gg_m,gg_e,gg_1,gg_2,gg_3,gg_g3);
            acc_ulp(ulpd(io[i],ref),ig_m,ig_e,ig_1,ig_2,ig_3,ig_g3);
            gen_idx_diff += std::memcmp(&go[i],&io[i],8)!=0;
        }
    }

    // Explicit near-zero attack: doubles immediately around many odd pi/2 roots.
    const int ROOTN=8192;
    std::vector<double> rx(ROOTN),bo(ROOTN),go(ROOTN),io(ROOTN);
    for(int i=0;i<ROOTN;i++) {
        int m=(i%4096)-2048;
        long double root=((long double)m+0.5L)*acosl(-1.0L);
        double d=(double)root;
        int step=(i&1)?1:-1;
        rx[i]=step>0?std::nextafter(d,INFINITY):std::nextafter(d,-INFINITY);
    }
    baseline.run(rx.data(),bo.data(),ROOTN);
    gen.run(rx.data(),go.data(),ROOTN);
    idx.run(rx.data(),io.data(),ROOTN);
    uint64_t root_b=0,root_g=0,root_i=0;
    for(int i=0;i<ROOTN;i++) {
        mpfr_set_d(z,rx[i],MPFR_RNDN); mpfr_cos(r,z,MPFR_RNDN);
        double ref=mpfr_get_d(r,MPFR_RNDN);
        root_b=std::max(root_b,ulpd(bo[i],ref));
        root_g=std::max(root_g,ulpd(go[i],ref));
        root_i=std::max(root_i,ulpd(io[i],ref));
        gen_idx_diff += std::memcmp(&go[i],&io[i],8)!=0;
    }
    mpfr_clear(r); mpfr_clear(z);

    std::printf("AMX_GENLUT_VERIFY stack=baseline total=%zu exact=%llu le1=%llu le2=%llu le3=%llu gt3=%llu maxulp=%llu root_maxulp=%llu\n",
        N*6,(unsigned long long)bg_e,(unsigned long long)bg_1,(unsigned long long)bg_2,(unsigned long long)bg_3,(unsigned long long)bg_g3,(unsigned long long)bg_m,(unsigned long long)root_b);
    std::printf("AMX_GENLUT_VERIFY stack=amxgen total=%zu exact=%llu le1=%llu le2=%llu le3=%llu gt3=%llu maxulp=%llu root_maxulp=%llu\n",
        N*6,(unsigned long long)gg_e,(unsigned long long)gg_1,(unsigned long long)gg_2,(unsigned long long)gg_3,(unsigned long long)gg_g3,(unsigned long long)gg_m,(unsigned long long)root_g);
    std::printf("AMX_GENLUT_VERIFY stack=amxidx total=%zu exact=%llu le1=%llu le2=%llu le3=%llu gt3=%llu maxulp=%llu root_maxulp=%llu\n",
        N*6,(unsigned long long)ig_e,(unsigned long long)ig_1,(unsigned long long)ig_2,(unsigned long long)ig_3,(unsigned long long)ig_g3,(unsigned long long)ig_m,(unsigned long long)root_i);
    std::printf("AMX_GENLUT_VERIFY_DONE gen_idx_bitdiff=%zu\n",gen_idx_diff);
    return 0;
}

static int bench_one(const std::string& stack,size_t n) {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0);
    Buffers b(n); size_t reps=reps_for(n); double cpu=0,wall=0;
    if(stack=="baseline") { FrozenBaselineRunner r; wall=measure_runner(r,b,reps,&cpu); }
    else if(stack=="amxgen") { apple_cos53_amx_genlut::Runner r(true); wall=measure_runner(r,b,reps,&cpu); }
    else if(stack=="amxidx") { apple_cos53_amx_genlut::Runner r(false); wall=measure_runner(r,b,reps,&cpu); }
    else return 3;
    std::printf("AMX_GENLUT_RESULT stack=%s n=%zu wall_ns_el=%.9f cpu_ns_el=%.9f effective_cores=%.6f reps=%zu sink=%.17g\n",
        stack.c_str(),n,wall,cpu,cpu/wall,reps,(double)g_sink);
    return 0;
}
'''
s=s[:vs]+replacement+s[ve:]
p.write_text(s)
PY

  FP="$(brew --prefix flint)"; MP="$(brew --prefix mpfr)"; GP="$(brew --prefix gmp)"
  clang++ -O3 -DNDEBUG -std=c++20 -mcpu=native -fno-fast-math -ffp-contract=off -fblocks \
    -DAPPLE_COS53_VALIDATE_MPFR \
    -I. -I/tmp -I/tmp/highway -I/tmp/pthreadpool-install/include -I/tmp/corsix-amx \
    -I"$MP/include" -I"$GP/include" \
    /tmp/base.cpp /tmp/pthreadpool-install/lib/libpthreadpool.a \
    -L"$MP/lib" -L"$GP/lib" -lmpfr -lgmp \
    -framework Accelerate -pthread -ldl -o /tmp/apple_cos53_amx_genlut
  exit 0
fi

if [[ "$MODE" == "probe" ]]; then
  exec /tmp/apple_cos53_amx_genlut probe
fi
if [[ "$MODE" == "verify" ]]; then
  exec /tmp/apple_cos53_amx_genlut verify
fi
if [[ "$MODE" == "one" ]]; then
  [[ $# -eq 3 ]]
  exec /tmp/apple_cos53_amx_genlut "$2" "$3"
fi

echo "usage: $0 build | probe | verify | one baseline|amxgen|amxidx N" >&2
exit 2
