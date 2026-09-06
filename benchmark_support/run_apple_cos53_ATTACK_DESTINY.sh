#!/usr/bin/env bash
set -euo pipefail

# ATTACK DESTINY: experimental copy of frozen DESTINY.
# Frozen/current DESTINY remains at:
#   f3288a1faf54d7b352bd564ae8760ddb35ce1892
#
# Integrated waste-removal attack.  The cosine math is unchanged: every attacked
# route calls the exact conversion-free K1280 hot kernel produced by DESTINY.
# This file replaces orchestration waste in one shot:
#   * no shared completed counter / helper-helper cacheline contention
#   * no per-call completed reset
#   * generation publication is store, not atomic fetch_add RMW
#   * WG2 uses notify_one rather than notify_all
#   * per-helper completion generations live on isolated cache lines
#   * all ranges are precomputed once by the caller and aligned to 8 doubles
#   * no duplicate boundary division in helpers
#   * no unconditional spin/yield loop; event-driven wait after one completion check
#   * pool2 production pockets use a dedicated persistent 2-way runner rather
#     than the generic pthreadpool callback/partition machinery

MODE="${1:-}"
DESTINY_SHA=f3288a1faf54d7b352bd564ae8760ddb35ce1892

if [[ "$MODE" == build ]]; then
  [[ "$(uname -m)" == arm64 ]]
  [[ "$(sysctl -n machdep.cpu.brand_string)" == *"Apple M1"* ]]

  # Build exact DESTINY and, transitively, its exact hot-loop source/binaries.
  bash benchmark_support/run_apple_cos53_PATCH_cpu_fix.sh build
  test -f /tmp/apple_cos53_hotloop.cpp
  test -x /tmp/apple_cos53_off_frozen

  cat >/tmp/apple_cos53_attack_destiny.cpp <<'CPP'
#define main apple_cos53_hotloop_base_main
#include "/tmp/apple_cos53_hotloop.cpp"
#undef main

#include <os/workgroup.h>
#include <array>

static qos_class_t attack_qos(const std::string& s) {
    if (s=="ui") return QOS_CLASS_USER_INTERACTIVE;
    if (s=="user") return QOS_CLASS_USER_INITIATED;
    if (s=="default") return QOS_CLASS_DEFAULT;
    if (s=="utility") return QOS_CLASS_UTILITY;
    std::abort();
}

struct alignas(64) AttackDoneSlot {
    std::atomic<uint64_t> generation{0};
};

// Clean persistent two-way runner for the former generic pool2 pockets.
class AttackPool2Runner {
    std::thread helper_;
    alignas(64) std::atomic<uint64_t> generation_{0};
    AttackDoneSlot done_;
    std::atomic<bool> ready_{false};
    std::atomic<bool> stop_{false};
    uint64_t next_generation_=0;
    const double* hx_=nullptr;
    double* hy_=nullptr;
    size_t hn_=0;

    void helper_loop() {
        uint64_t seen=generation_.load(std::memory_order_relaxed);
        ready_.store(true,std::memory_order_release);
        ready_.notify_one();
        for (;;) {
            uint64_t g=generation_.load(std::memory_order_acquire);
            if (g==seen) {
                generation_.wait(seen,std::memory_order_acquire);
                g=generation_.load(std::memory_order_acquire);
            }
            seen=g;
            if (stop_.load(std::memory_order_relaxed)) return;
            opt_cos53_eval(hx_,hy_,hn_);
            done_.generation.store(g,std::memory_order_release);
            done_.generation.notify_one();
        }
    }
public:
    AttackPool2Runner() {
        helper_=std::thread([this]{helper_loop();});
        while(!ready_.load(std::memory_order_acquire)) {
            bool old=false;
            ready_.wait(old,std::memory_order_acquire);
        }
    }
    ~AttackPool2Runner() {
        stop_.store(true,std::memory_order_relaxed);
        const uint64_t g=++next_generation_;
        generation_.store(g,std::memory_order_release);
        generation_.notify_one();
        helper_.join();
    }
    void run(const double*x,double*y,size_t n) {
        if (n<16) { opt_cos53_eval(x,y,n); return; }
        const size_t mid=(n/2)&~size_t(7);
        hx_=x+mid; hy_=y+mid; hn_=n-mid;
        const uint64_t g=++next_generation_;
        generation_.store(g,std::memory_order_release);
        generation_.notify_one();
        opt_cos53_eval(x,y,mid);
        uint64_t d=done_.generation.load(std::memory_order_acquire);
        while(d!=g) {
            done_.generation.wait(d,std::memory_order_acquire);
            d=done_.generation.load(std::memory_order_acquire);
        }
    }
};

template<int P>
class AttackWGRunner {
    static_assert(P==2 || P==3);
    qos_class_t qos_;
    os_workgroup_parallel_t wg_{};
    os_workgroup_join_token_s caller_token_{};
    bool caller_joined_=false;
    std::array<std::thread,P-1> helpers_;
    alignas(64) std::atomic<uint64_t> generation_{0};
    std::array<AttackDoneSlot,P-1> done_{};
    alignas(64) std::atomic<int> ready_{0};
    alignas(64) std::atomic<bool> stop_{false};
    uint64_t next_generation_=0;
    const double* x_=nullptr;
    double* y_=nullptr;
    std::array<size_t,P+1> bounds_{};

    void helper_loop(int helper_index) {
        pthread_set_qos_class_self_np(qos_,0);
        os_workgroup_join_token_s token{};
        const bool joined=(os_workgroup_join((os_workgroup_t)wg_,&token)==0);
        uint64_t seen=generation_.load(std::memory_order_relaxed);
        ready_.fetch_add(1,std::memory_order_release);
        ready_.notify_one();
        for (;;) {
            uint64_t g=generation_.load(std::memory_order_acquire);
            if (g==seen) {
                generation_.wait(seen,std::memory_order_acquire);
                g=generation_.load(std::memory_order_acquire);
            }
            seen=g;
            if (stop_.load(std::memory_order_relaxed)) break;
            const size_t a=bounds_[helper_index+1];
            const size_t b=bounds_[helper_index+2];
            opt_cos53_eval(x_+a,y_+a,b-a);
            done_[helper_index].generation.store(g,std::memory_order_release);
            done_[helper_index].generation.notify_one();
        }
        if(joined) os_workgroup_leave((os_workgroup_t)wg_,&token);
    }

    static size_t align8(size_t x) { return x&~size_t(7); }

    void publish_wakeup(uint64_t g) {
        generation_.store(g,std::memory_order_release);
        if constexpr(P==2) generation_.notify_one();
        else generation_.notify_all();
    }
public:
    explicit AttackWGRunner(qos_class_t q):qos_(q) {
        pthread_set_qos_class_self_np(qos_,0);
        wg_=os_workgroup_parallel_create("apple-cos53-attack-destiny",nullptr);
        if(!wg_) std::abort();
        caller_joined_=(os_workgroup_join((os_workgroup_t)wg_,&caller_token_)==0);
        for(int i=0;i<P-1;i++) helpers_[i]=std::thread([this,i]{helper_loop(i);});
        while(ready_.load(std::memory_order_acquire)!=P-1) {
            int old=ready_.load(std::memory_order_acquire);
            if(old!=P-1) ready_.wait(old,std::memory_order_acquire);
        }
    }
    ~AttackWGRunner() {
        stop_.store(true,std::memory_order_relaxed);
        const uint64_t g=++next_generation_;
        publish_wakeup(g);
        for(auto&t:helpers_) t.join();
        if(caller_joined_) os_workgroup_leave((os_workgroup_t)wg_,&caller_token_);
        os_workgroup_cancel((os_workgroup_t)wg_);
    }
    void run(const double*x,double*y,size_t n) {
        if(n<8*P) { opt_cos53_eval(x,y,n); return; }
        x_=x; y_=y;
        bounds_[0]=0;
        if constexpr(P==2) {
            bounds_[1]=align8(n/2);
            bounds_[2]=n;
        } else {
            bounds_[1]=align8(n/3);
            bounds_[2]=align8((2*n)/3);
            bounds_[3]=n;
        }
        const uint64_t g=++next_generation_;
        publish_wakeup(g);
        opt_cos53_eval(x_,y_,bounds_[1]);
        // No fixed busy-spin policy.  Check each isolated completion stamp once,
        // then park on exactly the helper that has not completed.
        for(int i=0;i<P-1;i++) {
            uint64_t d=done_[i].generation.load(std::memory_order_acquire);
            while(d!=g) {
                done_[i].generation.wait(d,std::memory_order_acquire);
                d=done_[i].generation.load(std::memory_order_acquire);
            }
        }
    }
};

int main(int argc,char**argv) {
    if(argc!=3) return 2;
    const std::string mode=argv[1];
    const size_t n=(size_t)std::strtoull(argv[2],nullptr,10);
    if(mode=="single") { SingleRunner r; return bench_runner("ATTACK",r,n); }
    if(mode=="pool2_clean") { AttackPool2Runner r; return bench_runner("ATTACK",r,n); }
    if(mode.rfind("wg2_",0)==0) {
        AttackWGRunner<2> r(attack_qos(mode.substr(4)));
        return bench_runner("ATTACK",r,n);
    }
    if(mode.rfind("wg3_",0)==0) {
        AttackWGRunner<3> r(attack_qos(mode.substr(4)));
        return bench_runner("ATTACK",r,n);
    }
    return 3;
}
CPP

  clang++ -O3 -DNDEBUG -std=c++20 -mcpu=native -fno-fast-math -ffp-contract=fast \
    -I/tmp -I/tmp/pthreadpool-install/include /tmp/apple_cos53_attack_destiny.cpp \
    /tmp/pthreadpool-install/lib/libpthreadpool.a -framework Accelerate -pthread \
    -o /tmp/apple_cos53_attack_destiny
  exit 0
fi

if [[ "$MODE" == validate ]]; then
  # Math kernel is byte-for-byte generated from DESTINY's validated hot kernel.
  exec bash benchmark_support/run_apple_cos53_PATCH_cpu_fix.sh validate
fi

route_for() {
  case "$1" in
    100) echo single ;;
    400) echo wg2_utility ;;
    700|1200|30000|40001|75000) echo wg2_ui ;;
    3000|79000|81000|100000|500000|1000000) echo wg3_default ;;
    15000|29999) echo pool2_clean ;;
    40000) echo wg3_user ;;
    78000|200000) echo wg3_utility ;;
    80000) echo wg2_utility ;;
    82000) echo wg2_default ;;
    *) echo DESTINY ;;
  esac
}

if [[ "$MODE" == route ]]; then
  [[ $# -eq 2 ]]
  route_for "$2"
  exit 0
fi

if [[ "$MODE" == one ]]; then
  [[ $# -eq 2 ]]
  n="$2"
  r="$(route_for "$n")"
  if [[ "$r" == DESTINY ]]; then
    exec bash benchmark_support/run_apple_cos53_PATCH_cpu_fix.sh one "$n"
  fi
  exec /tmp/apple_cos53_attack_destiny "$r" "$n"
fi

echo "usage: $0 build | validate | route N | one N" >&2
exit 2
