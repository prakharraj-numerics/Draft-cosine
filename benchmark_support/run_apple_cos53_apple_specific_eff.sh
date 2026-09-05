#!/usr/bin/env bash
set -euo pipefail

[[ "$(uname -m)" == arm64 ]]
[[ "$(sysctl -n machdep.cpu.brand_string)" == *"Apple M1"* ]]

MODE="${1:-}"

if [[ "$MODE" == build ]]; then
  test -f /tmp/apple_cos53_cpu_eff.cpp
  cat >/tmp/apple_cos53_apple_specific.cpp <<'CPP'
#define main apple_cos53_base_main
#include "/tmp/apple_cos53_cpu_eff.cpp"
#undef main

#include <os/workgroup.h>
#include <cerrno>

static qos_class_t qos_for(const std::string& s) {
    if (s == "ui") return QOS_CLASS_USER_INTERACTIVE;
    if (s == "user") return QOS_CLASS_USER_INITIATED;
    if (s == "default") return QOS_CLASS_DEFAULT;
    if (s == "utility") return QOS_CLASS_UTILITY;
    std::abort();
}

class AppleWorkgroupRunner {
    size_t p_;
    qos_class_t qos_;
    os_workgroup_parallel_t wg_;
    os_workgroup_join_token_s caller_token_{};
    bool caller_joined_ = false;
    std::vector<std::thread> helpers_;
    std::atomic<uint64_t> generation_{0};
    std::atomic<int> completed_{0};
    std::atomic<int> ready_{0};
    std::atomic<bool> stop_{false};
    const double* x_ = nullptr;
    double* y_ = nullptr;
    size_t n_ = 0;

    static inline void relax() { __asm__ volatile("yield"); }

    size_t boundary(size_t k) const {
        if (k == 0) return 0;
        if (k == p_) return n_;
        return ((n_ * k) / p_) & ~size_t(1);
    }

    void helper_loop(size_t idx) {
        pthread_set_qos_class_self_np(qos_, 0);
        os_workgroup_join_token_s token{};
        const int jrc = os_workgroup_join((os_workgroup_t)wg_, &token);
        const bool joined = (jrc == 0);
        ready_.fetch_add(1, std::memory_order_release);
        ready_.notify_one();

        uint64_t seen = generation_.load(std::memory_order_relaxed);
        for (;;) {
            uint64_t g = generation_.load(std::memory_order_acquire);
            if (g == seen) {
                generation_.wait(seen, std::memory_order_acquire);
                g = generation_.load(std::memory_order_acquire);
            }
            seen = g;
            if (stop_.load(std::memory_order_relaxed)) break;
            const size_t a = boundary(idx), b = boundary(idx + 1);
            opt_cos53_eval(x_ + a, y_ + a, b - a);
            completed_.fetch_add(1, std::memory_order_release);
            completed_.notify_one();
        }
        if (joined) os_workgroup_leave((os_workgroup_t)wg_, &token);
    }

public:
    AppleWorkgroupRunner(size_t p, qos_class_t qos) : p_(p), qos_(qos) {
        pthread_set_qos_class_self_np(qos_, 0);
        wg_ = os_workgroup_parallel_create("apple-cos53-parallel", nullptr);
        if (!wg_) std::abort();
        const int jrc = os_workgroup_join((os_workgroup_t)wg_, &caller_token_);
        caller_joined_ = (jrc == 0);
        for (size_t i = 1; i < p_; ++i) helpers_.emplace_back([this, i] { helper_loop(i); });
        while (ready_.load(std::memory_order_acquire) != (int)p_ - 1) {
            int old = ready_.load(std::memory_order_acquire);
            if (old != (int)p_ - 1) ready_.wait(old, std::memory_order_acquire);
        }
    }

    ~AppleWorkgroupRunner() {
        stop_.store(true, std::memory_order_relaxed);
        generation_.fetch_add(1, std::memory_order_release);
        generation_.notify_all();
        for (auto& t : helpers_) t.join();
        if (caller_joined_) os_workgroup_leave((os_workgroup_t)wg_, &caller_token_);
        os_workgroup_cancel((os_workgroup_t)wg_);
    }

    void run(const double* x, double* y, size_t n) {
        if (n < 2 * p_) { opt_cos53_eval(x, y, n); return; }
        x_ = x; y_ = y; n_ = n;
        completed_.store(0, std::memory_order_relaxed);
        generation_.fetch_add(1, std::memory_order_release);
        generation_.notify_all();
        const size_t b = boundary(1);
        opt_cos53_eval(x_, y_, b);

        // Tiny latency-preserving spin, then event-driven park. This bounds
        // wasted CPU while avoiding a kernel sleep/wake for near-synchronous completion.
        for (int k = 0; k < 32; ++k) {
            if (completed_.load(std::memory_order_acquire) == (int)p_ - 1) return;
            relax();
        }
        int old = completed_.load(std::memory_order_acquire);
        while (old != (int)p_ - 1) {
            completed_.wait(old, std::memory_order_acquire);
            old = completed_.load(std::memory_order_acquire);
        }
    }
};

class QoSSingleRunner {
public:
    explicit QoSSingleRunner(qos_class_t q) { pthread_set_qos_class_self_np(q, 0); }
    void run(const double* x, double* y, size_t n) { opt_cos53_eval(x, y, n); }
};

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    const std::string mode = argv[1];
    const size_t n = (size_t)std::strtoull(argv[2], nullptr, 10);
    auto pos = mode.rfind('_');
    if (pos == std::string::npos) return 3;
    const std::string family = mode.substr(0, pos);
    const std::string qname = mode.substr(pos + 1);
    const qos_class_t q = qos_for(qname);

    if (family == "single") {
        QoSSingleRunner r(q);
        return bench_runner(mode.c_str(), r, n);
    }
    if (family == "wg2") {
        AppleWorkgroupRunner r(2, q);
        return bench_runner(mode.c_str(), r, n);
    }
    if (family == "wg3") {
        AppleWorkgroupRunner r(3, q);
        return bench_runner(mode.c_str(), r, n);
    }
    return 4;
}
CPP

  COMMON="-O3 -DNDEBUG -std=c++20 -mcpu=native -fno-fast-math -ffp-contract=off"
  clang++ $COMMON -I/tmp -I/tmp/pthreadpool-install/include \
    /tmp/apple_cos53_apple_specific.cpp /tmp/pthreadpool-install/lib/libpthreadpool.a \
    -framework Accelerate -pthread -o /tmp/apple_cos53_apple_specific_bench
  exit 0
fi

if [[ "$MODE" == one ]]; then
  [[ $# -eq 3 ]]
  exec /tmp/apple_cos53_apple_specific_bench "$2" "$3"
fi

echo "usage: $0 build | one {single|wg2|wg3}_{ui|user|default|utility} N" >&2
exit 2
