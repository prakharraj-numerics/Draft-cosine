#pragma once

/* Apple port of the frozen COS53/EXP53 Custom2 scheduler.
 *
 * Scheduling only: this does not alter cosine mathematics.
 *
 * Custom2 mechanics retained:
 *   - one permanent helper thread
 *   - release/acquire generation handoff
 *   - completion generation counter
 *   - no task queue, work stealing, allocation, or per-call thread creation
 *   - permanent spin worker between generations
 *
 * Apple-specific substitutions:
 *   - Linux CPU affinity is not portable to macOS, so caller/helper request
 *     QOS_CLASS_USER_INTERACTIVE.
 *   - The Xeon implementation split on a 32-double engine boundary. The Apple
 *     Highway kernel is 2-double Full128, so the Apple port uses a balanced
 *     half split aligned to 2 doubles. This preserves Custom2 scheduling while
 *     avoiding Intel-specific 32/68 imbalance at small Apple batches.
 *   - The helper uses the ARM YIELD hint while waiting for the next generation.
 */

#if !defined(__APPLE__)
#error "apple_cos53_custom2_core.hpp is Apple-only"
#endif

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <pthread.h>

class AppleCos53CustomPermanent2Core {
public:
    using fn_t = void (*)(const double *, double *, std::size_t);

    explicit AppleCos53CustomPermanent2Core(fn_t fn)
        : generation_(0), completed_(0), helper_ready_(false), stop_(false),
          in_(nullptr), out_(nullptr), n2_(0), fn_(fn) {
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
        helper_ = std::thread([this] { helper_loop(); });
        while (!helper_ready_.load(std::memory_order_acquire)) cpu_relax();
    }

    ~AppleCos53CustomPermanent2Core() {
        stop_.store(true, std::memory_order_relaxed);
        generation_.fetch_add(1, std::memory_order_release);
        if (helper_.joinable()) helper_.join();
    }

    AppleCos53CustomPermanent2Core(const AppleCos53CustomPermanent2Core&) = delete;
    AppleCos53CustomPermanent2Core& operator=(const AppleCos53CustomPermanent2Core&) = delete;

    void run(const double *in, double *out, std::size_t n) {
        if (!n) return;

        // Balanced Apple split, aligned to one Full128<double> vector (2 doubles).
        const std::size_t split = (n / 2) & ~std::size_t(1);
        if (split == 0 || split >= n) {
            fn_(in, out, n);
            return;
        }

        in_ = in + split;
        out_ = out + split;
        n2_ = n - split;

        const std::uint64_t g = generation_.fetch_add(1, std::memory_order_release) + 1;
        fn_(in, out, split);
        while (completed_.load(std::memory_order_acquire) != g) cpu_relax();
    }

private:
    static inline void cpu_relax() noexcept {
#if defined(__aarch64__) || defined(__arm64__)
        __asm__ volatile("yield");
#else
        std::this_thread::yield();
#endif
    }

    void helper_loop() {
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
        helper_ready_.store(true, std::memory_order_release);
        std::uint64_t seen = generation_.load(std::memory_order_relaxed);

        for (;;) {
            std::uint64_t g;
            while ((g = generation_.load(std::memory_order_acquire)) == seen) cpu_relax();
            seen = g;
            if (stop_.load(std::memory_order_relaxed)) return;

            const double *in = in_;
            double *out = out_;
            const std::size_t n = n2_;
            fn_(in, out, n);
            completed_.store(g, std::memory_order_release);
        }
    }

    std::thread helper_;
    alignas(64) std::atomic<std::uint64_t> generation_;
    alignas(64) std::atomic<std::uint64_t> completed_;
    alignas(64) std::atomic<bool> helper_ready_;
    std::atomic<bool> stop_;

    const double *in_;
    double *out_;
    std::size_t n2_;
    fn_t fn_;
};
