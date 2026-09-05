#pragma once

#if !defined(__APPLE__)
#error "apple_cos53_production_routing.hpp is Apple-only"
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <dispatch/dispatch.h>
#include <HalideRuntime.h>
#include <pthreadpool.h>

/*
 * Apple COS53 production routing, frozen baseline 2026-09-06.
 *
 * Kernel/accuracy state:
 *   - Google Highway 1.4.0 Apple path
 *   - frozen K=2048, degree=3, terms=1 constants/kernel
 *   - established MPFR256 validation state unchanged
 *
 * Batch routing:
 *              n <  5,000 : existing frozen AppleTwoCoreHighway path
 *    5,000 <= n < 30,000 : pthreadpool, exactly 2 tasks / 2 threads
 *   30,000 <= n <=40,000 : Halide runtime scheduler, 32 pieces
 *   40,001 <= n < 50,000 : dispatch_apply_f / DISPATCH_APPLY_AUTO, 32 pieces
 *   50,000 <= n <100,000 : dispatch_apply_f / DISPATCH_APPLY_AUTO, 16 pieces
 *  100,000 <= n <1,000,000: dispatch_apply_f / DISPATCH_APPLY_AUTO, 12 pieces
 * 1,000,000 <= n           : dispatch_apply_f / DISPATCH_APPLY_AUTO, 24 pieces
 *
 * Frozen pthreadpool contract:
 *   - Maratyszcza/pthreadpool
 *   - pthreadpool_create(2)
 *   - build with PTHREADPOOL_SYNC_PRIMITIVE=condvar on Apple
 *   - two balanced contiguous chunks, midpoint aligned to 2 doubles
 *   - no PTHREADPOOL_FLAG_YIELD_WORKERS (workers remain hot)
 *
 * Integration contract:
 *   - AppleTwoCoreHighway must be the already-frozen helper used below 5K.
 *   - cos53_eval_hwy(const double*, double*, size_t) must be the already-frozen
 *     K=2048 degree-3 Highway evaluator in the including translation unit.
 *   - link pthreadpool built with the condvar backend and a Halide standalone
 *     runtime providing halide_do_par_for and halide_set_num_threads.
 *
 * This file changes orchestration only. It does not replace the COS53 math.
 */

namespace apple_cos53_production {

static constexpr std::size_t kPThreadPoolBegin = 5000;
static constexpr std::size_t kDispatchThreshold = 30000;
static constexpr std::size_t kHalideEndInclusive = 40000;
static constexpr std::size_t kP32End = 50000;
static constexpr std::size_t kP16End = 100000;
static constexpr std::size_t kP12End = 1000000;
static constexpr std::size_t kHalidePieces = 32;
static constexpr int kHalideThreads = 3;

static inline std::size_t frozen_piece_count(std::size_t n) noexcept {
    if (n < kDispatchThreshold) return 0;
    if (n < kP32End) return 32;
    if (n < kP16End) return 16;
    if (n < kP12End) return 12;
    return 24;
}

static inline bool use_pthreadpool(std::size_t n) noexcept {
    return n >= kPThreadPoolBegin && n < kDispatchThreshold;
}

static inline bool use_halide(std::size_t n) noexcept {
    return n >= kDispatchThreshold && n <= kHalideEndInclusive;
}

struct PThreadPoolCtx {
    const double* x;
    double* y;
    std::size_t n;
    std::size_t mid;
};

static inline void pthreadpool_piece(void* vp, std::size_t task) {
    auto* c = static_cast<PThreadPoolCtx*>(vp);
    if (task == 0) {
        cos53_eval_hwy(c->x, c->y, c->mid);
    } else {
        cos53_eval_hwy(c->x + c->mid, c->y + c->mid, c->n - c->mid);
    }
}

class FrozenPThreadPool2 {
    pthreadpool_t pool_;

public:
    FrozenPThreadPool2() : pool_(pthreadpool_create(2)) {}
    ~FrozenPThreadPool2() {
        if (pool_ != nullptr) pthreadpool_destroy(pool_);
    }

    FrozenPThreadPool2(const FrozenPThreadPool2&) = delete;
    FrozenPThreadPool2& operator=(const FrozenPThreadPool2&) = delete;

    void run(const double* x, double* y, std::size_t n) {
        if (n == 0) return;
        if (pool_ == nullptr || n < 4) {
            cos53_eval_hwy(x, y, n);
            return;
        }

        const std::size_t mid = (n / 2) & ~std::size_t(1);
        PThreadPoolCtx ctx{x, y, n, mid};
        pthreadpool_parallelize_1d(pool_, pthreadpool_piece, &ctx, 2, 0);
    }
};

static inline FrozenPThreadPool2& frozen_pthreadpool2() {
    static FrozenPThreadPool2 pool;
    return pool;
}

struct ApplyCtx {
    const double* x;
    double* y;
    std::size_t n;
    std::size_t pieces;
};

static inline void apply_piece(void* vp, std::size_t j) {
    auto* c = static_cast<ApplyCtx*>(vp);
    const std::size_t a = (c->n * j) / c->pieces;
    const std::size_t b = (c->n * (j + 1)) / c->pieces;
    cos53_eval_hwy(c->x + a, c->y + a, b - a);
}

static inline int halide_apply_piece(void*, int j, uint8_t* closure) {
    auto* c = reinterpret_cast<ApplyCtx*>(closure);
    const std::size_t jj = static_cast<std::size_t>(j);
    const std::size_t a = (c->n * jj) / c->pieces;
    const std::size_t b = (c->n * (jj + 1)) / c->pieces;
    cos53_eval_hwy(c->x + a, c->y + a, b - a);
    return 0;
}

static inline void run_stable_dispatch(const double* x,
                                       double* y,
                                       std::size_t n,
                                       std::size_t pieces) {
    const std::size_t p = std::min(pieces, n);
    ApplyCtx ctx{x, y, n, p};
    dispatch_apply_f(p, DISPATCH_APPLY_AUTO, &ctx, apply_piece);
}

static inline void ensure_halide_threads() {
    static const bool initialized = []() {
        halide_set_num_threads(kHalideThreads);
        return true;
    }();
    (void)initialized;
}

static inline void run_halide_dispatch(const double* x,
                                       double* y,
                                       std::size_t n,
                                       std::size_t pieces) {
    ensure_halide_threads();
    const std::size_t p = std::min(pieces, n);
    ApplyCtx ctx{x, y, n, p};
    const int rc = halide_do_par_for(nullptr,
                                     halide_apply_piece,
                                     0,
                                     static_cast<int>(p),
                                     reinterpret_cast<uint8_t*>(&ctx));
    if (rc != 0) {
        run_stable_dispatch(x, y, n, pieces);
    }
}

/*
 * Single production entry point for Apple routing.
 * tc is persistent and owned by the caller for the <5K route.
 */
static inline void run(AppleTwoCoreHighway& tc,
                       const double* x,
                       double* y,
                       std::size_t n) {
    if (n < kPThreadPoolBegin) {
        tc.run(x, y, n);
        return;
    }

    if (use_pthreadpool(n)) {
        frozen_pthreadpool2().run(x, y, n);
        return;
    }

    if (use_halide(n)) {
        run_halide_dispatch(x, y, n, kHalidePieces);
        return;
    }

    run_stable_dispatch(x, y, n, frozen_piece_count(n));
}

} // namespace apple_cos53_production
