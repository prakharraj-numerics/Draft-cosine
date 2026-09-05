#pragma once

#if !defined(__APPLE__)
#error "apple_cos53_production_routing.hpp is Apple-only"
#endif

#include <algorithm>
#include <cstddef>
#include <dispatch/dispatch.h>

/*
 * Frozen Apple COS53 production routing, 2026-09-05.
 *
 * Kernel/accuracy state:
 *   - Google Highway 1.4.0 Apple path
 *   - frozen K=2048, degree=3, terms=1 constants/kernel
 *   - established 9600-case MPFR256 validator state: max <= 2 ULP
 *
 * Batch routing:
 *   n <      30,000 : existing frozen AppleTwoCoreHighway path
 *   30,000 <= n < 50,000 : dispatch_apply_f / DISPATCH_APPLY_AUTO, 32 pieces
 *   50,000 <= n <100,000 : dispatch_apply_f / DISPATCH_APPLY_AUTO, 16 pieces
 *  100,000 <= n <1,000,000: dispatch_apply_f / DISPATCH_APPLY_AUTO, 12 pieces
 * 1,000,000 <= n          : dispatch_apply_f / DISPATCH_APPLY_AUTO, 24 pieces
 *
 * Integration contract:
 *   - AppleTwoCoreHighway must be the already-frozen <30K helper.
 *   - cos53_eval_hwy(const double*, double*, size_t) must be the already-frozen
 *     K=2048 degree-3 Highway evaluator in the including translation unit.
 *
 * This file changes orchestration only. It does not replace the COS53 math.
 */

namespace apple_cos53_production {

static constexpr std::size_t kDispatchThreshold = 30000;
static constexpr std::size_t kP32End = 50000;
static constexpr std::size_t kP16End = 100000;
static constexpr std::size_t kP12End = 1000000;

static inline std::size_t frozen_piece_count(std::size_t n) noexcept {
    if (n < kDispatchThreshold) return 0;  // existing AppleTwoCoreHighway
    if (n < kP32End) return 32;
    if (n < kP16End) return 16;
    if (n < kP12End) return 12;
    return 24;
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

static inline void run_stable_dispatch(const double* x,
                                       double* y,
                                       std::size_t n,
                                       std::size_t pieces) {
    const std::size_t p = std::min(pieces, n);
    ApplyCtx ctx{x, y, n, p};
    dispatch_apply_f(p, DISPATCH_APPLY_AUTO, &ctx, apply_piece);
}

/*
 * Single production entry point for the frozen Apple routing map.
 * tc is persistent and owned by the caller, matching the existing <30K path.
 */
static inline void run(AppleTwoCoreHighway& tc,
                       const double* x,
                       double* y,
                       std::size_t n) {
    const std::size_t pieces = frozen_piece_count(n);
    if (pieces == 0) {
        tc.run(x, y, n);
        return;
    }
    run_stable_dispatch(x, y, n, pieces);
}

} // namespace apple_cos53_production
