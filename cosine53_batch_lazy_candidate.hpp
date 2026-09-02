#pragma once

/* EXPERIMENTAL COS53 lazy custom2 routing candidate.

   Goal: test small-load resource proportionality without touching frozen
   production or changing cosine mathematics/scheduling.

   Routing is identical to production:
       n < 1600   -> current COS53 evaluator
       n >= 1600  -> frozen custom permanent 2-core scheduler

   Difference from production:
     - the custom2 scheduler is NOT constructed in this object's constructor;
     - therefore no helper thread is created/pinned/spun for small-only usage;
     - on the first n >= 1600 call, the exact frozen custom2 object is created;
     - after creation it stays alive, preserving the permanent-worker behavior
       for subsequent large batches.

   This is an experiment only. Do not treat as production without benchmark
   evidence and explicit promotion.
*/

#include <cstddef>
#include <memory>
#include "cosine53_custom_2core_1600_frozen.hpp"

class Cosine53BatchLazyCandidate {
public:
    using fn_t = void (*)(double *, const double *, size_t);
    static constexpr size_t kCustom2MinN = 1600;

    explicit Cosine53BatchLazyCandidate(fn_t current_eval)
        : current_eval_(current_eval) {}

    Cosine53BatchLazyCandidate(const Cosine53BatchLazyCandidate&) = delete;
    Cosine53BatchLazyCandidate& operator=(const Cosine53BatchLazyCandidate&) = delete;

    void run(double *out, const double *in, size_t n) {
        if (n < kCustom2MinN) {
            current_eval_(out, in, n);
            return;
        }
        if (!custom2_) {
            custom2_ = std::make_unique<Cosine53CustomPermanent2Core1600Frozen>(current_eval_);
        }
        custom2_->run(out, in, n);
    }

    bool helper_started() const noexcept { return static_cast<bool>(custom2_); }

private:
    fn_t current_eval_;
    std::unique_ptr<Cosine53CustomPermanent2Core1600Frozen> custom2_;
};
