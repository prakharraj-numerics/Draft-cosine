#pragma once

/* Production batch routing:
   n < 4500  : current COS53 evaluator
   n >= 4500 : permanent two-core scheduler

   The scheduler is created lazily and retained after first use.
   Keep the threshold and scheduler unchanged unless they are re-benchmarked.
*/

#include <cstddef>
#include <memory>
#include "cosine53_custom_2core_1600_frozen.hpp"

class Cosine53BatchProductionFrozen {
public:
    using fn_t = void (*)(double *, const double *, size_t);
    static constexpr size_t kCustom2MinN = 4500;

    explicit Cosine53BatchProductionFrozen(fn_t current_eval)
        : current_eval_(current_eval) {}

    Cosine53BatchProductionFrozen(const Cosine53BatchProductionFrozen&) = delete;
    Cosine53BatchProductionFrozen& operator=(const Cosine53BatchProductionFrozen&) = delete;

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
