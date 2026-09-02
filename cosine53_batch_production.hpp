#pragma once

/* COS53 FROZEN production batch routing.

   Resource-elastic routing frozen from exact Intel Xeon 6973P-C evidence:

       n < 4500   -> current COS53 evaluator; custom2 helper does not exist
       n >= 4500  -> construct/use the exact frozen permanent 2-core scheduler

   Once constructed, the helper remains alive for later large batches.

   Evidence for this lifecycle/routing promotion:
     Full native + SDE benchmark: GitHub Actions run 33690497495.
     Exact Xeon 6973P-C artifacts: shards 3 and 20.
     Tested sizes: 100, 700, 3500, 4500, 5000, 8000, 15000, 50000,
                   1000000, 2000000.

   The promotion changes scheduling/lifecycle only. It does not change the COS53
   mathematics, the supplied evaluator, or the frozen custom2 implementation.

   At n=100, 700, and 3500 the helper remains absent, eliminating the previous
   permanent-worker resource overhead. At n=4500 and above the same frozen
   custom2 scheduler is used.

   Accuracy in the promoted benchmark remained within an observed maximum
   comparator difference of 2 ULP versus Intel VML_HA over the tested map.

   FROZEN: do not change the <4500 lazy region, the 4500 activation boundary,
   or the frozen scheduler without a new benchmark and explicit promotion.
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
