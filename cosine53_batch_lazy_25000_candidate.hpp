#pragma once

/* EXPERIMENTAL ONLY: COS53 lazy-25000 routing candidate.

   Frozen production remains untouched at its 4500 activation boundary.

   Temporary experimental routing:
       n < 25000   -> current COS53 evaluator; custom2 helper does not exist
       n >= 25000  -> construct/use exact frozen permanent 2-core scheduler

   Mathematics and the frozen custom2 implementation are unchanged.
*/

#include <cstddef>
#include <memory>
#include "cosine53_custom_2core_1600_frozen.hpp"

class Cosine53BatchLazy25000Candidate {
public:
    using fn_t = void (*)(double *, const double *, size_t);
    static constexpr size_t kCustom2MinN = 25000;

    explicit Cosine53BatchLazy25000Candidate(fn_t current_eval)
        : current_eval_(current_eval) {}

    Cosine53BatchLazy25000Candidate(const Cosine53BatchLazy25000Candidate&) = delete;
    Cosine53BatchLazy25000Candidate& operator=(const Cosine53BatchLazy25000Candidate&) = delete;

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
