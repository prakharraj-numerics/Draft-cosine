#pragma once

/* EXPERIMENTAL COS53 lazy-4500 routing candidate.

   Goal: extend the resource-elastic single-core region while leaving frozen
   production and cosine mathematics untouched.

   Experimental routing:
       n < 4500   -> current COS53 evaluator; custom2 helper does not exist
       n >= 4500  -> construct/use exact frozen permanent 2-core scheduler

   Once constructed, the frozen helper remains alive for later large batches.
   This file is experimental only and must not be treated as production without
   exact-Xeon benchmark evidence and explicit promotion.
*/

#include <cstddef>
#include <memory>
#include "cosine53_custom_2core_1600_frozen.hpp"

class Cosine53BatchLazy4500Candidate {
public:
    using fn_t = void (*)(double *, const double *, size_t);
    static constexpr size_t kCustom2MinN = 4500;

    explicit Cosine53BatchLazy4500Candidate(fn_t current_eval)
        : current_eval_(current_eval) {}

    Cosine53BatchLazy4500Candidate(const Cosine53BatchLazy4500Candidate&) = delete;
    Cosine53BatchLazy4500Candidate& operator=(const Cosine53BatchLazy4500Candidate&) = delete;

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
