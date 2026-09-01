#pragma once

/* COS53 FROZEN production batch routing.

   Routing rule frozen from exact Intel Xeon 6973P-C benchmark evidence:

       n < 5000   -> current COS53 evaluator
       n >= 5000  -> frozen custom permanent 2-core scheduler

   Evidence:
     GitHub Actions run 33562041646, exact Xeon shard 41.
     custom2 was bit-identical to current COS53 over all 72 requested cells.
     At every tested n >= 5000 point, custom2 beat current COS53 in all six
     sign/range cells. The exact crossover below 5000 was not measured and is
     intentionally not guessed here.

   This dispatcher changes scheduling only. The supplied evaluator remains the
   same current COS53 implementation.

   FROZEN: do not change the 5000 threshold or scheduler in this file without a
   new benchmark and an explicit production promotion.
*/

#include <cstddef>
#include "cosine53_custom_2core_5000_frozen.hpp"

class Cosine53BatchProductionFrozen {
public:
    using fn_t = void (*)(double *, const double *, size_t);
    static constexpr size_t kCustom2MinN = 5000;

    explicit Cosine53BatchProductionFrozen(fn_t current_eval)
        : current_eval_(current_eval), custom2_(current_eval) {}

    Cosine53BatchProductionFrozen(const Cosine53BatchProductionFrozen&) = delete;
    Cosine53BatchProductionFrozen& operator=(const Cosine53BatchProductionFrozen&) = delete;

    void run(double *out, const double *in, size_t n) {
        if (n >= kCustom2MinN) {
            custom2_.run(out, in, n);
        } else {
            current_eval_(out, in, n);
        }
    }

private:
    fn_t current_eval_;
    Cosine53CustomPermanent2Core5000Frozen custom2_;
};
