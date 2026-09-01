#pragma once

/* COS53 FROZEN production batch routing.

   Routing rule frozen from exact Intel Xeon 6973P-C benchmark evidence:

       n < 2000   -> current COS53 evaluator
       n >= 2000  -> frozen custom permanent 2-core scheduler

   Evidence:
     Broad benchmark: GitHub Actions run 33562041646, exact Xeon shard 41.
     Boundary benchmark: GitHub Actions run 33563237026, exact Xeon shard 5.

     custom2 was bit-identical to current COS53 over the tested maps.
     At every tested point from n=2000 through n=4500, custom2 beat current
     COS53 in all six sign/range cells. At every tested point from n=5000
     through n=4000000, custom2 likewise beat current in all six cells.
     At n=1500, current COS53 remained slightly faster on the all-six average.

   This dispatcher changes scheduling only. The supplied evaluator remains the
   same current COS53 implementation.

   FROZEN: do not change the 2000 threshold or scheduler in this file without a
   new benchmark and an explicit production promotion.
*/

#include <cstddef>
#include "cosine53_custom_2core_2000_frozen.hpp"

class Cosine53BatchProductionFrozen {
public:
    using fn_t = void (*)(double *, const double *, size_t);
    static constexpr size_t kCustom2MinN = 2000;

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
    Cosine53CustomPermanent2Core2000Frozen custom2_;
};
