#define _GNU_SOURCE
#include <mkl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef COSINE53_GENERATED_SOURCE
#error "compile with -DCOSINE53_GENERATED_SOURCE=\"...generated source...\""
#endif

/* The generated production source has its original benchmark main renamed by
   the workflow before this wrapper is compiled. Its static production helpers
   therefore remain available in this translation unit without changing them. */
#include COSINE53_GENERATED_SOURCE

#ifndef SHALLOW_ENGINE_WIDE
#define SHALLOW_ENGINE_WIDE 0
#endif

enum { SHALLOW_TRIALS = 5 };

static uint64_t shallow_mix64(uint64_t x)
{
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

static double shallow_u01(uint64_t x)
{
    return ((double)(shallow_mix64(x) >> 11) + 0.5) * 0x1p-53;
}

static double shallow_now_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (double)t.tv_sec * 1.0e9 + (double)t.tv_nsec;
}

static int shallow_cmp_d(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static void *shallow_al64(size_t bytes)
{
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0) return NULL;
    return p;
}

/* case ids: 0 unit+, 1 unit-, 2 mid+, 3 mid-, 4 far+, 5 far- */
static const char *shallow_case_name(int c)
{
    static const char *n[6] = {
        "unit_pos", "unit_neg", "mid_pos", "mid_neg", "far_pos", "far_neg"
    };
    return n[c];
}

static void shallow_fill(double *x, size_t n, int c)
{
    const double edge = 0x1p-20;
    double lo, hi;
    if (c < 2) { lo = edge; hi = 1.0 - edge; }
    else if (c < 4) { lo = 1.0 + edge; hi = 500.0 - edge; }
    else { lo = 1000.0 + edge; hi = 10000.0 - edge; }

    uint64_t seed = UINT64_C(0xd1b54a32d192ed03)
                  ^ ((uint64_t)n * UINT64_C(0x94d049bb133111eb))
                  ^ ((uint64_t)c << 58);
    for (size_t i = 0; i < n; ++i) {
        double u = shallow_u01(seed + (uint64_t)i * UINT64_C(0x9e3779b97f4a7c15));
        double v = lo + (hi - lo) * u;
        x[i] = (c & 1) ? -v : v;
    }
}

static size_t shallow_reps(size_t n)
{
    /* Shallow timing only: same six arrays are reused; repetitions reduce clock noise. */
    const size_t target_values = 2000000;
    size_t r = target_values / n;
    if (r < 1) r = 1;
    if (r > 50000) r = 50000;
    return r;
}

static double shallow_run_ours(const s53w_kernel *k, const double *x, double *y,
                               size_t n, size_t reps, volatile double *sink)
{
    double t0 = shallow_now_ns();
    for (size_t r = 0; r < reps; ++r) octant_eval_v8(k, x, y, n);
    double t1 = shallow_now_ns();
    *sink += y[(n * 7u / 11u) % n];
    return (t1 - t0) / ((double)reps * (double)n);
}

static double shallow_run_intel(const double *x, double *y, size_t n, size_t reps,
                                volatile double *sink)
{
    double t0 = shallow_now_ns();
    for (size_t r = 0; r < reps; ++r) vmdCos((MKL_INT)n, x, y, VML_HA);
    double t1 = shallow_now_ns();
    *sink += y[(n * 5u / 13u) % n];
    return (t1 - t0) / ((double)reps * (double)n);
}

int main(void)
{
    static const size_t sizes[] = {
        50, 250, 1200, 5000, 10000, 30000, 50000, 100000,
        500000, 1000000, 2000000, 4000000
    };

    int cpu = pin();
    mkl_set_num_threads_local(1);
    if (!redtab2_init()) return 2;
    s53w_kernel *k = kernel_create(2);
    if (!k) { redtab2_clear(); return 3; }

    const int c0 = SHALLOW_ENGINE_WIDE ? 2 : 0;
    const int c1 = SHALLOW_ENGINE_WIDE ? 6 : 2;
    printf("COS53_SHALLOW engine=%s cpu_pin=%d sizes=12 trials=%d random_batches_per_size=%d intel=oneMKL_vmdCos_VML_HA\n",
           SHALLOW_ENGINE_WIDE ? "X67_wide" : "X50_unit", cpu, SHALLOW_TRIALS, c1-c0);

    volatile double sink = 0.0;
    for (size_t si = 0; si < sizeof(sizes)/sizeof(sizes[0]); ++si) {
        size_t n = sizes[si];
        for (int c = c0; c < c1; ++c) {
            double *x = (double *)shallow_al64(n * sizeof(double));
            double *yo = (double *)shallow_al64(n * sizeof(double));
            double *yi = (double *)shallow_al64(n * sizeof(double));
            if (!x || !yo || !yi) {
                free(yi); free(yo); free(x);
                kernel_destroy(k); redtab2_clear();
                return 4;
            }
            shallow_fill(x, n, c);
            size_t reps = shallow_reps(n);

            /* Warm the exact same arrays used for timing. */
            for (int w = 0; w < 3; ++w) {
                octant_eval_v8(k, x, yo, n);
                vmdCos((MKL_INT)n, x, yi, VML_HA);
            }

            double ot[SHALLOW_TRIALS], it[SHALLOW_TRIALS];
            for (int t = 0; t < SHALLOW_TRIALS; ++t) {
                if (t & 1) {
                    it[t] = shallow_run_intel(x, yi, n, reps, &sink);
                    ot[t] = shallow_run_ours(k, x, yo, n, reps, &sink);
                } else {
                    ot[t] = shallow_run_ours(k, x, yo, n, reps, &sink);
                    it[t] = shallow_run_intel(x, yi, n, reps, &sink);
                }
            }
            qsort(ot, SHALLOW_TRIALS, sizeof(double), shallow_cmp_d);
            qsort(it, SHALLOW_TRIALS, sizeof(double), shallow_cmp_d);
            double om = ot[SHALLOW_TRIALS/2], im = it[SHALLOW_TRIALS/2];
            printf("RESULT n=%zu case=%s reps=%zu ours_ns=%.9f intel_ns=%.9f intel_over_ours=%.6f winner=%s\n",
                   n, shallow_case_name(c), reps, om, im, im/om,
                   om < im ? "OURS" : "INTEL");

            free(yi); free(yo); free(x);
        }
    }
    printf("COS53_SHALLOW_DONE engine=%s sink=%.17g\n",
           SHALLOW_ENGINE_WIDE ? "X67_wide" : "X50_unit", (double)sink);
    kernel_destroy(k);
    redtab2_clear();
    flint_cleanup_master();
    return 0;
}
