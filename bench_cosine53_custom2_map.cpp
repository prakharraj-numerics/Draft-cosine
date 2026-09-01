#define _GNU_SOURCE
#include <mkl.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <time.h>
#include <pthread.h>
#include <sched.h>

#include "cosine53_custom_2core.hpp"

extern "C" int cos53_engine_init(void);
extern "C" void cos53_engine_eval(double *, const double *, size_t);
extern "C" void cos53_engine_cleanup(void);

#ifndef CUSTOM2_ENGINE_WIDE
#define CUSTOM2_ENGINE_WIDE 0
#endif

enum { INNER_TRIALS = 5 };

static uint64_t mix64(uint64_t x)
{
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

static double u01(uint64_t x)
{
    return ((double)(mix64(x) >> 11) + 0.5) * 0x1p-53;
}

static double now_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (double)t.tv_sec * 1.0e9 + (double)t.tv_nsec;
}

static void pin_current(int cpu)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

static void *al64(size_t bytes)
{
    void *p = nullptr;
    if (posix_memalign(&p, 64, bytes) != 0) return nullptr;
    return p;
}

static const char *case_name(int c)
{
    static const char *names[6] = {
        "unit_pos", "unit_neg", "mid_pos", "mid_neg", "far_pos", "far_neg"
    };
    return names[c];
}

static void fill(double *x, size_t n, int c)
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
        double q = u01(seed + (uint64_t)i * UINT64_C(0x9e3779b97f4a7c15));
        double v = lo + (hi - lo) * q;
        x[i] = (c & 1) ? -v : v;
    }
}

static size_t reps_for(size_t n)
{
    const size_t target_values = 2000000;
    size_t r = target_values / n;
    if (r < 1) r = 1;
    if (r > 50000) r = 50000;
    return r;
}

static double median5(double a[INNER_TRIALS])
{
    std::sort(a, a + INNER_TRIALS);
    return a[INNER_TRIALS / 2];
}

static double run_current(const double *x, double *y, size_t n, size_t reps,
                          volatile double *sink)
{
    const double t0 = now_ns();
    for (size_t r = 0; r < reps; ++r) cos53_engine_eval(y, x, n);
    const double t1 = now_ns();
    *sink += y[(n * 7u / 11u) % n];
    return (t1 - t0) / ((double)reps * (double)n);
}

static double run_custom(Cosine53CustomPermanent2Core &custom,
                         const double *x, double *y, size_t n, size_t reps,
                         volatile double *sink)
{
    const double t0 = now_ns();
    for (size_t r = 0; r < reps; ++r) custom.run(y, x, n);
    const double t1 = now_ns();
    *sink += y[(n * 7u / 11u) % n];
    return (t1 - t0) / ((double)reps * (double)n);
}

static double run_intel(const double *x, double *y, size_t n, size_t reps,
                        volatile double *sink)
{
    const double t0 = now_ns();
    for (size_t r = 0; r < reps; ++r) vmdCos((MKL_INT)n, x, y, VML_HA);
    const double t1 = now_ns();
    *sink += y[(n * 5u / 13u) % n];
    return (t1 - t0) / ((double)reps * (double)n);
}

static int verify_sync(void)
{
    static const size_t sizes[] = {
        50,250,1200,5000,10000,30000,50000,100000,
        500000,1000000,2000000,4000000
    };
    const int c0 = CUSTOM2_ENGINE_WIDE ? 2 : 0;
    const int c1 = CUSTOM2_ENGINE_WIDE ? 6 : 2;
    Cosine53CustomPermanent2Core custom(cos53_engine_eval);
    size_t total_diff = 0;

    for (size_t n : sizes) {
        for (int c = c0; c < c1; ++c) {
            double *x = (double *)al64(n * sizeof(double));
            double *a = (double *)al64(n * sizeof(double));
            double *b = (double *)al64(n * sizeof(double));
            if (!x || !a || !b) return 20;
            fill(x, n, c);
            cos53_engine_eval(a, x, n);
            custom.run(b, x, n);
            size_t diff = 0;
            for (size_t i = 0; i < n; ++i)
                diff += std::memcmp(a + i, b + i, sizeof(double)) != 0;
            total_diff += diff;
            std::printf("VERIFY n=%zu case=%s bitdiff=%zu\n", n, case_name(c), diff);
            std::free(b); std::free(a); std::free(x);
        }
    }
    std::printf("VERIFY_DONE engine=%s total_bitdiff=%zu\n",
                CUSTOM2_ENGINE_WIDE ? "X67_wide" : "X50_unit", total_diff);
    return total_diff ? 21 : 0;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s verify|current|custom2|intel\n", argv[0]);
        return 2;
    }
    const std::string mode = argv[1];
    if (mode != "verify" && mode != "current" && mode != "custom2" && mode != "intel") return 2;

    pin_current(0);
    mkl_set_num_threads_local(1);
    if (!cos53_engine_init()) return 3;

    if (mode == "verify") {
        const int rc = verify_sync();
        cos53_engine_cleanup();
        return rc;
    }

    static const size_t sizes[] = {
        50,250,1200,5000,10000,30000,50000,100000,
        500000,1000000,2000000,4000000
    };
    const int c0 = CUSTOM2_ENGINE_WIDE ? 2 : 0;
    const int c1 = CUSTOM2_ENGINE_WIDE ? 6 : 2;
    const char *outer = std::getenv("COS53_OUTER_REP");
    if (!outer) outer = "0";

    std::printf("COS53_CUSTOM2 engine=%s mode=%s outer=%s trials=%d\n",
                CUSTOM2_ENGINE_WIDE ? "X67_wide" : "X50_unit",
                mode.c_str(), outer, INNER_TRIALS);

    volatile double sink = 0.0;
    Cosine53CustomPermanent2Core *custom = nullptr;
    if (mode == "custom2") custom = new Cosine53CustomPermanent2Core(cos53_engine_eval);

    for (size_t n : sizes) {
        for (int c = c0; c < c1; ++c) {
            double *x = (double *)al64(n * sizeof(double));
            double *y = (double *)al64(n * sizeof(double));
            if (!x || !y) return 4;
            fill(x, n, c);
            const size_t reps = reps_for(n);

            for (int w = 0; w < 3; ++w) {
                if (mode == "current") cos53_engine_eval(y, x, n);
                else if (mode == "custom2") custom->run(y, x, n);
                else vmdCos((MKL_INT)n, x, y, VML_HA);
            }

            double samples[INNER_TRIALS];
            for (int t = 0; t < INNER_TRIALS; ++t) {
                if (mode == "current") samples[t] = run_current(x, y, n, reps, &sink);
                else if (mode == "custom2") samples[t] = run_custom(*custom, x, y, n, reps, &sink);
                else samples[t] = run_intel(x, y, n, reps, &sink);
            }
            const double med = median5(samples);
            std::printf("RESULT outer=%s mode=%s n=%zu case=%s reps=%zu ns=%.9f\n",
                        outer, mode.c_str(), n, case_name(c), reps, med);
            std::free(y); std::free(x);
        }
    }

    delete custom;
    std::printf("COS53_CUSTOM2_DONE engine=%s mode=%s outer=%s sink=%.17g\n",
                CUSTOM2_ENGINE_WIDE ? "X67_wide" : "X50_unit",
                mode.c_str(), outer, (double)sink);
    cos53_engine_cleanup();
    return 0;
}
