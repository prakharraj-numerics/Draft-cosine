#define _GNU_SOURCE
#include <mkl.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <time.h>
#include <sys/resource.h>
#include <pthread.h>
#include <sched.h>
#include <vector>

#include "cosine53_batch_production.hpp"

extern "C" int cos53_engine_init(void);
extern "C" void cos53_engine_eval(double *, const double *, size_t);
extern "C" void cos53_engine_cleanup(void);

#ifndef COS53_ENGINE_WIDE
#define COS53_ENGINE_WIDE 0
#endif

static volatile double g_sink = 0.0;

extern "C" __attribute__((noinline)) void cos53_profile_start(void) {
    asm volatile("" ::: "memory");
}
extern "C" __attribute__((noinline)) void cos53_profile_stop(void) {
    asm volatile("" ::: "memory");
}

static uint64_t mix64(uint64_t x) {
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

static double u01(uint64_t x) {
    return ((double)(mix64(x) >> 11) + 0.5) * 0x1p-53;
}

static void pin_current(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

static void *al64(size_t bytes) {
    void *p = nullptr;
    if (posix_memalign(&p, 64, bytes) != 0) return nullptr;
    return p;
}

static double clock_ns(clockid_t id) {
    struct timespec t;
    clock_gettime(id, &t);
    return (double)t.tv_sec * 1.0e9 + (double)t.tv_nsec;
}

static const char *case_name(int c) {
    static const char *names[6] = {
        "unit_pos", "unit_neg", "mid_pos", "mid_neg", "far_pos", "far_neg"
    };
    return names[c];
}

static void fill(double *x, size_t n, int c) {
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

static uint64_t ordered_bits(double x) {
    uint64_t u;
    std::memcpy(&u, &x, sizeof(u));
    if (u >> 63) return ~u;
    return u | UINT64_C(0x8000000000000000);
}

static uint64_t ulp_distance(double a, double b) {
    if (std::isnan(a) || std::isnan(b)) return UINT64_MAX;
    uint64_t x = ordered_bits(a), y = ordered_bits(b);
    return x > y ? x - y : y - x;
}

struct Buffers {
    std::vector<double*> x;
    std::vector<double*> y;
    std::vector<double*> ref;
    size_t n = 0;
    int c0 = 0, c1 = 0;
    ~Buffers() {
        for (double *p : x) std::free(p);
        for (double *p : y) std::free(p);
        for (double *p : ref) std::free(p);
    }
};

static bool alloc_buffers(Buffers &b, size_t n) {
    b.n = n;
    b.c0 = COS53_ENGINE_WIDE ? 2 : 0;
    b.c1 = COS53_ENGINE_WIDE ? 6 : 2;
    const int nc = b.c1 - b.c0;
    b.x.resize(nc); b.y.resize(nc); b.ref.resize(nc);
    for (int j = 0; j < nc; ++j) {
        b.x[j] = (double*)al64(n * sizeof(double));
        b.y[j] = (double*)al64(n * sizeof(double));
        b.ref[j] = (double*)al64(n * sizeof(double));
        if (!b.x[j] || !b.y[j] || !b.ref[j]) return false;
        fill(b.x[j], n, b.c0 + j);
    }
    return true;
}

static void run_prod_once(Cosine53BatchProductionFrozen &prod, Buffers &b) {
    for (size_t j = 0; j < b.x.size(); ++j) prod.run(b.y[j], b.x[j], b.n);
}

static void run_intel_once(Buffers &b) {
    for (size_t j = 0; j < b.x.size(); ++j)
        vmdCos((MKL_INT)b.n, b.x[j], b.y[j], VML_HA);
}

static uint64_t verify_vs_intel(Cosine53BatchProductionFrozen &prod, Buffers &b) {
    run_prod_once(prod, b);
    for (size_t j = 0; j < b.x.size(); ++j)
        vmdCos((MKL_INT)b.n, b.x[j], b.ref[j], VML_HA);
    uint64_t mx = 0;
    for (size_t j = 0; j < b.x.size(); ++j)
        for (size_t i = 0; i < b.n; ++i)
            mx = std::max(mx, ulp_distance(b.y[j][i], b.ref[j][i]));
    return mx;
}

static size_t reps_for(size_t n, size_t cases) {
    const size_t target_values = 4000000;
    size_t r = target_values / (n * cases);
    if (r < 1) r = 1;
    if (r > 20000) r = 20000;
    return r;
}

static double median5(double v[5]) {
    std::sort(v, v + 5);
    return v[2];
}

static int native_mode(const std::string &which) {
    static const size_t sizes[] = {100,700,3500,15000,50000,1000000,2000000};
    Cosine53BatchProductionFrozen *prod = nullptr;
    if (which == "prod") prod = new Cosine53BatchProductionFrozen(cos53_engine_eval);
    for (size_t n : sizes) {
        Buffers b;
        if (!alloc_buffers(b, n)) { delete prod; return 10; }
        const size_t cases = b.x.size();
        const size_t reps = reps_for(n, cases);
        uint64_t maxulp = 0;
        if (which == "prod") maxulp = verify_vs_intel(*prod, b);
        for (int w = 0; w < 3; ++w) {
            if (which == "prod") run_prod_once(*prod, b); else run_intel_once(b);
        }
        double wall[5], cpu[5];
        for (int t = 0; t < 5; ++t) {
            double w0 = clock_ns(CLOCK_MONOTONIC_RAW);
            double c0 = clock_ns(CLOCK_PROCESS_CPUTIME_ID);
            for (size_t r = 0; r < reps; ++r) {
                if (which == "prod") run_prod_once(*prod, b); else run_intel_once(b);
            }
            double c1 = clock_ns(CLOCK_PROCESS_CPUTIME_ID);
            double w1 = clock_ns(CLOCK_MONOTONIC_RAW);
            const double denom = (double)reps * (double)n * (double)cases;
            wall[t] = (w1 - w0) / denom;
            cpu[t] = (c1 - c0) / denom;
            g_sink += b.y[(size_t)t % cases][(n * 7u / 11u) % n];
        }
        struct rusage ru{};
        getrusage(RUSAGE_SELF, &ru);
        const double mw = median5(wall), mc = median5(cpu);
        std::printf("NATIVE engine=%s stack=%s n=%zu cases=%zu reps=%zu wall_ns_el=%.9f cpu_ns_el=%.9f effective_cores=%.6f maxrss_kib=%ld maxulp_vs_intel=%llu\n",
                    COS53_ENGINE_WIDE ? "wide" : "unit", which.c_str(), n, cases, reps,
                    mw, mc, mw > 0.0 ? mc/mw : 0.0, ru.ru_maxrss,
                    (unsigned long long)maxulp);
    }
    delete prod;
    std::printf("NATIVE_DONE engine=%s stack=%s sink=%.17g\n",
                COS53_ENGINE_WIDE ? "wide" : "unit", which.c_str(), (double)g_sink);
    return 0;
}

static int sde_mode(const std::string &which, size_t n) {
    Buffers b;
    if (!alloc_buffers(b, n)) return 20;
    Cosine53BatchProductionFrozen *prod = nullptr;
    if (which == "prod") prod = new Cosine53BatchProductionFrozen(cos53_engine_eval);
    if (which == "prod") run_prod_once(*prod, b);
    else if (which == "intel") run_intel_once(b);
    cos53_profile_start();
    if (which == "prod") run_prod_once(*prod, b);
    else if (which == "intel") run_intel_once(b);
    else {
        for (size_t j = 0; j < b.x.size(); ++j) {
            asm volatile("" : : "r"(b.x[j]), "r"(b.y[j]), "r"(b.n) : "memory");
        }
    }
    cos53_profile_stop();
    if (which != "noop") g_sink += b.y[0][(n * 5u / 13u) % n];
    std::printf("SDE_PROGRAM engine=%s stack=%s n=%zu cases=%zu sink=%.17g\n",
                COS53_ENGINE_WIDE ? "wide" : "unit", which.c_str(), n, b.x.size(), (double)g_sink);
    delete prod;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    pin_current(0);
    mkl_set_num_threads_local(1);
    if (!cos53_engine_init()) return 3;
    std::string mode = argv[1];
    int rc = 2;
    if (mode == "native-prod") rc = native_mode("prod");
    else if (mode == "native-intel") rc = native_mode("intel");
    else if ((mode == "sde-prod" || mode == "sde-intel" || mode == "sde-noop") && argc == 3) {
        size_t n = (size_t)std::strtoull(argv[2], nullptr, 10);
        rc = sde_mode(mode.substr(4), n);
    }
    cos53_engine_cleanup();
    return rc;
}
