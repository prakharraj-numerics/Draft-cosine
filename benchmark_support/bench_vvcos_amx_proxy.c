#include <Accelerate/Accelerate.h>
#include <mach/mach_time.h>
#include <sys/resource.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double wall_seconds(void) {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) mach_timebase_info(&tb);
    uint64_t t = mach_absolute_time();
    return (double)t * (double)tb.numer / (double)tb.denom * 1e-9;
}

static double cpu_seconds(void) {
    struct rusage r;
    getrusage(RUSAGE_SELF, &r);
    return (double)r.ru_utime.tv_sec + 1e-6 * (double)r.ru_utime.tv_usec +
           (double)r.ru_stime.tv_sec + 1e-6 * (double)r.ru_stime.tv_usec;
}

static uint64_t step(uint64_t *s) {
    uint64_t x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * UINT64_C(2685821657736338717);
}

static void fill_inputs(double *x, int n) {
    uint64_t s = UINT64_C(0x6a09e667f3bcc909);
    for (int i = 0; i < n; ++i) {
        uint64_t r = step(&s);
        double u = (double)(r >> 11) * 0x1.0p-53;
        double mag;
        switch (i % 3) {
            case 0: mag = 0.000001 + u * 0.999999; break;
            case 1: mag = 1.0 + u * 499.0; break;
            default: mag = 1000.0 + u * 9000.0; break;
        }
        x[i] = (r & 1) ? mag : -mag;
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s N\n", argv[0]);
        return 2;
    }
    long nl = strtol(argv[1], NULL, 10);
    if (nl <= 0 || nl > INT32_MAX) return 2;
    int n = (int)nl;

    double *x = NULL, *y = NULL;
    if (posix_memalign((void **)&x, 64, (size_t)n * sizeof(double)) ||
        posix_memalign((void **)&y, 64, (size_t)n * sizeof(double))) {
        return 3;
    }
    fill_inputs(x, n);

    int nn = n;
    for (int i = 0; i < 8; ++i) vvcos(y, x, &nn);

    long long target = 2000000LL;
    int iters = (int)((target + n - 1) / n);
    if (iters < 3) iters = 3;
    if (iters > 20000) iters = 20000;

    double c0 = cpu_seconds();
    double w0 = wall_seconds();
    for (int r = 0; r < iters; ++r) vvcos(y, x, &nn);
    double w1 = wall_seconds();
    double c1 = cpu_seconds();

    double elems = (double)n * (double)iters;
    double wall_ns = (w1 - w0) * 1e9 / elems;
    double cpu_ns = (c1 - c0) * 1e9 / elems;
    double cores = (w1 > w0) ? (c1 - c0) / (w1 - w0) : 0.0;
    volatile double checksum = y[0] + y[n / 2] + y[n - 1];

#if defined(__x86_64__)
    const char *arch = "x86_64";
#elif defined(__aarch64__)
    const char *arch = "arm64";
#else
    const char *arch = "unknown";
#endif

    printf("arch=%s n=%d iters=%d wall_ns_per_el=%.9f cpu_ns_per_el=%.9f effective_cores=%.6f checksum=%.17g\n",
           arch, n, iters, wall_ns, cpu_ns, cores, (double)checksum);

    free(x);
    free(y);
    return 0;
}
