// EARTH streaming experiment — derived from frozen EARTH, which remains untouched.
// Changes only hot-loop streaming schedule: ptrue full-vector loop + 4x unroll + one tail.

#include <arm_sve.h>
#include <cstddef>
#include <cstdint>

static constexpr double EARTHX_MH = -0x1.ffffff92c5f94p-2;
static constexpr double EARTHX_M6 = -0x1.5555551eb851fp-3;

static inline void earthx_vec(
    svbool_t pg, double* y, size_t i,
    const double* delta, const double* c0, const double* c1,
    const uint64_t* qbits, svuint64_t one) {
    svfloat64_t de = svld1_f64(pg, delta + i);
    svfloat64_t a0 = svld1_f64(pg, c0 + i);
    svfloat64_t a1 = svld1_f64(pg, c1 + i);
    svfloat64_t c2 = svmul_n_f64_x(pg, a0, EARTHX_MH);
    svfloat64_t c3 = svmul_n_f64_x(pg, a1, EARTHX_M6);
    svfloat64_t p = svmla_f64_x(pg, c2, c3, de);
    p = svmla_f64_x(pg, a1, p, de);
    p = svmla_f64_x(pg, a0, p, de);
    svuint64_t qb = svld1_u64(pg, qbits + i);
    svuint64_t parity = svand_u64_x(pg, qb, one);
    svuint64_t outsign = svlsl_n_u64_x(pg, parity, 63);
    p = svreinterpret_f64_u64(sveor_u64_x(pg, svreinterpret_u64_f64(p), outsign));
    svst1_f64(pg, y + i, p);
}

extern "C" void earthx_stream(
    const double* x, double* y, size_t n,
    const double* delta, const double* c0, const double* c1,
    const uint64_t* qbits) __arm_streaming;

extern "C" void earthx_stream(
    const double* x, double* y, size_t n,
    const double* delta, const double* c0, const double* c1,
    const uint64_t* qbits) __arm_streaming {
    (void)x;
    const svuint64_t one = svdup_u64(1);
    const svbool_t pg = svptrue_b64();
    size_t i = 0;

    // M4 Pro streaming vector length is 8 FP64 lanes; unroll four vectors.
    for (; i + 32 <= n; i += 32) {
        earthx_vec(pg, y, i + 0,  delta, c0, c1, qbits, one);
        earthx_vec(pg, y, i + 8,  delta, c0, c1, qbits, one);
        earthx_vec(pg, y, i + 16, delta, c0, c1, qbits, one);
        earthx_vec(pg, y, i + 24, delta, c0, c1, qbits, one);
    }
    for (; i + 8 <= n; i += 8)
        earthx_vec(pg, y, i, delta, c0, c1, qbits, one);
    if (i < n) {
        svbool_t tail = svwhilelt_b64((uint64_t)i, (uint64_t)n);
        earthx_vec(tail, y, i, delta, c0, c1, qbits, one);
    }
}
