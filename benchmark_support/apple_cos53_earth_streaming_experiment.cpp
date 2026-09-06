// EARTH streaming experiment — derived from frozen EARTH, which remains untouched.
// Changes only hot-loop streaming schedule: ptrue full-vector loop + 4x unroll + one tail.

#include <arm_sve.h>
#include <cstddef>
#include <cstdint>

static constexpr double EARTHX_MH = -0x1.ffffff92c5f94p-2;
static constexpr double EARTHX_M6 = -0x1.5555551eb851fp-3;

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

#define EARTHX_VEC(PG, IDX) do { \
    const size_t _i = (IDX); \
    svfloat64_t de = svld1_f64((PG), delta + _i); \
    svfloat64_t a0 = svld1_f64((PG), c0 + _i); \
    svfloat64_t a1 = svld1_f64((PG), c1 + _i); \
    svfloat64_t c2 = svmul_n_f64_x((PG), a0, EARTHX_MH); \
    svfloat64_t c3 = svmul_n_f64_x((PG), a1, EARTHX_M6); \
    svfloat64_t p = svmla_f64_x((PG), c2, c3, de); \
    p = svmla_f64_x((PG), a1, p, de); \
    p = svmla_f64_x((PG), a0, p, de); \
    svuint64_t qb = svld1_u64((PG), qbits + _i); \
    svuint64_t parity = svand_u64_x((PG), qb, one); \
    svuint64_t outsign = svlsl_n_u64_x((PG), parity, 63); \
    p = svreinterpret_f64_u64(sveor_u64_x((PG), svreinterpret_u64_f64(p), outsign)); \
    svst1_f64((PG), y + _i, p); \
} while (0)

    // M4 Pro streaming vector length is 8 FP64 lanes; unroll four vectors.
    for (; i + 32 <= n; i += 32) {
        EARTHX_VEC(pg, i + 0);
        EARTHX_VEC(pg, i + 8);
        EARTHX_VEC(pg, i + 16);
        EARTHX_VEC(pg, i + 24);
    }
    for (; i + 8 <= n; i += 8) {
        EARTHX_VEC(pg, i);
    }
    if (i < n) {
        svbool_t tail = svwhilelt_b64((uint64_t)i, (uint64_t)n);
        EARTHX_VEC(tail, i);
    }

#undef EARTHX_VEC
}
