// EARTH VL-agnostic experiment — derived from frozen EARTH; frozen source is untouched.
// Only change: advance by the current SME streaming FP64 vector length via svcntd().

#include <arm_sve.h>
#include <cstddef>
#include <cstdint>

static constexpr double EARTH_VL_MH = -0x1.ffffff92c5f94p-2;
static constexpr double EARTH_VL_M6 = -0x1.5555551eb851fp-3;

extern "C" void earth_vl_stream(
    const double* x,
    double* y,
    size_t n,
    const double* delta,
    const double* c0,
    const double* c1,
    const uint64_t* qbits) __arm_streaming;

extern "C" void earth_vl_stream(
    const double* x,
    double* y,
    size_t n,
    const double* delta,
    const double* c0,
    const double* c1,
    const uint64_t* qbits) __arm_streaming {
    (void)x;
    const svuint64_t one = svdup_u64(1);
    const size_t vl = svcntd();

    for (size_t i = 0; i < n; i += vl) {
        svbool_t pg = svwhilelt_b64((uint64_t)i, (uint64_t)n);

        svfloat64_t de = svld1_f64(pg, delta + i);
        svfloat64_t a0 = svld1_f64(pg, c0 + i);
        svfloat64_t a1 = svld1_f64(pg, c1 + i);

        svfloat64_t c2 = svmul_n_f64_x(pg, a0, EARTH_VL_MH);
        svfloat64_t c3 = svmul_n_f64_x(pg, a1, EARTH_VL_M6);

        svfloat64_t p = svmla_f64_x(pg, c2, c3, de);
        p = svmla_f64_x(pg, a1, p, de);
        p = svmla_f64_x(pg, a0, p, de);

        svuint64_t qb = svld1_u64(pg, qbits + i);
        svuint64_t parity = svand_u64_x(pg, qb, one);
        svuint64_t outsign = svlsl_n_u64_x(pg, parity, 63);
        p = svreinterpret_f64_u64(
            sveor_u64_x(pg, svreinterpret_u64_f64(p), outsign));

        svst1_f64(pg, y + i, p);
    }
}
