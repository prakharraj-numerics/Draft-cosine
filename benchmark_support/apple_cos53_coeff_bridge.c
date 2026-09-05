#include <flint/arf.h>
#include <flint/flint.h>
#include <flint/fmpz.h>
#include <stddef.h>
#include <stdint.h>

/* Generated in the workflow from the production coefficient source by applying
   the existing cosine conversion plus the production SF_K=8/LUT_N=403 patch. */
#include "apple_cosine53_coeff_source.c"

static double fixed103_to_double(const mp_limb_t q[2], int neg)
{
    fmpz_t z;
    arf_t a;
    fmpz_init(z);
    arf_init(a);
    fmpz_set_ui_array(z, q, 2);
    arf_set_fmpz(a, z);
    arf_mul_2exp_si(a, a, -103);
    double d = arf_get_d(a, ARF_RND_NEAR);
    arf_clear(a);
    fmpz_clear(z);
    return neg ? -d : d;
}

int cos53_apple_build_c0_c1(double *c0, double *c1, size_t cap)
{
    sine_fixed_ctx *c = s53_coeff_create_terms(2);
    if (!c) return 0;
    if (c->poly_deg != 5 || cap < (size_t)SF_LUT_N) {
        s53_coeff_destroy(c);
        return 0;
    }
    for (size_t a = 0; a < (size_t)SF_LUT_N; ++a) {
        size_t off = a * (size_t)(c->poly_deg + 1);
        c0[a] = fixed103_to_double(c->coef + 2 * (off + 0), c->coef_sign[off + 0] != 0);
        c1[a] = fixed103_to_double(c->coef + 2 * (off + 1), c->coef_sign[off + 1] != 0);
    }
    s53_coeff_destroy(c);
    return (int)SF_LUT_N;
}

void cos53_apple_coeff_cleanup(void)
{
    flint_cleanup_master();
}
