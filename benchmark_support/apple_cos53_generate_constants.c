#include <mpfr.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

int cos53_apple_build_c0_c1(double *c0, double *c1, size_t cap);
void cos53_apple_coeff_cleanup(void);

#define LUTN 403
#define REDN 3185

int main(int argc, char **argv)
{
    if (argc != 2) return 2;
    FILE *f = fopen(argv[1], "w");
    if (!f) return 3;

    double *c0 = (double *)malloc(LUTN * sizeof(double));
    double *c1 = (double *)malloc(LUTN * sizeof(double));
    if (!c0 || !c1) return 4;
    if (cos53_apple_build_c0_c1(c0, c1, LUTN) != LUTN) return 5;

    fprintf(f, "#pragma once\n#include <stddef.h>\n");
    fprintf(f, "#define APPLE_COS53_LUTN %d\n#define APPLE_COS53_REDN %d\n", LUTN, REDN);
    fprintf(f, "static const double apple_cos53_c0[APPLE_COS53_LUTN] = {\n");
    for (int i = 0; i < LUTN; ++i)
        fprintf(f, "  %a,%s", c0[i], (i % 4 == 3) ? "\n" : " ");
    fprintf(f, "};\nstatic const double apple_cos53_c1[APPLE_COS53_LUTN] = {\n");
    for (int i = 0; i < LUTN; ++i)
        fprintf(f, "  %a,%s", c1[i], (i % 4 == 3) ? "\n" : " ");
    fprintf(f, "};\n");

    mpfr_t pi, t;
    mpfr_init2(pi, 256);
    mpfr_init2(t, 256);
    mpfr_const_pi(pi, MPFR_RNDN);
    fprintf(f, "static const double apple_cos53_pih[APPLE_COS53_REDN] = {\n");
    for (int q = 0; q < REDN; ++q) {
        mpfr_mul_ui(t, pi, (unsigned long)q, MPFR_RNDN);
        double h = mpfr_get_d(t, MPFR_RNDN);
        fprintf(f, "  %a,%s", h, (q % 4 == 3) ? "\n" : " ");
    }
    fprintf(f, "};\nstatic const double apple_cos53_pil[APPLE_COS53_REDN] = {\n");
    for (int q = 0; q < REDN; ++q) {
        mpfr_mul_ui(t, pi, (unsigned long)q, MPFR_RNDN);
        double h = mpfr_get_d(t, MPFR_RNDN);
        mpfr_sub_d(t, t, h, MPFR_RNDN);
        double l = mpfr_get_d(t, MPFR_RNDN);
        fprintf(f, "  %a,%s", l, (q % 4 == 3) ? "\n" : " ");
    }
    fprintf(f, "};\n");
    mpfr_clear(t);
    mpfr_clear(pi);

    fclose(f);
    cos53_apple_coeff_cleanup();
    free(c1);
    free(c0);
    return 0;
}
