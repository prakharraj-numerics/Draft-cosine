#define _GNU_SOURCE
#include <stddef.h>

#ifndef COSINE53_GENERATED_SOURCE
#error "compile with -DCOSINE53_GENERATED_SOURCE=\"generated source\""
#endif

#include COSINE53_GENERATED_SOURCE

static s53w_kernel *cos53_adapter_kernel = NULL;

int cos53_engine_init(void)
{
    if (!redtab2_init()) return 0;
    cos53_adapter_kernel = kernel_create(2);
    if (!cos53_adapter_kernel) {
        redtab2_clear();
        return 0;
    }
    return 1;
}

void cos53_engine_eval(double *out, const double *in, size_t n)
{
    octant_eval_v8(cos53_adapter_kernel, in, out, n);
}

void cos53_engine_cleanup(void)
{
    if (cos53_adapter_kernel) {
        kernel_destroy(cos53_adapter_kernel);
        cos53_adapter_kernel = NULL;
    }
    redtab2_clear();
    flint_cleanup_master();
}
