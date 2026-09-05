#include <flint/arith.h>
#include <flint/arf.h>
#include <flint/flint.h>
#include <flint/fmpz.h>
#include <flint/gr.h>
#include <flint/gr_vec.h>
#include <flint/nfloat.h>
#include <stdint.h>
#include <stddef.h>

/* Exact coefficient-generation spine used by the 53-bit experiment.
   This is the same Mode-5 secant-spine builder used by the research kernel;
   only unrelated fixed-point runtime code is omitted from this public harness. */

#define SF_K 12
#define SF_LUT_N ((1UL << SF_K) + 1UL)
#define SF_MAX_TERMS 24
#define SF_MAX_DEG 8
#define SF_GUARD_BITS 96
#define S53_SOURCE_TARGET_BITS 57
#define S53_SOURCE_B 103

typedef struct
{
    slong target_bits;
    slong B;
    slong n;
    int terms;
    int spine_deg;
    int poly_deg;
    mp_limb_t *coef;
    unsigned char *coef_sign;
    fmpz_t conv_z;
    arf_t conv_a;
} sine_fixed_ctx;

typedef struct
{
    gr_ctx_t nctx;
    gr_vec_t residual;
    gr_vec_t pcoef;
    gr_vec_t qcoef;
    gr_vec_t ctab;
    gr_vec_t stab;
    gr_vec_t scratch;
    fmpz_t e, fact;
} sine_builder;

enum
{
    BS_ONE = 0, BS_TWO, BS_PI, BS_PI2,
    BS_A0, BS_A1, BS_Q0, BS_Q1, BS_QP0, BS_QP1,
    BS_G, BS_ACC, BS_T0, BS_T1,
    BS_DELTA, BS_Z, BS_R, BS_GF, BS_Q, BS_P, BS_DEN,
    BS_LO, BS_HI, BS_SEEDC, BS_SEEDS, BS_LUTC, BS_LUTS,
    BS_COUNT
};

static inline nfloat_ptr bv(sine_builder *b, slong i)
{ return (nfloat_ptr) gr_vec_entry_ptr(b->scratch, i, b->nctx); }
static inline nfloat_ptr br(sine_builder *b, slong i)
{ return (nfloat_ptr) gr_vec_entry_ptr(b->residual, i, b->nctx); }
static inline nfloat_ptr bp(sine_builder *b, slong i)
{ return (nfloat_ptr) gr_vec_entry_ptr(b->pcoef, i, b->nctx); }
static inline nfloat_ptr bq(sine_builder *b, slong i)
{ return (nfloat_ptr) gr_vec_entry_ptr(b->qcoef, i, b->nctx); }
static inline nfloat_ptr bc(sine_builder *b, slong i)
{ return (nfloat_ptr) gr_vec_entry_ptr(b->ctab, i, b->nctx); }
static inline nfloat_ptr bs(sine_builder *b, slong i)
{ return (nfloat_ptr) gr_vec_entry_ptr(b->stab, i, b->nctx); }
static inline nfloat_ptr poly_v(gr_vec_t v, int i, gr_ctx_t ctx)
{ return (nfloat_ptr) gr_vec_entry_ptr(v, i, ctx); }

static void builder_poly_zero(gr_vec_t v, int maxd, gr_ctx_t ctx)
{
    for (int i = 0; i <= maxd; i++)
        nfloat_zero((nfloat_ptr) gr_vec_entry_ptr(v, i, ctx), ctx);
}

static int builder_poly_mul(gr_vec_t out, gr_vec_t a, int da,
                            gr_vec_t b, int db, int maxd,
                            gr_ctx_t ctx, nfloat_ptr tmp)
{
    int st = GR_SUCCESS;
    builder_poly_zero(out, maxd, ctx);
    for (int i = 0; i <= da; i++)
        for (int j = 0; j <= db && i + j <= maxd; j++)
        {
            st |= nfloat_mul(tmp, poly_v(a, i, ctx), poly_v(b, j, ctx), ctx);
            st |= nfloat_add(poly_v(out, i + j, ctx),
                             poly_v(out, i + j, ctx), tmp, ctx);
        }
    return st;
}

static int builder_coefficients(sine_builder *b)
{
    int st = GR_SUCCESS;
    nfloat_ptr one = bv(b, BS_ONE);
    nfloat_ptr pi = bv(b, BS_PI), pi2 = bv(b, BS_PI2);
    nfloat_ptr A0 = bv(b, BS_A0), A1 = bv(b, BS_A1);
    nfloat_ptr Q0 = bv(b, BS_Q0), Q1 = bv(b, BS_Q1);
    nfloat_ptr qp0 = bv(b, BS_QP0), qp1 = bv(b, BS_QP1);
    nfloat_ptr G = bv(b, BS_G), acc = bv(b, BS_ACC);
    nfloat_ptr t0 = bv(b, BS_T0), t1 = bv(b, BS_T1);

    st |= nfloat_one(one, b->nctx);
    st |= nfloat_set_ui(bv(b, BS_TWO), 2, b->nctx);
    st |= nfloat_pi(pi, b->nctx);
    st |= nfloat_sqr(pi2, pi, b->nctx);

    st |= nfloat_set_ui(A0, 4, b->nctx);
    st |= nfloat_div(A0, A0, pi, b->nctx);
    st |= nfloat_set(A1, A0, b->nctx);
    st |= nfloat_div_ui(A1, A1, 3, b->nctx);
    st |= nfloat_neg(A1, A1, b->nctx);

    st |= nfloat_set_ui(Q0, 4, b->nctx);
    st |= nfloat_div(Q0, Q0, pi2, b->nctx);
    st |= nfloat_set(Q1, Q0, b->nctx);
    st |= nfloat_div_ui(Q1, Q1, 9, b->nctx);

    st |= nfloat_one(bq(b, 0), b->nctx);
    st |= nfloat_add(bq(b, 1), Q0, Q1, b->nctx);
    st |= nfloat_neg(bq(b, 1), bq(b, 1), b->nctx);
    st |= nfloat_mul(bq(b, 2), Q0, Q1, b->nctx);

    st |= nfloat_mul(bp(b, 0), A0, Q0, b->nctx);
    st |= nfloat_mul(t0, A1, Q1, b->nctx);
    st |= nfloat_add(bp(b, 0), bp(b, 0), t0, b->nctx);
    st |= nfloat_add(t0, A0, A1, b->nctx);
    st |= nfloat_mul(bp(b, 1), bq(b, 2), t0, b->nctx);
    st |= nfloat_neg(bp(b, 1), bp(b, 1), b->nctx);

    st |= nfloat_set(qp0, Q0, b->nctx);
    st |= nfloat_set(qp1, Q1, b->nctx);

    for (slong k = 0; k < SF_MAX_TERMS; k++)
    {
        ulong m = (ulong) k + 1UL;
        arith_euler_number(b->e, 2UL * m);
        fmpz_abs(b->e, b->e);
        fmpz_fac_ui(b->fact, 2UL * m);
        st |= nfloat_set_fmpz(G, b->e, b->nctx);
        st |= nfloat_set_fmpz(t1, b->fact, b->nctx);
        st |= nfloat_div(G, G, t1, b->nctx);

        st |= nfloat_mul(acc, A0, qp0, b->nctx);
        st |= nfloat_mul(t0, A1, qp1, b->nctx);
        st |= nfloat_add(acc, acc, t0, b->nctx);
        st |= nfloat_sub(br(b, k), G, acc, b->nctx);

        st |= nfloat_mul(qp0, qp0, Q0, b->nctx);
        st |= nfloat_mul(qp1, qp1, Q1, b->nctx);
    }
    return st;
}

static int builder_delta_cos(sine_builder *b, nfloat_ptr out,
                             nfloat_srcptr delta, int terms)
{
    int st = GR_SUCCESS;
    nfloat_ptr one = bv(b, BS_ONE), z = bv(b, BS_Z), R = bv(b, BS_R);
    nfloat_ptr gf = bv(b, BS_GF), Q = bv(b, BS_Q), P = bv(b, BS_P);
    nfloat_ptr den = bv(b, BS_DEN);

    st |= nfloat_sqr(z, delta, b->nctx);
    st |= nfloat_set(R, br(b, terms - 1), b->nctx);
    for (slong k = terms - 2; k >= 0; k--)
    {
        st |= nfloat_mul(R, R, z, b->nctx);
        st |= nfloat_add(R, R, br(b, k), b->nctx);
    }

    st |= nfloat_set(gf, one, b->nctx);
    st |= nfloat_addmul(gf, z, R, b->nctx);

    st |= nfloat_set(Q, bq(b, 2), b->nctx);
    st |= nfloat_mul(Q, Q, z, b->nctx);
    st |= nfloat_add(Q, Q, bq(b, 1), b->nctx);
    st |= nfloat_mul(Q, Q, z, b->nctx);
    st |= nfloat_add(Q, Q, bq(b, 0), b->nctx);

    st |= nfloat_set(P, bp(b, 1), b->nctx);
    st |= nfloat_mul(P, P, z, b->nctx);
    st |= nfloat_add(P, P, bp(b, 0), b->nctx);

    st |= nfloat_mul(den, Q, gf, b->nctx);
    st |= nfloat_addmul(den, z, P, b->nctx);
    st |= nfloat_div(out, Q, den, b->nctx);
    return st;
}

static int builder_lut(sine_builder *b)
{
    int st = GR_SUCCESS;
    nfloat_ptr one = bv(b, BS_ONE), delta = bv(b, BS_DELTA);
    nfloat_ptr seedc = bv(b, BS_SEEDC), seeds = bv(b, BS_SEEDS);
    nfloat_ptr lo = bv(b, BS_LO), hi = bv(b, BS_HI);
    nfloat_ptr lc = bv(b, BS_LUTC), ls = bv(b, BS_LUTS);

    st |= nfloat_set(delta, one, b->nctx);
    st |= nfloat_mul_2exp_si(delta, delta, -SF_K, b->nctx);
    st |= builder_delta_cos(b, seedc, delta, SF_MAX_TERMS);

    st |= nfloat_sub(lo, one, seedc, b->nctx);
    st |= nfloat_add(hi, one, seedc, b->nctx);
    st |= nfloat_mul(lo, lo, hi, b->nctx);
    st |= nfloat_sqrt(seeds, lo, b->nctx);

    st |= nfloat_one(bc(b, 0), b->nctx);
    st |= nfloat_zero(bs(b, 0), b->nctx);
    for (slong i = 1; i < (slong) SF_LUT_N; i++)
    {
        st |= nfloat_mul(lc, bc(b, i - 1), seedc, b->nctx);
        st |= nfloat_submul(lc, bs(b, i - 1), seeds, b->nctx);
        st |= nfloat_mul(ls, bs(b, i - 1), seedc, b->nctx);
        st |= nfloat_addmul(ls, bc(b, i - 1), seeds, b->nctx);
        st |= nfloat_set(bc(b, i), lc, b->nctx);
        st |= nfloat_set(bs(b, i), ls, b->nctx);
    }
    return st;
}

static int export_signed_fixed(sine_fixed_ctx *c, sine_builder *b,
                               mp_limb_t *dst, unsigned char *sgn,
                               nfloat_srcptr x)
{
    if (nfloat_get_arf(c->conv_a, x, b->nctx) != GR_SUCCESS) return 1;
    int neg = arf_sgn(c->conv_a) < 0;
    if (neg) arf_neg(c->conv_a, c->conv_a);
    arf_mul_2exp_si(c->conv_a, c->conv_a, c->B);
    arf_get_fmpz(c->conv_z, c->conv_a, ARF_RND_NEAR);
    if (fmpz_bits(c->conv_z) > (flint_bitcnt_t) (64 * c->n)) return 1;
    fmpz_get_ui_array(dst, c->n, c->conv_z);
    *sgn = (unsigned char) neg;
    return 0;
}

static int build_fused_coefficients(sine_fixed_ctx *c, sine_builder *b)
{
    const int d = c->spine_deg;
    gr_vec_t R,Q,P,GF,D,H,F,W,S,INV,C,T,TMP;
    gr_vec_init(R, SF_MAX_DEG + 1, b->nctx);
    gr_vec_init(Q, SF_MAX_DEG + 1, b->nctx);
    gr_vec_init(P, SF_MAX_DEG + 1, b->nctx);
    gr_vec_init(GF, SF_MAX_DEG + 1, b->nctx);
    gr_vec_init(D, SF_MAX_DEG + 1, b->nctx);
    gr_vec_init(H, SF_MAX_DEG + 1, b->nctx);
    gr_vec_init(F, SF_MAX_DEG + 1, b->nctx);
    gr_vec_init(W, SF_MAX_DEG + 1, b->nctx);
    gr_vec_init(S, SF_MAX_DEG + 1, b->nctx);
    gr_vec_init(INV, SF_MAX_DEG + 1, b->nctx);
    gr_vec_init(C, SF_MAX_DEG + 1, b->nctx);
    gr_vec_init(T, SF_MAX_DEG + 1, b->nctx);
    gr_vec_init(TMP, 4, b->nctx);

    for (int i = 0; i <= SF_MAX_DEG; i++)
    {
        nfloat_zero(poly_v(R,i,b->nctx),b->nctx);
        nfloat_zero(poly_v(Q,i,b->nctx),b->nctx);
        nfloat_zero(poly_v(P,i,b->nctx),b->nctx);
        nfloat_zero(poly_v(GF,i,b->nctx),b->nctx);
        nfloat_zero(poly_v(D,i,b->nctx),b->nctx);
        nfloat_zero(poly_v(H,i,b->nctx),b->nctx);
        nfloat_zero(poly_v(F,i,b->nctx),b->nctx);
        nfloat_zero(poly_v(W,i,b->nctx),b->nctx);
        nfloat_zero(poly_v(S,i,b->nctx),b->nctx);
        nfloat_zero(poly_v(INV,i,b->nctx),b->nctx);
        nfloat_zero(poly_v(C,i,b->nctx),b->nctx);
        nfloat_zero(poly_v(T,i,b->nctx),b->nctx);
    }

    nfloat_ptr t0 = poly_v(TMP,0,b->nctx);
    nfloat_ptr acc = poly_v(TMP,1,b->nctx);
    nfloat_ptr acoef = poly_v(TMP,2,b->nctx);
    nfloat_ptr bcoef = poly_v(TMP,3,b->nctx);
    int st = GR_SUCCESS;

    for (int i = 0; i < c->terms; i++)
        st |= nfloat_set(poly_v(R,i,b->nctx), br(b,i), b->nctx);

    st |= nfloat_one(poly_v(Q,0,b->nctx), b->nctx);
    st |= nfloat_set(poly_v(Q,1,b->nctx), bq(b,1), b->nctx);
    st |= nfloat_set(poly_v(Q,2,b->nctx), bq(b,2), b->nctx);
    st |= nfloat_set(poly_v(P,0,b->nctx), bp(b,0), b->nctx);
    st |= nfloat_set(poly_v(P,1,b->nctx), bp(b,1), b->nctx);

    st |= nfloat_one(poly_v(GF,0,b->nctx), b->nctx);
    for (int i = 0; i < c->terms && i + 1 <= SF_MAX_DEG; i++)
        st |= nfloat_set(poly_v(GF,i+1,b->nctx), poly_v(R,i,b->nctx), b->nctx);

    st |= builder_poly_mul(D,Q,2,GF,c->terms,SF_MAX_DEG,b->nctx,t0);
    st |= nfloat_add(poly_v(D,1,b->nctx),poly_v(D,1,b->nctx),poly_v(P,0,b->nctx),b->nctx);
    st |= nfloat_add(poly_v(D,2,b->nctx),poly_v(D,2,b->nctx),poly_v(P,1,b->nctx),b->nctx);

    st |= builder_poly_mul(H,Q,2,R,c->terms-1,SF_MAX_DEG,b->nctx,t0);
    st |= nfloat_add(poly_v(H,0,b->nctx),poly_v(H,0,b->nctx),poly_v(P,0,b->nctx),b->nctx);
    st |= nfloat_add(poly_v(H,1,b->nctx),poly_v(H,1,b->nctx),poly_v(P,1,b->nctx),b->nctx);

    for (int i = 0; i <= 2; i++)
        st |= nfloat_add(poly_v(F,i,b->nctx),poly_v(Q,i,b->nctx),poly_v(Q,i,b->nctx),b->nctx);
    for (int i = 0; i <= c->terms + 1 && i + 1 <= SF_MAX_DEG; i++)
        st |= nfloat_add(poly_v(F,i+1,b->nctx),poly_v(F,i+1,b->nctx),poly_v(H,i,b->nctx),b->nctx);

    st |= builder_poly_mul(W,H,c->terms+1,F,c->terms+2,SF_MAX_DEG,b->nctx,t0);

    st |= nfloat_one(poly_v(S,0,b->nctx), b->nctx);
    for (int k = 1; k <= d; k++)
    {
        st |= nfloat_set(acc, poly_v(W,k,b->nctx), b->nctx);
        for (int i = 1; i < k; i++)
        {
            st |= nfloat_mul(t0,poly_v(S,i,b->nctx),poly_v(S,k-i,b->nctx),b->nctx);
            st |= nfloat_sub(acc,acc,t0,b->nctx);
        }
        st |= nfloat_div_ui(poly_v(S,k,b->nctx),acc,2,b->nctx);
    }

    st |= nfloat_one(poly_v(INV,0,b->nctx), b->nctx);
    for (int k = 1; k <= d; k++)
    {
        st |= nfloat_zero(acc,b->nctx);
        for (int i = 1; i <= k; i++)
        {
            st |= nfloat_mul(t0,poly_v(D,i,b->nctx),poly_v(INV,k-i,b->nctx),b->nctx);
            st |= nfloat_add(acc,acc,t0,b->nctx);
        }
        st |= nfloat_neg(poly_v(INV,k,b->nctx),acc,b->nctx);
    }

    st |= builder_poly_mul(C,Q,2,INV,d,d,b->nctx,t0);
    st |= builder_poly_mul(T,S,d,INV,d,d,b->nctx,t0);
    if (st != GR_SUCCESS) goto done;

    for (slong a = 0; a < (slong) SF_LUT_N; a++)
    {
        for (int k = 0; k <= d; k++)
        {
            st |= nfloat_mul(acoef, bs(b,a), poly_v(C,k,b->nctx), b->nctx);
            st |= nfloat_mul(bcoef, bc(b,a), poly_v(T,k,b->nctx), b->nctx);
            if (st != GR_SUCCESS) goto done;
            size_t even = ((size_t)a * (size_t)(c->poly_deg + 1) + (size_t)(2*k));
            size_t odd = even + 1;
            if (export_signed_fixed(c,b,c->coef + even*(size_t)c->n,&c->coef_sign[even],acoef)) { st=GR_DOMAIN; goto done; }
            if (export_signed_fixed(c,b,c->coef + odd*(size_t)c->n,&c->coef_sign[odd],bcoef)) { st=GR_DOMAIN; goto done; }
        }
    }

done:
    gr_vec_clear(TMP,b->nctx);
    gr_vec_clear(T,b->nctx); gr_vec_clear(C,b->nctx);
    gr_vec_clear(INV,b->nctx); gr_vec_clear(S,b->nctx);
    gr_vec_clear(W,b->nctx); gr_vec_clear(F,b->nctx);
    gr_vec_clear(H,b->nctx); gr_vec_clear(D,b->nctx);
    gr_vec_clear(GF,b->nctx); gr_vec_clear(P,b->nctx);
    gr_vec_clear(Q,b->nctx); gr_vec_clear(R,b->nctx);
    return st == GR_SUCCESS ? 0 : 1;
}

static int builder_init(sine_builder *b, slong bits)
{
    if (nfloat_ctx_init(b->nctx, bits, 0) != GR_SUCCESS) return 1;
    gr_vec_init(b->residual, SF_MAX_TERMS, b->nctx);
    gr_vec_init(b->pcoef, 2, b->nctx);
    gr_vec_init(b->qcoef, 3, b->nctx);
    gr_vec_init(b->ctab, SF_LUT_N, b->nctx);
    gr_vec_init(b->stab, SF_LUT_N, b->nctx);
    gr_vec_init(b->scratch, BS_COUNT, b->nctx);
    fmpz_init(b->e);
    fmpz_init(b->fact);
    if (builder_coefficients(b) != GR_SUCCESS || builder_lut(b) != GR_SUCCESS)
    {
        fmpz_clear(b->fact); fmpz_clear(b->e);
        gr_vec_clear(b->scratch,b->nctx); gr_vec_clear(b->stab,b->nctx);
        gr_vec_clear(b->ctab,b->nctx); gr_vec_clear(b->qcoef,b->nctx);
        gr_vec_clear(b->pcoef,b->nctx); gr_vec_clear(b->residual,b->nctx);
        gr_ctx_clear(b->nctx);
        return 1;
    }
    return 0;
}

static void builder_clear(sine_builder *b)
{
    fmpz_clear(b->fact); fmpz_clear(b->e);
    gr_vec_clear(b->scratch,b->nctx); gr_vec_clear(b->stab,b->nctx);
    gr_vec_clear(b->ctab,b->nctx); gr_vec_clear(b->qcoef,b->nctx);
    gr_vec_clear(b->pcoef,b->nctx); gr_vec_clear(b->residual,b->nctx);
    gr_ctx_clear(b->nctx);
}

static sine_fixed_ctx *s53_coeff_create_terms(int terms)
{
    if (terms < 1 || terms > 3) return NULL;
    sine_fixed_ctx *c = (sine_fixed_ctx *) flint_calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->target_bits = S53_SOURCE_TARGET_BITS;
    c->B = S53_SOURCE_B;
    c->n = 2;
    c->terms = terms;
    c->spine_deg = terms;
    c->poly_deg = 2 * terms + 1;
    size_t coeff_count = (size_t)SF_LUT_N * (size_t)(c->poly_deg + 1);
    c->coef = (mp_limb_t *) flint_calloc(2 * coeff_count, sizeof(mp_limb_t));
    c->coef_sign = (unsigned char *) flint_calloc(coeff_count, 1);
    fmpz_init(c->conv_z); arf_init(c->conv_a);
    if (!c->coef || !c->coef_sign) goto fail;
    sine_builder b;
    if (builder_init(&b, c->target_bits + SF_GUARD_BITS)) goto fail;
    int bad = build_fused_coefficients(c, &b);
    builder_clear(&b);
    if (bad) goto fail;
    return c;
fail:
    arf_clear(c->conv_a); fmpz_clear(c->conv_z);
    flint_free(c->coef_sign); flint_free(c->coef); flint_free(c);
    return NULL;
}

static void s53_coeff_destroy(sine_fixed_ctx *c)
{
    if (!c) return;
    arf_clear(c->conv_a); fmpz_clear(c->conv_z);
    flint_free(c->coef_sign); flint_free(c->coef); flint_free(c);
}
