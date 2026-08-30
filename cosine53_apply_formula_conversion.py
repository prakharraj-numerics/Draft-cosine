#!/usr/bin/env python3
from pathlib import Path
import argparse


def replace_exact(text, old, new, *, count=None, label="replacement"):
    n = text.count(old)
    if count is not None and n != count:
        raise SystemExit(f"{label}: expected {count} occurrence(s), found {n}: {old!r}")
    if count is None and n == 0:
        raise SystemExit(f"{label}: pattern not found: {old!r}")
    return text.replace(old, new)


def patch_coeff(path: Path):
    s = path.read_text()
    old = """            st |= nfloat_mul(acoef, bs(b,a), poly_v(C,k,b->nctx), b->nctx);\n            st |= nfloat_mul(bcoef, bc(b,a), poly_v(T,k,b->nctx), b->nctx);"""
    new = """            /* Cosine fusion, preserving the exact Mode-5 secant spine:\n               cos(a+d) = cos(a) C(d^2) - sin(a) d T(d^2). */\n            st |= nfloat_mul(acoef, bc(b,a), poly_v(C,k,b->nctx), b->nctx);\n            st |= nfloat_mul(bcoef, bs(b,a), poly_v(T,k,b->nctx), b->nctx);\n            st |= nfloat_neg(bcoef, bcoef, b->nctx);"""
    s = replace_exact(s, old, new, count=1, label="coefficient fusion rotation")
    path.write_text(s)


def patch_fast2(path: Path):
    s = path.read_text()
    s = replace_exact(s, "vmdSin", "vmdCos", label="fast2 Intel comparator")
    s = replace_exact(s, "arb_sin", "arb_cos", label="fast2 Arb reference")
    s = replace_exact(
        s,
        "p=fma(rl,dp,p);if(signbit(x)^(int)(q&1)^rn)p=-p;return p;",
        "p=fma(rl,dp,p);if((int)(q&1))p=-p;return p;",
        count=1,
        label="fast2 scalar cosine parity",
    )
    s = replace_exact(
        s,
        "__mmask8 sg=(__mmask8)((minput^odd^rn)&active);",
        "__mmask8 sg=(__mmask8)(odd&active);",
        count=1,
        label="fast2 vector cosine parity",
    )
    path.write_text(s)


def patch_intel_base(path: Path):
    s = path.read_text()
    s = replace_exact(s, "vmdSin", "vmdCos", label="base Intel comparator")
    s = replace_exact(s, "arb_sin", "arb_cos", label="base Arb reference")
    old_reduce = "static inline double reduce_scalar(double x,int64_t *qout){double ax=fabs(x);int64_t q=(int64_t)(ax*INVPI);double qd=(double)q;double r=fma(-qd,PI_HI,ax);r=fma(-qd,PI_LO,r);if(r<0.0){q--;r+=PI_HI;r+=PI_LO;}else if(r>PI_HI){q++;r-=PI_HI;r-=PI_LO;}if(r>HALFPI_HI){r=(PI_HI-r)+PI_LO;}*qout=q;return r;}"
    new_reduce = "static inline double reduce_scalar(double x,int64_t *qout){double ax=fabs(x);int64_t q=(int64_t)nearbyint(ax*INVPI);double qd=(double)q;double r=fma(-qd,PI_HI,ax);r=fma(-qd,PI_LO,r);if(r<0.0)r=-r;*qout=q;return r;}"
    s = replace_exact(s, old_reduce, new_reduce, count=1, label="base nearest-pi cosine reducer")
    s = replace_exact(
        s,
        "if(signbit(x)^(int)(q&1))y=-y;return y;",
        "if((int)(q&1))y=-y;return y;",
        count=1,
        label="base scalar cosine parity",
    )
    s = s.replace("__mmask8 msign=(__mmask8)(minput^modd);", "__mmask8 msign=modd;")
    path.write_text(s)


def patch_x67_raw_polynomial(s: str, name: str) -> str:
    """Rotate X67's static sin/cos anchor fast path to cosine.

    X67 bypasses k->tab and gathers x65_s/x65_c directly.  Its original grouped
    polynomial is sin(a+d).  Keep the same degree-5/FMA grouping but evaluate
    cos(a+d)=c-d*s + z[c*(-1/2+z/24) - (s*d)*(-1/6+z/120)].
    The existing sg bit is the pi-period sign bit from the 512-entry table index,
    so it remains valid for cosine as well.
    """
    marker = "OVEC __attribute__((noinline,hot,aligned(64))) static void octant_vector_v8_rawx67("
    if marker not in s:
        return s

    old = """            __m512d cd=_mm512_mul_pd(c1[g],d[g]);
            /* Same polynomial, but fuse the potentially cancelling leading
               terms first: base = c0 + c1*d.  Corrections are O(z). */
            __m512d base=_mm512_fmadd_pd(c1[g],d[g],c0[g]);
            __m512d ep=_mm512_mul_pd(c0[g],ec);
            __m512d inner=_mm512_fmadd_pd(cd,oc,ep);
            pv[g]=_mm512_fmadd_pd(z,inner,base);"""
    new = """            __m512d sd=_mm512_mul_pd(c0[g],d[g]);
            /* Cosine rotation of the same grouped degree-5 local polynomial:
               base = cos(a) - sin(a)*d; corrections remain O(z). */
            __m512d base=_mm512_fnmadd_pd(c0[g],d[g],c1[g]);
            __m512d ep=_mm512_mul_pd(c1[g],ec);
            __m512d inner=_mm512_fnmadd_pd(sd,oc,ep);
            pv[g]=_mm512_fmadd_pd(z,inner,base);"""
    return replace_exact(s, old, new, count=1, label=f"{name}: X67 raw cosine polynomial")


def patch_production(path: Path):
    s = path.read_text()
    s = replace_exact(s, "vmdSin", "vmdCos", label=f"{path.name}: Intel comparator")
    s = replace_exact(s, "arb_sin", "arb_cos", label=f"{path.name}: Arb reference")

    # cos is even on the direct |x|<1 path.
    s = s.replace("return signbit(x)?-p:p;", "return p;")

    # Pure-unit vector blocks must never inherit the input sign.
    s = s.replace("mode5_poly_x11(k,ax,inneg)", "mode5_poly_x11(k,ax,0)")
    s = s.replace("*sign_out=inneg;", "*sign_out=0;")

    # Nearest-pi reducers: cos(x)=(-1)^q cos(|r|); input sign and residual sign vanish.
    s = s.replace("(inneg^parity^rneg)&active", "parity&active")
    s = s.replace("inneg^parity^rneg", "parity")

    # Guarded pi/4 reducer: cosine is negative in octants 2,3,4,5.
    s = s.replace("inneg^wide_neg", "rev^wide_neg")

    # X67's static 512-anchor fast path bypasses generated k->tab, so rotate its
    # grouped local polynomial explicitly while preserving the Mode-5 degree/FMA shape.
    s = patch_x67_raw_polynomial(s, path.name)

    # Any source-level scalar direct sign reconstruction must disappear for cosine.
    s = s.replace("signbit(x)?-p:p", "p")

    forbidden = ["vmdSin", "arb_sin", "inneg^parity^rneg", "inneg^wide_neg", "signbit(x)?-p:p"]
    for token in forbidden:
        if token in s:
            raise SystemExit(f"{path.name}: forbidden sine semantic remains: {token}")
    path.write_text(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coeff", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--fast2", type=Path, required=True)
    ap.add_argument("--source", type=Path, action="append", required=True)
    args = ap.parse_args()

    patch_coeff(args.coeff)
    patch_intel_base(args.base)
    patch_fast2(args.fast2)
    for p in args.source:
        patch_production(p)

    print("COSINE53_FORMULA_CONVERSION_OK spine=Mode5_secant fusion=cosC_minus_sinT sign=cosine_parity x67_raw=cosine_rotated")


if __name__ == "__main__":
    main()
