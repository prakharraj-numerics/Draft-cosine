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
    # The scalar base reducer originally floors to a pi interval and folds around pi/2,
    # which is sign-free for sine but not for cosine.  Nearest-pi reduction makes the
    # cosine reconstruction exactly (-1)^q cos(|r|).
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
    # This older base AVX-512 evaluator is not the production X50/X67 path, but keep
    # its sign semantics cosine-correct as well.
    s = s.replace("__mmask8 msign=(__mmask8)(minput^modd);", "__mmask8 msign=modd;")
    path.write_text(s)


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
    # With existing masks this is exactly reflection_bit XOR sine_quadrant_bit.
    s = s.replace("inneg^wide_neg", "rev^wide_neg")

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

    print("COSINE53_FORMULA_CONVERSION_OK spine=Mode5_secant fusion=cosC_minus_sinT sign=cosine_parity")


if __name__ == "__main__":
    main()
