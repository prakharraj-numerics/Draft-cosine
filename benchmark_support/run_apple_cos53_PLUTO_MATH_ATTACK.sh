#!/usr/bin/env bash
set -euo pipefail

# Mathematical successor search from frozen PLUTO.
# Every variant preserves the Mode-5/secant-spine cosine coefficient source.
# Only representation changes: local grid K, degree-3 vs degree-5 local form,
# grouped X50/X67-style degree-5 evaluation, and X50-style |x|<1 bypass.

[[ "$(uname -m)" == arm64 ]]
[[ "$(sysctl -n machdep.cpu.brand_string)" == *"Apple M1"* ]]
MODE="${1:-}"
VARIANTS=(k256d5g k512d5g k1024d3 k2048d3 k512d3 k1024d5g pluto_unit k1024d3_unit)

if [[ "$MODE" == build ]]; then
  # Exact controls and exact current PLUTO source/LUT generation machinery.
  bash benchmark_support/run_apple_cos53_PLUTO.sh build
  test -f /tmp/kernel_attack_no_p3.cpp
  test -x /tmp/kernel_attack_no_p3
  test -x /tmp/apple_cos53_off_frozen

  FP="$(brew --prefix flint)"; MP="$(brew --prefix mpfr)"; GP="$(brew --prefix gmp)"
  cp benchmark_support/sine_53_coeff_source.c /tmp/math_coeff_source_master.c
  cp cosine53_apply_formula_conversion.py /tmp/cosine53_apply_formula_conversion.py

  cat >/tmp/math_make_bridge.py <<'PY'
from pathlib import Path
import math, sys
name=sys.argv[1]; K=int(sys.argv[2]); terms=int(sys.argv[3]); deg=2*terms+1
sfk=int(math.log2(K)); assert 2**sfk==K
lutn=int(math.floor(K*math.pi/2.0+0.5))+1
s=Path('/tmp/math_coeff_source_master.c').read_text()
s=s.replace('#define SF_K 12',f'#define SF_K {sfk}',1)
s=s.replace('#define SF_LUT_N ((1UL << SF_K) + 1UL)',f'#define SF_LUT_N {lutn}UL',1)
Path(f'/tmp/math_src_{name}.c').write_text(s)
import importlib.util
spec=importlib.util.spec_from_file_location('conv','/tmp/cosine53_apply_formula_conversion.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.patch_coeff(Path(f'/tmp/math_src_{name}.c'))
bridge=f'''#include <flint/arf.h>\n#include <flint/flint.h>\n#include <flint/fmpz.h>\n#include <stddef.h>\n#include <stdint.h>\n#include \"/tmp/math_src_{name}.c\"\nstatic double f103(const mp_limb_t q[2], int neg){{fmpz_t z;arf_t a;fmpz_init(z);arf_init(a);fmpz_set_ui_array(z,q,2);arf_set_fmpz(a,z);arf_mul_2exp_si(a,a,-103);double d=arf_get_d(a,ARF_RND_NEAR);arf_clear(a);fmpz_clear(z);return neg?-d:d;}}\nint main(void){{sine_fixed_ctx*c=s53_coeff_create_terms({terms});if(!c||c->poly_deg!={deg})return 2;for(size_t a=0;a<(size_t)SF_LUT_N;a++){{size_t off=a*(size_t)(c->poly_deg+1);double c0=f103(c->coef+2*(off+0),c->coef_sign[off+0]!=0);double c1=f103(c->coef+2*(off+1),c->coef_sign[off+1]!=0);printf(\"%a %a\\n\",c0,c1);}}s53_coeff_destroy(c);flint_cleanup_master();return 0;}}\n'''
Path(f'/tmp/math_bridge_{name}.c').write_text(bridge)
print(lutn)
PY

  gen_profile() {
    local name="$1" K="$2" terms="$3" form="$4" unit="${5:-0}"
    local lutn
    lutn="$(python3 /tmp/math_make_bridge.py "$name" "$K" "$terms")"
    clang -O2 -DNDEBUG -I/tmp -I"$FP/include" -I"$MP/include" -I"$GP/include" \
      "/tmp/math_bridge_${name}.c" -L"$FP/lib" -L"$MP/lib" -L"$GP/lib" \
      -lflint -lmpfr -lgmp -lm -o "/tmp/math_gen_${name}"
    "/tmp/math_gen_${name}" > "/tmp/math_raw_${name}.txt"
    test "$(wc -l < "/tmp/math_raw_${name}.txt" | tr -d ' ')" = "$lutn"

    python3 - "$name" "$K" "$terms" "$form" "$unit" "$lutn" <<'PY'
from pathlib import Path
from fractions import Fraction
import math,sys
name=sys.argv[1]; K=int(sys.argv[2]); terms=int(sys.argv[3]); form=sys.argv[4]; unit=int(sys.argv[5]); lutn=int(sys.argv[6])
pairs=[]
for line in Path(f'/tmp/math_raw_{name}.txt').read_text().splitlines():
    a,b=line.split(); pairs.append((float.fromhex(a),float.fromhex(b)))
assert len(pairs)==lutn
if terms==1:
    h=Fraction(1,2*K); h2=h*h; h4=h2*h2
    A0=Fraction(1)-h4/Fraction(192)
    B1=Fraction(1)-h4/Fraction(384)
    A2=-Fraction(1,2)+h2/Fraction(24)
    B3=-Fraction(1,6)+h2/Fraction(96)
    mh=float(A2/A0); m6=float(B3/B1)
    pairs=[(float(Fraction.from_float(a)*A0),float(Fraction.from_float(b)*B1)) for a,b in pairs]
else:
    mh=-0.5; m6=-1.0/6.0
hdr=['#pragma once','#include <cstddef>','#include <cstdint>',f'static constexpr std::size_t OPT_COS53_LUTN = {lutn};','alignas(16) static const double opt_cos53_coeff_aos[OPT_COS53_LUTN*2] = {']
for a,b in pairs: hdr.append(f'  {a.hex()}, {b.hex()},')
hdr.append('};')
Path(f'/tmp/math_coeff_{name}.h').write_text('\n'.join(hdr)+'\n')

s=Path('/tmp/kernel_attack_no_p3.cpp').read_text()
s=s.replace('#include "apple_cos53_coeff_aos.h"',f'#include "math_coeff_{name}.h"',1)
s=s.replace('static constexpr double KGRID = 1280.0;',f'static constexpr double KGRID = {float(K).hex()};',1)
# Dyadic K: local grid subtraction needs one FMA instead of PLUTO's split 1/1280.
s=s.replace('static constexpr double NINVK_HI = -0x1.999999999999ap-11;\nstatic constexpr double NINVK_LO = 0x1.999999999999ap-65;',f'static constexpr double NINVK = {-1.0/K.hex() if False else (-1.0/K).hex()};',1)
old='''    // Split reciprocal subtraction: delta = |r| - j/1280 + low.\n    float64x2_t delta=vfmaq_n_f64(ah,jd,NINVK_HI);\n    delta=vfmaq_n_f64(delta,jd,NINVK_LO);\n    delta=vaddq_f64(delta,al);\n'''
new='''    // Dyadic-grid subtraction: delta = |r| - j/K + low in one FMA.\n    float64x2_t delta=vfmaq_n_f64(ah,jd,NINVK);\n    delta=vaddq_f64(delta,al);\n'''
assert old in s
s=s.replace(old,new,1)
# Retuned degree-3 multipliers, or exact degree-5 Taylor structure around Mode-5 anchors.
s=s.replace('static constexpr double MH = -0x1.ffffff92c5f94p-2;',f'static constexpr double MH = {mh.hex()};',1)
s=s.replace('static constexpr double M6 = -0x1.5555551eb851fp-3;',f'static constexpr double M6 = {m6.hex()};',1)
oldpoly='''    // Frozen degree-3 minimax-retuned polynomial.\n    float64x2_t c2=vmulq_n_f64(c0,MH);\n    float64x2_t c3=vmulq_n_f64(c1,M6);\n    float64x2_t p=vfmaq_f64(c2,c3,delta);\n    p=vfmaq_f64(c1,p,delta);\n    p=vfmaq_f64(c0,p,delta);\n'''
if terms==1:
    newpoly=oldpoly.replace('Frozen degree-3 minimax-retuned','Mode-5 degree-3 grid-retuned')
else:
    if form=='grouped':
        newpoly='''    // X50/X67-inspired grouped degree-5 local cosine polynomial.\n    float64x2_t z=vmulq_f64(delta,delta);\n    float64x2_t base=vfmaq_f64(c0,c1,delta);\n    float64x2_t even=vfmaq_n_f64(vdupq_n_f64(-0.5),z,1.0/24.0);\n    float64x2_t odd=vfmaq_n_f64(vdupq_n_f64(-1.0/6.0),z,1.0/120.0);\n    float64x2_t cd=vmulq_f64(c1,delta);\n    float64x2_t inner=vfmaq_f64(vmulq_f64(c0,even),cd,odd);\n    float64x2_t p=vfmaq_f64(base,z,inner);\n'''
    else:
        newpoly='''    // Degree-5 Horner local cosine polynomial.\n    float64x2_t c2=vmulq_n_f64(c0,-0.5);\n    float64x2_t c3=vmulq_n_f64(c1,-1.0/6.0);\n    float64x2_t c4=vmulq_n_f64(c0,1.0/24.0);\n    float64x2_t c5=vmulq_n_f64(c1,1.0/120.0);\n    float64x2_t p=vfmaq_f64(c4,c5,delta);\n    p=vfmaq_f64(c3,p,delta);\n    p=vfmaq_f64(c2,p,delta);\n    p=vfmaq_f64(c1,p,delta);\n    p=vfmaq_f64(c0,p,delta);\n'''
assert oldpoly in s
s=s.replace(oldpoly,newpoly,1)
maxj=int(math.floor(K*math.pi/2.0+0.5)); rootj=max(0,maxj-2)
s=s.replace('j0 >= 2009',f'j0 >= {rootj}').replace('j1 >= 2009',f'j1 >= {rootj}')
# X50 idea: when both lanes are |x|<1, skip q/pi range reduction entirely.
if unit:
    start=s.index('    // q = nearest integer to |x|/pi.')
    end=s.index('    // K=1280 and |r|<=pi/2 imply j is in [0,2011].')
    block=s[start:end]
    # Extract original q/reduction body, then wrap it with a homogeneous-pair fast path.
    repl='''    const float64x2_t magic=vdupq_n_f64(0x1p52);\n    float64x2_t ah, al;\n    uint64x2_t qbits;\n    uint64x2_t um=vcltq_f64(ax,vdupq_n_f64(1.0));\n    if (__builtin_expect((vgetq_lane_u64(um,0)&vgetq_lane_u64(um,1))==UINT64_MAX, 0)) {\n        ah=ax; al=vdupq_n_f64(0.0); qbits=vdupq_n_u64(0);\n    } else {\n        float64x2_t qscaled=vmulq_n_f64(ax,INVPI);\n        float64x2_t qmagic=vaddq_f64(qscaled,magic);\n        float64x2_t qd=vsubq_f64(qmagic,magic);\n        qbits=vreinterpretq_u64_f64(qmagic);\n        float64x2_t qp1=vmulq_n_f64(qd,PI_P1);\n        float64x2_t t=vsubq_f64(ax,qp1);\n        float64x2_t rh=vfmaq_n_f64(t,qd,-PI_P2);\n        float64x2_t d=vsubq_f64(t,rh);\n        float64x2_t rl=vfmaq_n_f64(d,qd,-PI_P2);\n        const uint64x2_t signmask=vdupq_n_u64(UINT64_C(0x8000000000000000));\n        uint64x2_t rsign=vandq_u64(vreinterpretq_u64_f64(rh),signmask);\n        ah=vreinterpretq_f64_u64(vbicq_u64(vreinterpretq_u64_f64(rh),signmask));\n        al=vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(rl),rsign));\n    }\n\n'''
    s=s[:start]+repl+s[end:]
# Keep hard proof markers.
assert 'rl=vfmaq_n_f64(rl,qd,-PI_P3);' not in s
assert 'builder_delta_cos' not in s
Path(f'/tmp/math_{name}.cpp').write_text(s)
PY

    local COMMON="-O3 -DNDEBUG -std=c++20 -mcpu=native -fno-fast-math -ffp-contract=fast"
    clang++ $COMMON -I/tmp -I/tmp/pthreadpool-install/include \
      "/tmp/math_${name}.cpp" /tmp/pthreadpool-install/lib/libpthreadpool.a \
      -framework Accelerate -pthread -o "/tmp/math_${name}_bench"
    clang++ $COMMON -DOPT_VALIDATE_MPFR -I/tmp -I/tmp/pthreadpool-install/include -I"$MP/include" -I"$GP/include" \
      "/tmp/math_${name}.cpp" /tmp/pthreadpool-install/lib/libpthreadpool.a \
      -framework Accelerate -pthread -L"$MP/lib" -L"$GP/lib" -lmpfr -lgmp -o "/tmp/math_${name}_validate"
  }

  # Intel-derived and mathematical profiles.
  gen_profile k256d5g 256 2 grouped 0
  gen_profile k512d5g 512 2 grouped 0
  gen_profile k1024d3 1024 1 d3 0
  gen_profile k2048d3 2048 1 d3 0
  gen_profile k512d3 512 1 d3 0
  gen_profile k1024d5g 1024 2 grouped 0

  # X50-style direct unit-domain bypass on exact PLUTO and on the strongest dyadic D3 candidate.
  # PLUTO-unit is generated by using K=1280's existing LUT/source then only restructuring q reduction.
  python3 - <<'PY'
from pathlib import Path
s=Path('/tmp/kernel_attack_no_p3.cpp').read_text()
start=s.index('    // q = nearest integer to |x|/pi.')
end=s.index('    // K=1280 and |r|<=pi/2 imply j is in [0,2011].')
repl='''    const float64x2_t magic=vdupq_n_f64(0x1p52);\n    float64x2_t ah, al; uint64x2_t qbits;\n    uint64x2_t um=vcltq_f64(ax,vdupq_n_f64(1.0));\n    if (__builtin_expect((vgetq_lane_u64(um,0)&vgetq_lane_u64(um,1))==UINT64_MAX, 0)) {\n        ah=ax; al=vdupq_n_f64(0.0); qbits=vdupq_n_u64(0);\n    } else {\n        float64x2_t qscaled=vmulq_n_f64(ax,INVPI); float64x2_t qmagic=vaddq_f64(qscaled,magic);\n        float64x2_t qd=vsubq_f64(qmagic,magic); qbits=vreinterpretq_u64_f64(qmagic);\n        float64x2_t qp1=vmulq_n_f64(qd,PI_P1); float64x2_t t=vsubq_f64(ax,qp1);\n        float64x2_t rh=vfmaq_n_f64(t,qd,-PI_P2); float64x2_t d=vsubq_f64(t,rh);\n        float64x2_t rl=vfmaq_n_f64(d,qd,-PI_P2);\n        const uint64x2_t signmask=vdupq_n_u64(UINT64_C(0x8000000000000000));\n        uint64x2_t rsign=vandq_u64(vreinterpretq_u64_f64(rh),signmask);\n        ah=vreinterpretq_f64_u64(vbicq_u64(vreinterpretq_u64_f64(rh),signmask));\n        al=vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(rl),rsign));\n    }\n\n'''
s=s[:start]+repl+s[end:]
Path('/tmp/math_pluto_unit.cpp').write_text(s)
PY
  COMMON="-O3 -DNDEBUG -std=c++20 -mcpu=native -fno-fast-math -ffp-contract=fast"
  clang++ $COMMON -I/tmp -I/tmp/pthreadpool-install/include /tmp/math_pluto_unit.cpp /tmp/pthreadpool-install/lib/libpthreadpool.a -framework Accelerate -pthread -o /tmp/math_pluto_unit_bench
  clang++ $COMMON -DOPT_VALIDATE_MPFR -I/tmp -I/tmp/pthreadpool-install/include -I"$MP/include" -I"$GP/include" /tmp/math_pluto_unit.cpp /tmp/pthreadpool-install/lib/libpthreadpool.a -framework Accelerate -pthread -L"$MP/lib" -L"$GP/lib" -lmpfr -lgmp -o /tmp/math_pluto_unit_validate
  # Alias the already-generated K1024D3 unit variant.
  cp /tmp/math_k1024d3_unit_bench /tmp/math_k1024d3_unit_bench 2>/dev/null || true
  exit 0
fi

if [[ "$MODE" == variants ]]; then printf '%s\n' "${VARIANTS[@]}"; exit 0; fi
if [[ "$MODE" == pluto ]]; then [[ $# -eq 2 ]]; exec /tmp/kernel_attack_no_p3 single "$2"; fi
if [[ "$MODE" == apple ]]; then [[ $# -eq 2 ]]; exec env VECLIB_MAXIMUM_THREADS=1 /tmp/apple_cos53_off_frozen apple "$2"; fi
if [[ "$MODE" == one ]]; then [[ $# -eq 3 ]]; exec "/tmp/math_${2}_bench" single "$3"; fi
if [[ "$MODE" == validate ]]; then [[ $# -eq 2 ]]; exec "/tmp/math_${2}_validate" validate; fi

echo "usage: $0 build | variants | pluto N | apple N | one VARIANT N | validate VARIANT" >&2
exit 2
