#!/usr/bin/env bash
set -euo pipefail
[[ "$(uname -m)" == arm64 ]]
[[ "$(sysctl -n machdep.cpu.brand_string)" == *"Apple M1"* ]]
brew list flint >/dev/null 2>&1 || brew install flint
brew list mpfr >/dev/null 2>&1 || brew install mpfr
brew list gmp >/dev/null 2>&1 || brew install gmp

FREEZE=aefbe778e860ef70e64fc8d6b6d470b3575f3bbc
git show "$FREEZE":benchmark_support/sine_53_coeff_source.c > /tmp/src.c
git show "$FREEZE":benchmark_support/apple_cos53_coeff_bridge.c > /tmp/bridge.c
git show "$FREEZE":benchmark_support/apple_cos53_generate_constants.c > /tmp/gen.c
git show "$FREEZE":cosine53_apply_formula_conversion.py > /tmp/cosine53_apply_formula_conversion.py
python3 - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0,'/tmp')
p=Path('/tmp/src.c')
s=p.read_text().replace('#define SF_K 12','#define SF_K 11').replace('#define SF_LUT_N ((1UL << SF_K) + 1UL)','#define SF_LUT_N 3218UL')
p.write_text(s)
p=Path('/tmp/bridge.c')
s=p.read_text().replace('#include "apple_cosine53_coeff_source.c"','#include "src.c"').replace('s53_coeff_create_terms(2)','s53_coeff_create_terms(1)').replace('c->poly_deg != 5','c->poly_deg != 3')
p.write_text(s)
p=Path('/tmp/gen.c'); p.write_text(p.read_text().replace('#define LUTN 403','#define LUTN 3218'))
import cosine53_apply_formula_conversion as m
m.patch_coeff(Path('/tmp/src.c'))
PY
FP="$(brew --prefix flint)"; MP="$(brew --prefix mpfr)"; GP="$(brew --prefix gmp)"
clang -O2 -DNDEBUG -I/tmp -I$FP/include -I$MP/include -I$GP/include /tmp/bridge.c /tmp/gen.c -L$FP/lib -L$MP/lib -L$GP/lib -lflint -lmpfr -lgmp -lm -o /tmp/gen
/tmp/gen /tmp/apple_cos53_constants_2ulp.h

cat >/tmp/decompose.cpp <<'CPP'
#include <mpfr.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "apple_cos53_constants_2ulp.h"

static uint64_t ob(double x){uint64_t u;memcpy(&u,&x,8);return (u>>63)?~u:(u|UINT64_C(0x8000000000000000));}
static uint64_t ulp(double a,double b){if(a==b)return 0;uint64_t x=ob(a),y=ob(b);return x>y?x-y:y-x;}
static void twos(double a,double b,double &h,double &l){double s=a+b,bv=s-a,av=s-bv,br=b-bv,ar=a-av;h=s;l=ar+br;}
static double eval_base(double rh,double rl,int j,double c0,double c1){double a=(double)j/2048.0,d=(rh-a)+rl,c2=-0.5*c0,c3=-(1.0/6.0)*c1;double p=std::fma(c3,d,c2);p=std::fma(p,d,c1);return std::fma(p,d,c0);}
static double eval_comp(double rh,double rl,int j,double c0,double c1){double a=(double)j/2048.0,d=rh-a,c2=-0.5*c0,c3=-(1.0/6.0)*c1;double p=c3,dp=0;dp=std::fma(dp,d,p);p=std::fma(p,d,c2);dp=std::fma(dp,d,p);p=std::fma(p,d,c1);dp=std::fma(dp,d,p);p=std::fma(p,d,c0);return std::fma(rl,dp,p);}
int main(){
 const double x=-2492.8537705956187, ax=fabs(x); const double invpi=0x1.45f306dc9c883p-2;
 long q=lround(ax*invpi); double s=ax-apple_cos53_pih[q],rh,rl; twos(s,-apple_cos53_pil[q],rh,rl); bool rn=(rh<0)||(rh==0&&rl<0); if(rn){rh=-rh;rl=-rl;}
 double r=rh+rl; int j=(int)llround(r*2048.0); double a=(double)j/2048.0;
 double c0=apple_cos53_c0[j],c1=apple_cos53_c1[j];
 mpfr_t mx,mpi,mq,mres,mabs,mt,mc,ms,mref,mpoly,md; for(mpfr_t* p:{&mx,&mpi,&mq,&mres,&mabs,&mt,&mc,&ms,&mref,&mpoly,&md}) mpfr_init2(*p,256);
 mpfr_set_d(mx,ax,MPFR_RNDN);mpfr_const_pi(mpi,MPFR_RNDN);mpfr_mul_si(mq,mpi,q,MPFR_RNDN);mpfr_sub(mres,mx,mq,MPFR_RNDN);mpfr_abs(mabs,mres,MPFR_RNDN);
 double erh=mpfr_get_d(mabs,MPFR_RNDN); mpfr_sub_d(mt,mabs,erh,MPFR_RNDN); double erl=mpfr_get_d(mt,MPFR_RNDN);
 mpfr_set_d(mt,a,MPFR_RNDN); mpfr_cos(mc,mt,MPFR_RNDN); mpfr_sin(ms,mt,MPFR_RNDN); mpfr_neg(ms,ms,MPFR_RNDN);
 double ec0=mpfr_get_d(mc,MPFR_RNDN), ec1=mpfr_get_d(ms,MPFR_RNDN);
 mpfr_set_d(mx,x,MPFR_RNDN);mpfr_cos(mref,mx,MPFR_RNDN);double ref=mpfr_get_d(mref,MPFR_RNDN);
 auto signedv=[&](double v){return ((q&1)^rn)?-v:v;};
 double b=signedv(eval_base(rh,rl,j,c0,c1));
 double cp=signedv(eval_comp(rh,rl,j,c0,c1));
 double er_curc=signedv(eval_comp(erh,erl,j,c0,c1));
 double cr_exc=signedv(eval_comp(rh,rl,j,ec0,ec1));
 double er_exc=signedv(eval_comp(erh,erl,j,ec0,ec1));
 // MPFR degree-3 Taylor using exact residual and exact anchor cos/sin.
 mpfr_set_d(md,a,MPFR_RNDN); mpfr_sub(md,mabs,md,MPFR_RNDN);
 mpfr_mul(mpoly,md,md,MPFR_RNDN); // d2
 mpfr_mul(mt,mc,mpoly,MPFR_RNDN); mpfr_div_ui(mt,mt,2,MPFR_RNDN); mpfr_sub(mpoly,mc,mt,MPFR_RNDN); // c0-c0*d2/2
 mpfr_mul(mt,ms,md,MPFR_RNDN); mpfr_add(mpoly,mpoly,mt,MPFR_RNDN); // + c1*d
 mpfr_mul(mt,md,md,MPFR_RNDN);mpfr_mul(mt,mt,md,MPFR_RNDN);mpfr_mul(mt,mt,ms,MPFR_RNDN);mpfr_div_ui(mt,mt,6,MPFR_RNDN);mpfr_sub(mpoly,mpoly,mt,MPFR_RNDN); // - c1*d3/6
 if(((q&1)^rn))mpfr_neg(mpoly,mpoly,MPFR_RNDN); double hp3=mpfr_get_d(mpoly,MPFR_RNDN);
 printf("DECOMP x=%.17g q=%ld rn=%d j=%d anchor=%.17g\n",x,q,(int)rn,j,a);
 printf("RED current_rh=%.17g current_rl=%.17g exact_rh=%.17g exact_rl=%.17g r_err=%.17g\n",rh,rl,erh,erl,(rh+rl)-(erh+erl));
 printf("COEFF c0=%.17g exact_round_c0=%.17g c0_abs_err=%.17g c0_ulp_to_correct=%llu\n",c0,ec0,c0-ec0,(unsigned long long)ulp(c0,ec0));
 printf("COEFF c1=%.17g exact_round_c1=%.17g c1_abs_err=%.17g c1_ulp_to_correct=%llu\n",c1,ec1,c1-ec1,(unsigned long long)ulp(c1,ec1));
 printf("VAR baseline=%llu comp=%llu exactResidual_currentCoeff=%llu currentResidual_exactCoeff=%llu exactResidual_exactCoeff=%llu mpfrExactP3=%llu\n",
 (unsigned long long)ulp(b,ref),(unsigned long long)ulp(cp,ref),(unsigned long long)ulp(er_curc,ref),(unsigned long long)ulp(cr_exc,ref),(unsigned long long)ulp(er_exc,ref),(unsigned long long)ulp(hp3,ref));
 printf("OUT ref=%.17g baseline=%.17g comp=%.17g exactR_curC=%.17g curR_exactC=%.17g exactR_exactC=%.17g hp3=%.17g\n",ref,b,cp,er_curc,cr_exc,er_exc,hp3);
 mpfr_clear(md);mpfr_clear(mpoly);mpfr_clear(mref);mpfr_clear(ms);mpfr_clear(mc);mpfr_clear(mt);mpfr_clear(mabs);mpfr_clear(mres);mpfr_clear(mq);mpfr_clear(mpi);mpfr_clear(mx);
}
CPP
clang++ -O3 -std=c++20 -fno-fast-math -ffp-contract=off -I/tmp -I"$MP/include" -I"$GP/include" /tmp/decompose.cpp -L"$MP/lib" -L"$GP/lib" -lmpfr -lgmp -o /tmp/decompose
/tmp/decompose | tee /tmp/badinput_decompose.txt
