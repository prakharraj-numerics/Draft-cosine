#!/usr/bin/env bash
set -euo pipefail

[[ "$(uname -m)" == arm64 ]]
[[ "$(sysctl -n machdep.cpu.brand_string)" == *"Apple M1"* ]]

MODE="${1:-}"

if [[ "$MODE" == build ]]; then
  test -f /tmp/apple_cos53_hotloop.cpp
  test -x /tmp/apple_cos53_canonical_optimized_bench
  test -x /tmp/apple_cos53_hotloop_apple_specific_bench

  python3 - <<'PY'
from pathlib import Path
p=Path('/tmp/apple_cos53_hotloop.cpp')
s=p.read_text()
s=s.replace('#include <Accelerate/Accelerate.h>','#include <Accelerate/Accelerate.h>\n#include <dispatch/dispatch.h>',1)
marker='int main(int argc,char**argv) {'
inject=r'''
struct PSweepCtx { const double* x; double* y; size_t n, pieces; };
static void psweep_piece(void* vp, size_t j) {
    auto* c=(PSweepCtx*)vp;
    size_t a=(c->n*j)/c->pieces;
    size_t b=(c->n*(j+1))/c->pieces;
    opt_cos53_eval(c->x+a,c->y+a,b-a);
}
class PSweepDispatch {
    size_t pieces_;
public:
    explicit PSweepDispatch(size_t p):pieces_(p){}
    void run(const double*x,double*y,size_t n) {
        size_t p=std::min(pieces_,n);
        PSweepCtx c{x,y,n,p};
        dispatch_apply_f(p,DISPATCH_APPLY_AUTO,&c,psweep_piece);
    }
};
'''
assert marker in s
s=s.replace(marker,inject+'\n'+marker,1)
needle='''    if(mode=="apple"){AppleRunner r;return bench_runner("apple",r,n);}\n    return 3;'''
repl='''    if(mode=="apple"){AppleRunner r;return bench_runner("apple",r,n);}\n    if(mode.size()>1 && mode[0]=='p') {\n        size_t pieces=(size_t)std::strtoull(mode.c_str()+1,nullptr,10);\n        if(pieces<1) return 4;\n        PSweepDispatch r(pieces);\n        return bench_runner(mode.c_str(),r,n);\n    }\n    return 3;'''
assert needle in s
s=s.replace(needle,repl,1)
Path('/tmp/apple_cos53_psweep.cpp').write_text(s)
PY

  COMMON="-O3 -DNDEBUG -std=c++20 -mcpu=native -fno-fast-math -ffp-contract=off"
  clang++ $COMMON -I/tmp -I/tmp/pthreadpool-install/include \
    /tmp/apple_cos53_psweep.cpp /tmp/pthreadpool-install/lib/libpthreadpool.a \
    -framework Accelerate -pthread -o /tmp/apple_cos53_psweep_bench
  exit 0
fi

if [[ "$MODE" == p ]]; then
  [[ $# -eq 3 ]]
  exec /tmp/apple_cos53_psweep_bench "p$2" "$3"
fi

if [[ "$MODE" == baseline ]]; then
  [[ $# -eq 2 ]]
  n="$2"
  case "$n" in
    40000) exec bash benchmark_support/run_apple_cos53_hotloop_eff.sh apple_one wg3_user "$n" ;;
    40001) exec /tmp/apple_cos53_canonical_optimized_bench adapt2 "$n" ;;
    50000) exec bash benchmark_support/run_apple_cos53_hotloop_eff.sh apple_one wg2_ui "$n" ;;
    60000|80000|100000) exec /tmp/apple_cos53_canonical_optimized_bench auto "$n" ;;
    *) echo "unsupported baseline size $n" >&2; exit 3 ;;
  esac
fi

if [[ "$MODE" == apple ]]; then
  [[ $# -eq 2 ]]
  exec /tmp/apple_cos53_psweep_bench apple "$2"
fi

echo "usage: $0 build | p PIECES N | baseline N | apple N" >&2
exit 2
