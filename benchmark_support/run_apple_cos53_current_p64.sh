#!/usr/bin/env bash
set -euo pipefail

[[ "$(uname -m)" == arm64 ]]
[[ "$(sysctl -n machdep.cpu.brand_string)" == *"Apple M1"* ]]

MODE="${1:-}"

if [[ "$MODE" == build ]]; then
  test -f /tmp/apple_cos53_hotloop.cpp
  test -f /tmp/pthreadpool-install/lib/libpthreadpool.a

  python3 - <<'PY'
from pathlib import Path
p=Path('/tmp/apple_cos53_hotloop.cpp')
s=p.read_text()
s=s.replace('#include <Accelerate/Accelerate.h>', '#include <Accelerate/Accelerate.h>\n#include <dispatch/dispatch.h>', 1)
marker='int main(int argc,char**argv) {'
inject=r'''
struct CurrentP64Ctx { const double* x; double* y; size_t n, pieces; };
static void current_p64_piece(void* vp, size_t j) {
    auto* c=(CurrentP64Ctx*)vp;
    size_t a=(c->n*j)/c->pieces;
    size_t b=(c->n*(j+1))/c->pieces;
    opt_cos53_eval(c->x+a,c->y+a,b-a);
}
class CurrentP64Runner {
public:
    void run(const double*x,double*y,size_t n) {
        constexpr size_t pieces=64;
        size_t p=std::min(pieces,n);
        CurrentP64Ctx c{x,y,n,p};
        dispatch_apply_f(p,DISPATCH_APPLY_AUTO,&c,current_p64_piece);
    }
};
'''
assert marker in s
s=s.replace(marker,inject+'\n'+marker,1)
needle='''    if(mode=="apple"){AppleRunner r;return bench_runner("apple",r,n);}\n    return 3;'''
repl='''    if(mode=="apple"){AppleRunner r;return bench_runner("apple",r,n);}\n    if(mode=="p64"){CurrentP64Runner r;return bench_runner("p64",r,n);}\n    return 3;'''
assert needle in s
s=s.replace(needle,repl,1)
Path('/tmp/apple_cos53_current_p64.cpp').write_text(s)
PY

  COMMON="-O3 -DNDEBUG -std=c++20 -mcpu=native -fno-fast-math -ffp-contract=off"
  clang++ $COMMON -I/tmp -I/tmp/pthreadpool-install/include \
    /tmp/apple_cos53_current_p64.cpp /tmp/pthreadpool-install/lib/libpthreadpool.a \
    -framework Accelerate -pthread -o /tmp/apple_cos53_current_p64_bench
  exit 0
fi

if [[ "$MODE" == one ]]; then
  [[ $# -eq 2 ]]
  exec /tmp/apple_cos53_current_p64_bench p64 "$2"
fi

echo "usage: $0 build | one N" >&2
exit 2
