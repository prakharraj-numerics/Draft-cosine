#!/usr/bin/env bash
set -euo pipefail

[[ "$(uname -m)" == arm64 ]]
[[ "$(sysctl -n machdep.cpu.brand_string)" == *"Apple M1"* ]]
MODE="${1:-}"

if [[ "$MODE" == build ]]; then
  test -f /tmp/apple_cos53_hotloop.cpp
  test -f /tmp/pthreadpool-install/lib/libpthreadpool.a
  cat >/tmp/apple_cos53_cpu_pocket_attack.cpp <<'CPP'
#define main apple_cos53_hotloop_base_main
#include "/tmp/apple_cos53_hotloop.cpp"
#undef main

#include <os/workgroup.h>
#include <dispatch/dispatch.h>
#include <cerrno>

static qos_class_t pocket_qos(const std::string& s) {
  if (s=="ui") return QOS_CLASS_USER_INTERACTIVE;
  if (s=="user") return QOS_CLASS_USER_INITIATED;
  if (s=="default") return QOS_CLASS_DEFAULT;
  if (s=="utility") return QOS_CLASS_UTILITY;
  std::abort();
}
static inline size_t align8_boundary(size_t n,size_t p,size_t k) {
  if(k==0) return 0; if(k==p) return n;
  return ((n*k)/p)&~size_t(7);
}

struct APContext { const double*x; double*y; size_t n,p; };
static void aligned_pool_piece(void*vp,size_t j){
  auto*c=(APContext*)vp; size_t a=align8_boundary(c->n,c->p,j),b=align8_boundary(c->n,c->p,j+1);
  opt_cos53_eval(c->x+a,c->y+a,b-a);
}
class AlignedPoolRunner {
  pthreadpool_t pool_; size_t p_;
public:
  explicit AlignedPoolRunner(size_t p):pool_(pthreadpool_create(p)),p_(p){if(!pool_)std::abort();}
  ~AlignedPoolRunner(){pthreadpool_destroy(pool_);}
  void run(const double*x,double*y,size_t n){
    if(n<8*p_){opt_cos53_eval(x,y,n);return;}
    APContext c{x,y,n,p_}; pthreadpool_parallelize_1d(pool_,aligned_pool_piece,&c,p_,0);
  }
};

class TunedWGRunner {
  size_t p_; qos_class_t qos_; int spin_;
  os_workgroup_parallel_t wg_; os_workgroup_join_token_s caller_token_{}; bool caller_joined_=false;
  std::vector<std::thread> helpers_;
  std::atomic<uint64_t> generation_{0}; std::atomic<int> completed_{0},ready_{0}; std::atomic<bool> stop_{false};
  const double*x_=nullptr; double*y_=nullptr; size_t n_=0;
  static inline void relax(){__asm__ volatile("yield");}
  size_t boundary(size_t k)const{return align8_boundary(n_,p_,k);}
  void helper_loop(size_t idx){
    pthread_set_qos_class_self_np(qos_,0); os_workgroup_join_token_s tok{};
    bool joined=(os_workgroup_join((os_workgroup_t)wg_,&tok)==0);
    ready_.fetch_add(1,std::memory_order_release); ready_.notify_one();
    uint64_t seen=generation_.load(std::memory_order_relaxed);
    for(;;){
      uint64_t g=generation_.load(std::memory_order_acquire);
      if(g==seen){generation_.wait(seen,std::memory_order_acquire);g=generation_.load(std::memory_order_acquire);} seen=g;
      if(stop_.load(std::memory_order_relaxed)) break;
      size_t a=boundary(idx),b=boundary(idx+1); opt_cos53_eval(x_+a,y_+a,b-a);
      completed_.fetch_add(1,std::memory_order_release); completed_.notify_one();
    }
    if(joined) os_workgroup_leave((os_workgroup_t)wg_,&tok);
  }
public:
  TunedWGRunner(size_t p,qos_class_t q,int spin):p_(p),qos_(q),spin_(spin){
    pthread_set_qos_class_self_np(qos_,0); wg_=os_workgroup_parallel_create("apple-cos53-pocket",nullptr); if(!wg_)std::abort();
    caller_joined_=(os_workgroup_join((os_workgroup_t)wg_,&caller_token_)==0);
    for(size_t i=1;i<p_;++i) helpers_.emplace_back([this,i]{helper_loop(i);});
    while(ready_.load(std::memory_order_acquire)!=(int)p_-1){int old=ready_.load(std::memory_order_acquire);if(old!=(int)p_-1)ready_.wait(old,std::memory_order_acquire);}
  }
  ~TunedWGRunner(){stop_.store(true,std::memory_order_relaxed);generation_.fetch_add(1,std::memory_order_release);generation_.notify_all();for(auto&t:helpers_)t.join();if(caller_joined_)os_workgroup_leave((os_workgroup_t)wg_,&caller_token_);os_workgroup_cancel((os_workgroup_t)wg_);}
  void run(const double*x,double*y,size_t n){
    if(n<8*p_){opt_cos53_eval(x,y,n);return;} x_=x;y_=y;n_=n;completed_.store(0,std::memory_order_relaxed);
    generation_.fetch_add(1,std::memory_order_release);generation_.notify_all();size_t b=boundary(1);opt_cos53_eval(x_,y_,b);
    for(int k=0;k<spin_;++k){if(completed_.load(std::memory_order_acquire)==(int)p_-1)return;relax();}
    int old=completed_.load(std::memory_order_acquire);while(old!=(int)p_-1){completed_.wait(old,std::memory_order_acquire);old=completed_.load(std::memory_order_acquire);}
  }
};

struct PieceCtx { const double*x; double*y; size_t n,p; };
static void piece_fn(void*vp,size_t j){auto*c=(PieceCtx*)vp;size_t a=align8_boundary(c->n,c->p,j),b=align8_boundary(c->n,c->p,j+1);opt_cos53_eval(c->x+a,c->y+a,b-a);}
class PieceRunner { size_t p_; public: explicit PieceRunner(size_t p):p_(p){} void run(const double*x,double*y,size_t n){size_t p=std::min(p_,std::max<size_t>(1,n/8));PieceCtx c{x,y,n,p};dispatch_apply_f(p,DISPATCH_APPLY_AUTO,&c,piece_fn);} };

int main(int argc,char**argv){
  if(argc!=3)return 2;std::string m=argv[1];size_t n=(size_t)std::strtoull(argv[2],nullptr,10);
  if(m=="ap2"){AlignedPoolRunner r(2);return bench_runner(m.c_str(),r,n);} if(m=="ap3"){AlignedPoolRunner r(3);return bench_runner(m.c_str(),r,n);} if(m=="ap4"){AlignedPoolRunner r(4);return bench_runner(m.c_str(),r,n);}
  if(m.rfind("wg",0)==0){
    // format wg{2|3|4}_{ui|user|default|utility}_s{0|8}
    size_t u=m.find('_'),v=m.rfind("_s"); if(u==std::string::npos||v==std::string::npos||v<=u)return 3;
    size_t p=(size_t)std::strtoull(m.substr(2,u-2).c_str(),nullptr,10);std::string q=m.substr(u+1,v-u-1);int spin=std::atoi(m.substr(v+2).c_str());
    TunedWGRunner r(p,pocket_qos(q),spin);return bench_runner(m.c_str(),r,n);
  }
  if(m.size()>1&&m[0]=='p'){size_t p=(size_t)std::strtoull(m.c_str()+1,nullptr,10);PieceRunner r(p);return bench_runner(m.c_str(),r,n);}
  return 4;
}
CPP
  COMMON="-O3 -DNDEBUG -std=c++20 -mcpu=native -fno-fast-math -ffp-contract=fast"
  clang++ $COMMON -I/tmp -I/tmp/pthreadpool-install/include /tmp/apple_cos53_cpu_pocket_attack.cpp \
    /tmp/pthreadpool-install/lib/libpthreadpool.a -framework Accelerate -pthread -o /tmp/apple_cos53_cpu_pocket_attack
  exit 0
fi

if [[ "$MODE" == one ]]; then
  [[ $# -eq 3 ]]
  exec /tmp/apple_cos53_cpu_pocket_attack "$2" "$3"
fi

echo "usage: $0 build | one MODE N" >&2
exit 2
