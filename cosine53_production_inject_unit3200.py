from pathlib import Path
import sys
p=Path(sys.argv[1]);s=p.read_text();mainpos=s.index('\nint main(void)')
helper=r'''
static void x50u_make(double*x){const int n=3200;for(int j=0;j<n;j++){unsigned q=(unsigned)(((unsigned long long)j*1031ULL+17ULL)%3200ULL);double u=((double)q+0.5)/(double)n;x[j]=(j&1)?-u:u;}}
static void x50u_run(const s53w_kernel*k){const int n=3200,rr=32;double*x=al64(n*sizeof(double)),*yo=al64(n*sizeof(double)),*yi=al64(n*sizeof(double));x50u_make(x);int acc=verify_v8("abs_0_to_1",k,x,n);volatile double sink=0;for(int r=0;r<4;r++){octant_eval_v8(k,x,yo,n);vmdSin(n,x,yi,VML_HA);}double ot[7],it[7],calls=(double)n*rr;for(int t=0;t<7;t++){uint64_t a,b,t0;if(t&1){t0=now_ns();for(int r=0;r<rr;r++)vmdSin(n,x,yi,VML_HA);b=now_ns()-t0;t0=now_ns();for(int r=0;r<rr;r++)octant_eval_v8(k,x,yo,n);a=now_ns()-t0;}else{t0=now_ns();for(int r=0;r<rr;r++)octant_eval_v8(k,x,yo,n);a=now_ns()-t0;t0=now_ns();for(int r=0;r<rr;r++)vmdSin(n,x,yi,VML_HA);b=now_ns()-t0;}ot[t]=(double)a/calls;it[t]=(double)b/calls;sink+=yo[n-1]+yi[n-1];}qsort(ot,7,sizeof(double),cmpd);qsort(it,7,sizeof(double),cmpd);printf("X68_RESULT tag=abs_0_to_1 engine=X50 cases=3200 pos=1600 neg=1600 ours_ns=%.6f intel_ns=%.6f ours_over_intel=%.3fx intel_over_ours=%.3fx acc=%d sink=%.17g\n",ot[3],it[3],ot[3]/it[3],it[3]/ot[3],acc,(double)sink);free(yi);free(yo);free(x);}
'''
s=s[:mainpos]+helper+s[mainpos:];mainpos=s.index('\nint main(void)')
s=s[:mainpos]+r'''
int main(void){int cpu=pin();mkl_set_num_threads_local(1);printf("X68_UNIT_MAIN cpu=%d engine=X50 frozen=1 cases=3200 pos=1600 neg=1600 reference=Arb256\n",cpu);s53w_kernel*k=kernel_create(2);if(!k)return 5;x50u_run(k);kernel_destroy(k);flint_cleanup_master();return 0;}
'''
p.write_text(s)
