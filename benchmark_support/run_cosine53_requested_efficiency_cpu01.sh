#!/usr/bin/env bash
set -euo pipefail

ROOT="$PWD"
WORK=/tmp/cos-eff-final
rm -rf "$WORK"
mkdir -p "$WORK/bin" "$WORK/include" "$WORK/sdekit"

cp benchmark_support/bench_sine_53_wide_fast2.c "$WORK/bench_sine_53_wide_fast2.c"
cp benchmark_support/bench_sine_53_wide_intel.c "$WORK/bench_sine_53_wide_intel.c"
cp benchmark_support/sine_53_coeff_source.c "$WORK/sine_53_coeff_source.c"
sed -i 's/^int main(void){int cpu=pin();mkl_set_num_threads_local(1);printf("S53F2_DOMAIN/int s53f2_disabled_main(void){int cpu=pin();mkl_set_num_threads_local(1);printf("S53F2_DOMAIN/' "$WORK/bench_sine_53_wide_fast2.c"
sed -i 's/#define SF_K 12/#define SF_K 8/' "$WORK/sine_53_coeff_source.c"
sed -i 's/#define SF_LUT_N ((1UL << SF_K) + 1UL)/#define SF_LUT_N 403UL/' "$WORK/sine_53_coeff_source.c"
sed -i 's/#define KGRID 4096.0/#define KGRID 256.0/' "$WORK/bench_sine_53_wide_intel.c"
sed -i 's|#define INVK (1.0/4096.0)|#define INVK (1.0/256.0)|' "$WORK/bench_sine_53_wide_intel.c"
cp cosine53_x50_unit_production.c "$WORK/generated_x50.c"
cp cosine53_x67_wide_production.c "$WORK/generated_x67.c"
(
  cd "$WORK"
  python3 "$ROOT/cosine53_apply_formula_conversion.py" \
    --coeff sine_53_coeff_source.c --base bench_sine_53_wide_intel.c --fast2 bench_sine_53_wide_fast2.c \
    --source generated_x50.c --source generated_x67.c
  python3 - <<'PY'
from pathlib import Path
for name in ('generated_x50.c','generated_x67.c'):
    p=Path(name); s=p.read_text(); old='\nint main(void)\n{'
    if s.count(old)!=1: raise SystemExit(f'{name}: production main count={s.count(old)}')
    p.write_text(s.replace(old,'\nint cosine53_embedded_main(void)\n{',1))
PY
)

# Runner topology compatibility only: stage production headers and map the helper
# from logical CPU 2 to the runner's second available logical CPU 1. Repository
# production files are not modified.
cp cosine53_batch_production.hpp "$WORK/include/cosine53_batch_production.hpp"
cp cosine53_custom_2core_1600_frozen.hpp "$WORK/include/cosine53_custom_2core_1600_frozen.hpp"
python3 - <<'PY'
from pathlib import Path
p=Path('/tmp/cos-eff-final/include/cosine53_custom_2core_1600_frozen.hpp')
s=p.read_text()
old='pin_current_thread(2);'
if s.count(old)!=1: raise SystemExit(f'helper pin count={s.count(old)}')
p.write_text(s.replace(old,'pin_current_thread(1);',1))
PY

cp bench_cosine53_compute_efficiency.cpp "$WORK/bench_requested.cpp"
sed -i 's/{100,700,3500,15000,50000,1000000,2000000}/{100,400,1200,3000,5000,20000,50000,200000}/' "$WORK/bench_requested.cpp"
grep -F 'static const size_t sizes[] = {100,400,1200,3000,5000,20000,50000,200000};' "$WORK/bench_requested.cpp"

MKLROOT=/opt/intel/oneapi/mkl/latest
CC=/opt/intel/oneapi/compiler/latest/bin/icx
CXX=/opt/intel/oneapi/compiler/latest/bin/icpx
INC="-DNDEBUG -I$WORK/include -I$FLINT_PREFIX/include -I$MKLROOT/include -I$WORK -I$ROOT"
LIBS="-L$FLINT_PREFIX/lib -Wl,-rpath,$FLINT_PREFIX/lib -lflint -lmpfr -lgmp -L$MKLROOT/lib -Wl,-rpath,$MKLROOT/lib -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl"
COMMON='-O3 -g -xHost -qopt-zmm-usage=high -fp-model=precise -fno-math-errno'
"$CC" $COMMON $INC -DCOSINE53_GENERATED_SOURCE='"/tmp/cos-eff-final/generated_x50.c"' -c cosine53_engine_adapter.c -o "$WORK/bin/unit_adapter.o"
"$CC" $COMMON $INC -DCOSINE53_GENERATED_SOURCE='"/tmp/cos-eff-final/generated_x67.c"' -c cosine53_engine_adapter.c -o "$WORK/bin/wide_adapter.o"
"$CXX" -std=c++17 $COMMON $INC -DCOS53_ENGINE_WIDE=0 "$WORK/bench_requested.cpp" "$WORK/bin/unit_adapter.o" -o "$WORK/bin/unit" $LIBS
"$CXX" -std=c++17 $COMMON $INC -DCOS53_ENGINE_WIDE=1 "$WORK/bench_requested.cpp" "$WORK/bin/wide_adapter.o" -o "$WORK/bin/wide" $LIBS

curl -fsSL -o "$WORK/sdekit/sde.tar.xz" https://downloadmirror.intel.com/924984/sde-external-10.13.1-2026-07-28-lin.tar.xz
echo '94e97d623fec54385686e1e7ba65ebc9941748c05ee451423948334892bf2b50  /tmp/cos-eff-final/sdekit/sde.tar.xz' | sha256sum -c -
tar -xf "$WORK/sdekit/sde.tar.xz" -C "$WORK/sdekit"
SDE="$(find "$WORK/sdekit" -maxdepth 2 -type f -name sde64 -print -quit)"
test -x "$SDE"

{
  echo "CPU=$(lscpu | awk -F: '/Model name/{gsub(/^[ \t]+/,"",$2);print $2;exit}')"
  echo "HEAD=$(git rev-parse HEAD)"
  echo "ALLOWED=$(grep Cpus_allowed_list /proc/self/status | awk '{print $2}')"
  echo "CORE0=$(cat /sys/devices/system/cpu/cpu0/topology/core_id 2>/dev/null || echo '?')"
  echo "CORE1=$(cat /sys/devices/system/cpu/cpu1/topology/core_id 2>/dev/null || echo '?')"
  echo "SIZES=100,400,1200,3000,5000,20000,50000,200000"
  echo "OURS=COS53_production_logic_runner_pin_map_CPU2_to_CPU1"
  echo "INTEL=oneMKL_vmdCos_VML_HA_sequential_CPU0"
  echo "NOTE=repository_production_files_untouched;only_staged_scheduler_CPU_ID_remapped_for_current_2CPU_runner"
} > "$WORK/meta.txt"

export LD_LIBRARY_PATH="$FLINT_PREFIX/lib:$MKLROOT/lib:${LD_LIBRARY_PATH:-}"
: > "$WORK/native.txt"
for stack in prod intel; do
  for engine in unit wide; do
    exe="$WORK/bin/$engine"
    if [[ "$stack" == prod ]]; then cpus=0,1; else cpus=0; fi
    taskset -c "$cpus" "$exe" "native-$stack" | tee -a "$WORK/native.txt"
  done
done

for n in 100 400 1200 3000 5000 20000 50000 200000; do
  for engine in unit wide; do
    exe="$WORK/bin/$engine"
    for stack in prod intel noop; do
      d="$WORK/sde-${engine}-${stack}-${n}"; mkdir -p "$d"
      if [[ "$stack" == prod ]]; then cpus=0,1; else cpus=0; fi
      (cd "$d"; taskset -c "$cpus" "$SDE" -mix -iform -global_region \
        -control 'start:address:cos53_profile_start,stop:address:cos53_profile_stop' \
        -omix mix.txt -- "$exe" "sde-$stack" "$n" > program.txt)
    done
  done
done

python3 - <<'PY' | tee "$WORK/results.txt"
import pathlib,re
root=pathlib.Path('/tmp/cos-eff-final')
print(root.joinpath('meta.txt').read_text(),end='')
sizes=[100,400,1200,3000,5000,20000,50000,200000]
cases={'unit':2,'wide':4}
p=re.compile(r'NATIVE engine=(unit|wide) stack=(prod|intel) n=(\d+) cases=(\d+) reps=(\d+) wall_ns_el=([0-9.]+) cpu_ns_el=([0-9.]+) effective_cores=([0-9.]+) maxrss_kib=(\d+) maxulp_vs_intel=(\d+)')
native={}
for line in root.joinpath('native.txt').read_text().splitlines():
    m=p.search(line)
    if m: native[(m.group(1),m.group(2),int(m.group(3)))]=dict(wall=float(m.group(6)),cpu=float(m.group(7)),rss=int(m.group(9)),ulp=int(m.group(10)))
def mix(path):
    text=path.read_text(errors='replace')
    blocks=re.findall(r'# \$global-dynamic-counts(.*?)# END_GLOBAL_DYNAMIC_STATS',text,re.S)
    sec=blocks[-1] if blocks else text
    out={}
    for k,v in re.findall(r'^\s*(\S+)\s+([0-9]+)\s*$',sec,re.M): out[k]=int(v)
    if out.get('*total',0)<=0:
        ts=[int(x) for x in re.findall(r'^\*total\s+([0-9]+)\s*$',text,re.M)]
        if ts: out['*total']=ts[-1]
    return out
def delta(a,b,k): return max(0,a.get(k,0)-b.get(k,0))
def mem_bytes(a,b):
    total=0
    for k in set(a)|set(b):
        m=re.match(r'^\*mem-(?:read|write)-(\d+)$',k)
        if m: total += int(m.group(1))*max(0,a.get(k,0)-b.get(k,0))
    return total
print('HEADER n ours_wall_ns_el intel_wall_ns_el speed_eff_ratio ours_cpu_ns_el intel_cpu_ns_el cpu_eff_ratio ours_effective_cores intel_effective_cores ours_maxrss_kib intel_maxrss_kib rss_memory_eff_ratio ours_instr_el intel_instr_el instruction_eff_ratio ours_logical_b_el intel_logical_b_el logical_b_eff_ratio maxulp_vs_intel')
for n in sizes:
    agg={s:{'wall':0.0,'cpu':0.0,'rss':0,'ulp':0,'instr':0,'bytes':0} for s in ('prod','intel')}
    for eng,w in cases.items():
        for s in ('prod','intel'):
            z=native[(eng,s,n)]
            agg[s]['wall'] += z['wall']*w/6.0
            agg[s]['cpu'] += z['cpu']*w/6.0
            agg[s]['rss'] = max(agg[s]['rss'],z['rss'])
            agg[s]['ulp'] = max(agg[s]['ulp'],z['ulp'])
        raw={s:mix(root/f'sde-{eng}-{s}-{n}'/'mix.txt') for s in ('prod','intel','noop')}
        for s in ('prod','intel'):
            a,b=raw[s],raw['noop']
            agg[s]['instr'] += delta(a,b,'*total')
            agg[s]['bytes'] += mem_bytes(a,b)
    o,i=agg['prod'],agg['intel']; denom=6.0*n
    oi=o['instr']/denom; ii=i['instr']/denom; ob=o['bytes']/denom; ib=i['bytes']/denom
    oc=o['cpu']/o['wall']; ic=i['cpu']/i['wall']
    print(f'FINAL n={n} ours_wall_ns_el={o["wall"]:.9f} intel_wall_ns_el={i["wall"]:.9f} speed_eff_ratio={i["wall"]/o["wall"]:.6f} '
          f'ours_cpu_ns_el={o["cpu"]:.9f} intel_cpu_ns_el={i["cpu"]:.9f} cpu_eff_ratio={i["cpu"]/o["cpu"]:.6f} '
          f'ours_effective_cores={oc:.6f} intel_effective_cores={ic:.6f} '
          f'ours_maxrss_kib={o["rss"]} intel_maxrss_kib={i["rss"]} rss_memory_eff_ratio={i["rss"]/o["rss"]:.6f} '
          f'ours_instr_el={oi:.6f} intel_instr_el={ii:.6f} instruction_eff_ratio={ii/oi:.6f} '
          f'ours_logical_b_el={ob:.6f} intel_logical_b_el={ib:.6f} logical_b_eff_ratio={ib/ob:.6f} maxulp_vs_intel={o["ulp"]}')
PY
cat "$WORK/native.txt" >> "$WORK/results.txt"
