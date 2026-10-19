#!/bin/bash -l
# One periodic benchmark case on an exclusive genoa node (socket 0): the C++
# bench_steps and the Julia solver (sample_once default, SAMPLE_ONCE=0, SIMD
# density) at 48 and 1 threads, with box-by-box parity against the saved C++
# dumps. Writes $OUT/case_<cell>_n<n>.log, ending with CASE_DONE.
#   bash -l benchmark/periodic_case.sh <cell> <n>      (task of periodic_cases.disbatch)
set -u
cell=$1; n=$2
cd "$(dirname "$0")/.." || exit 1
CPP="${CPP:-$(cd .. && pwd)/PeriodicDMK_private}"
OUT="${OUT:-benchmark/results/periodic/cases}"
DUMPS=benchmark/results/periodic
log="$OUT/case_${cell}_n$n.log"
grep -qs CASE_DONE "$log" && exit 0          # idempotent
mkdir -p "$OUT"
JL=/mnt/home/xgao1/.juliaup/bin/julia
{
  echo "# host $(hostname) $(date -Is)"
  for th in 48 1; do
    if [ "$th" = 1 ]; then BIND="numactl --physcpubind=2 --membind=0"; PIN=0; else BIND="numactl --cpunodebind=0 --membind=0"; PIN=1; fi
    echo -n "cpp T$th: "
    OMP_NUM_THREADS=$th OMP_PROC_BIND=close OMP_PLACES=cores $BIND "$CPP/cpp/build-genoa/bench_steps" "$cell" multi "$n" 2>&1 | grep STEPS
    for v in once all once-simd; do
      SO=1; DEN=scalar; [ $v = all ] && SO=0; [ $v = once-simd ] && DEN=simd
      echo -n "$v: "
      SAMPLE_ONCE=$SO DENSITY=$DEN JULIA_EXCLUSIVE=$PIN OPENBLAS_NUM_THREADS=$th $BIND $JL --project -t $th \
        benchmark/periodic_vs_cpp.jl "$cell" "$n" multi 3 "$DUMPS/cpp_pot_${cell}_n$n.bin" "$DUMPS/cpp_tree_${cell}_n$n.bin" 2>&1 |
        grep -E "JULIA|PARITY|ERROR" | tr '\n' ' '
      echo
    done
  done
  echo CASE_DONE
} > "$log" 2>&1
