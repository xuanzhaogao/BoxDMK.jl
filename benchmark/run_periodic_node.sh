#!/bin/bash
# Periodic solver: Julia vs C++ (PeriodicDMK_private bench_steps) on one genoa
# socket, 1 and 48 threads, with a box-by-box parity check of the potentials.
#   ssh <node> "bash -lc 'bash <repo>/benchmark/run_periodic_node.sh'"
# Env: CPP (PeriodicDMK_private checkout), NS (digits), DIST, OUT (log dir).
set -u
cd "$(dirname "$0")/.." || exit 1
CPP="${CPP:-$(cd .. && pwd)/PeriodicDMK_private}"
NS="${NS:-3 6 9}"
DIST="${DIST:-multi}"
OUT="${OUT:-benchmark/results/periodic}"
mkdir -p "$OUT"
JL=/mnt/home/xgao1/.juliaup/bin/julia
BENCH="$CPP/cpp/build-genoa/bench_steps"
for n in $NS; do
  for th in 1 48; do
    if [ "$th" = 1 ]; then BIND="numactl --physcpubind=2 --membind=0"; else BIND="numactl --cpunodebind=0 --membind=0"; fi
    echo "== C++ periodic T$th n=$n =="
    OMP_NUM_THREADS=$th OMP_PROC_BIND=close OMP_PLACES=cores BDMK_DUMP_POT="$OUT/cpp_pot_n$n.bin" BDMK_DUMP_TREE="$OUT/cpp_tree_n$n.bin" \
      $BIND "$BENCH" cubic "$DIST" "$n" 2>&1 | grep -E 'STEPS|THROUGHPUT'
    echo "== Julia periodic T$th n=$n =="
    OPENBLAS_NUM_THREADS=$th $BIND $JL --project -t "$th" benchmark/periodic_vs_cpp.jl "$n" "$DIST" 3 \
      "$OUT/cpp_pot_n$n.bin" "$OUT/cpp_tree_n$n.bin" 2>&1
  done
done
echo PERIODIC_DONE
