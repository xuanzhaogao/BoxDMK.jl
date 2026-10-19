#!/bin/bash
# Periodic solver: Julia vs C++ (PeriodicDMK_private bench_steps) on one genoa
# socket, 1 and 48 threads, with a box-by-box parity check of the potentials.
#   ssh <node> "bash -lc 'bash <repo>/benchmark/run_periodic_node.sh'"
# Env: CPP (PeriodicDMK_private checkout), CELLS, NS (digits), DIST, THREADS, OUT (log dir).
# The benchmark density is not charge-neutral: the timed C++ runs use its
# default (no background correction), the parity dump is a separate C++ run
# with BDMK_RECTGRID_DC_GAUGE=1, the convention of the Julia solver.
set -u
cd "$(dirname "$0")/.." || exit 1
CPP="${CPP:-$(cd .. && pwd)/PeriodicDMK_private}"
CELLS="${CELLS:-cubic sheared hex}"
NS="${NS:-3 6 9}"
DIST="${DIST:-multi}"
THREADS="${THREADS:-1 48}"
OUT="${OUT:-benchmark/results/periodic}"
mkdir -p "$OUT"
JL=/mnt/home/xgao1/.juliaup/bin/julia
BENCH="$CPP/cpp/build-genoa/bench_steps"
for cell in $CELLS; do
  for n in $NS; do
    OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores BDMK_RECTGRID_DC_GAUGE=1 \
      BDMK_DUMP_POT="$OUT/cpp_pot_${cell}_n$n.bin" BDMK_DUMP_TREE="$OUT/cpp_tree_${cell}_n$n.bin" \
      numactl --cpunodebind=0 --membind=0 "$BENCH" "$cell" "$DIST" "$n" > /dev/null 2>&1
    for th in $THREADS; do
      # threads pinned to cores in both codes (OMP_PROC_BIND / JULIA_EXCLUSIVE)
      if [ "$th" = 1 ]; then BIND="numactl --physcpubind=2 --membind=0"; PIN=0; else BIND="numactl --cpunodebind=0 --membind=0"; PIN=1; fi
      echo "== C++ periodic $cell T$th n=$n =="
      OMP_NUM_THREADS=$th OMP_PROC_BIND=close OMP_PLACES=cores $BIND "$BENCH" "$cell" "$DIST" "$n" 2>&1 | grep -E 'STEPS'
      echo "== Julia periodic $cell T$th n=$n =="
      JULIA_EXCLUSIVE=$PIN OPENBLAS_NUM_THREADS=$th $BIND $JL --project -t "$th" benchmark/periodic_vs_cpp.jl "$cell" "$n" "$DIST" 3 \
        "$OUT/cpp_pot_${cell}_n$n.bin" "$OUT/cpp_tree_${cell}_n$n.bin" 2>&1
    done
  done
done
echo PERIODIC_DONE
