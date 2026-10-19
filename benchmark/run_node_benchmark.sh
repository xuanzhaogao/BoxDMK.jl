#!/bin/bash
# Run the native-vs-Fortran benchmark on the current node (interactive or batch).
#   bash benchmark/run_node_benchmark.sh [thread counts...]
set -euo pipefail
cd "$(dirname "$0")/.."
JULIA=/mnt/home/xgao1/.juliaup/bin/julia
TAG=${SLURM_JOB_ID:-$$}
OUT=benchmark/results/bench_$(hostname -s)_${TAG}.md
BUILD=${TMPDIR:-/tmp}/boxdmk_fortran_$TAG
cmake -S deps/boxdmk_fortran -B "$BUILD/hot" -DBOXDMK_ENABLE_OPENMP=ON -DBOXDMK_USE_MKL=OFF > /dev/null
cmake --build "$BUILD/hot" --target boxdmk -j 16 > /dev/null 2>&1
cmake -S deps/boxdmk_fortran -B "$BUILD/cb" -DBOXDMK_ENABLE_OPENMP=OFF -DBOXDMK_USE_MKL=OFF > /dev/null
cmake --build "$BUILD/cb" --target boxdmk -j 16 > /dev/null 2>&1
mkdir -p "$BUILD/lib"
cp "$BUILD/hot/libboxdmk.so" "$BUILD/lib/libboxdmk_hot.so"
cp "$BUILD/cb/libboxdmk.so" "$BUILD/lib/libboxdmk.so"
export BOXDMK_FORTRAN_LIBDIR="$BUILD/lib"
export OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=spread OMP_PLACES=cores
lscpu | grep -E "Model name|^CPU\(s\)|Socket|NUMA node\(s\)" | tee -a "$OUT"
"$JULIA" --project -e 'using BoxDMK'
THREADS=${*:-"1 8 16 32 48 96"}
for T in $THREADS; do
    OMP_NUM_THREADS=$T "$JULIA" --project -t "$T" benchmark/native_vs_fortran.jl "$OUT"
done
echo "results in $OUT"
