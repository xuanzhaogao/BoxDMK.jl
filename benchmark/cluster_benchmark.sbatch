#!/bin/bash
# Native Julia solver vs Fortran reference on one exclusive cluster node.
#
# Submit from the package root, e.g. on a genoa node:
#   env -u SBATCH_RESERVATION -u SLURM_RESERVATION -u SALLOC_RESERVATION \
#     sbatch -p ccm -C "genoa&rocky9" benchmark/cluster_benchmark.sbatch
#
#SBATCH --job-name=boxdmk-bench
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 02:00:00
#SBATCH -o benchmark/results/slurm_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
JULIA=/mnt/home/xgao1/.juliaup/bin/julia
mkdir -p benchmark/results
OUT=benchmark/results/bench_$(hostname -s)_${SLURM_JOB_ID}.md

# Node-local Fortran reference build (-march=native for this CPU).
BUILD=${TMPDIR:-/tmp}/boxdmk_fortran_$SLURM_JOB_ID
cmake -S deps/boxdmk_fortran -B "$BUILD/hot" -DBOXDMK_ENABLE_OPENMP=ON -DBOXDMK_USE_MKL=OFF > /dev/null
cmake --build "$BUILD/hot" --target boxdmk -j 16 > /dev/null
cmake -S deps/boxdmk_fortran -B "$BUILD/cb" -DBOXDMK_ENABLE_OPENMP=OFF -DBOXDMK_USE_MKL=OFF > /dev/null
cmake --build "$BUILD/cb" --target boxdmk -j 16 > /dev/null
mkdir -p "$BUILD/lib"
cp "$BUILD/hot/libboxdmk.so" "$BUILD/lib/libboxdmk_hot.so"
cp "$BUILD/cb/libboxdmk.so" "$BUILD/lib/libboxdmk.so"
export BOXDMK_FORTRAN_LIBDIR="$BUILD/lib"

export OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=spread OMP_PLACES=cores
NCORES=$(nproc --all)
lscpu | grep -E "Model name|^CPU\(s\)|Socket|NUMA node\(s\)" | tee -a "$OUT"

"$JULIA" --project -e "using BoxDMK"
for T in 1 8 16 32 64 96 128; do
    [ "$T" -le "$NCORES" ] || continue
    OMP_NUM_THREADS=$T "$JULIA" --project -t "$T" benchmark/native_vs_fortran.jl "$OUT"
done
echo "results in $OUT"
