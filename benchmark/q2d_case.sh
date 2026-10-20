#!/bin/bash -l
# One quasi-2D benchmark case on an exclusive genoa node (socket 0):
#   bash -l benchmark/q2d_case.sh bench <digits> <threads> [rc]   (benchmark/q2d_bench.jl)
#   bash -l benchmark/q2d_case.sh widecmp <digits>                 (wide transforms, q2d vs 3D periodic)
#   bash -l benchmark/q2d_case.sh test                             (full test suite)
# Writes $OUT/<name>.log ending with CASE_DONE (skipped when present).
set -u
cd "$(dirname "$0")/.." || exit 1
OUT="${OUT:-benchmark/results/q2d/cases}"
JL=/mnt/home/xgao1/.juliaup/bin/julia
mkdir -p "$OUT"
kind=$1; shift
name="${kind}_$(IFS=_; echo "$*")"
log="$OUT/$name.log"
grep -qs CASE_DONE "$log" && exit 0
{
  echo "# host $(hostname) $(date -Is) $kind $*"
  case $kind in
    bench)
      nd=$1; th=$2; rc=${3:-}
      if [ "$th" = 1 ]; then BIND="numactl --physcpubind=2 --membind=0"; PIN=0; else BIND="numactl --cpunodebind=0 --membind=0"; PIN=1; fi
      JULIA_EXCLUSIVE=$PIN $BIND $JL --project -t "$th" benchmark/q2d_bench.jl "$nd" 64 2 $rc 2>&1 | grep -E "Q2D|ERROR" ;;
    widecmp)
      JULIA_EXCLUSIVE=1 numactl --cpunodebind=0 --membind=0 $JL --project -t 48 benchmark/q2d_widecmp.jl "$1" 2>&1 | grep -E "eps=|ERROR" ;;
    dry)
      echo dry run ;;
    test)
      $JL --project -t 16 -e 'using Pkg; Pkg.test()' 2>&1 | grep -E "Test Summary|tests passed|Fail|Error" -A3 ;;
  esac
  echo CASE_DONE
} > "$log" 2>&1
