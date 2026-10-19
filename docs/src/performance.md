# Performance

All timings are wall times, best of 3, with threads pinned to cores
(`JULIA_EXCLUSIVE=1`, `OMP_PROC_BIND=close`). Logs are in `benchmark/results/`.

## Free space against Fortran

`bdmk` against `bdmk_fortran` on the same tree (Laplace kernel; Fortran built
with `-O3 -march=native`, OpenBLAS and OpenMP). The density is
``f = -\Delta u/(4\pi)`` for a sum of three Gaussians in a box of size 1.18, so
the exact potential ``u`` is known. Both codes reach the same error against
``u`` and agree with each other to about ``10^{-13}``
(`benchmark/native_vs_fortran.jl`).

Rusty genoa node (2× AMD EPYC 9474F, 96 cores), seconds, Fortran / Julia:

| norder | eps | outputs | boxes | 1 thread | 16 threads | 48 threads | 96 threads |
|---|---|---|---|---|---|---|---|
| 8  | 1e-6 | pot | 14265 | 15.5 / 3.47 (4.5×) | 5.27 / 0.40 (13×) | 4.60 / 0.29 (16×) | 4.66 / 0.30 (15×) |
| 16 | 1e-3 | pot | 849   | 0.59 / 0.16 (3.6×) | 0.13 / 0.020 (6.3×) | 0.12 / 0.016 (7.3×) | 0.12 / 0.025 (4.8×) |
| 16 | 1e-6 | pot | 1225  | 3.05 / 0.88 (3.4×) | 0.52 / 0.10 (5.1×) | 0.45 / 0.066 (6.8×) | 0.46 / 0.082 (5.6×) |
| 16 | 1e-9 | pot | 2129  | 16.9 / 4.37 (3.9×) | 2.29 / 0.41 (5.7×) | 1.87 / 0.25 (7.5×) | 1.87 / 0.25 (7.5×) |
| 12 | 1e-9 | pot | 6553  | 41.6 / 9.87 (4.2×) | 6.80 / 0.84 (8.1×) | 5.57 / 0.45 (13×) | 5.56 / 0.41 (14×) |
| 16 | 1e-6 | pot+grad+hess | 1225 | 28.4 / 4.91 (5.8×) | 2.59 / 0.48 (5.4×) | 1.15 / 0.27 (4.3×) | 0.78 / 0.23 (3.3×) |

Workstation (2× Xeon Gold 6234, 16 cores), seconds, Fortran / Julia:

| norder | eps | outputs | 1 thread | 16 threads |
|---|---|---|---|---|
| 8  | 1e-6 | pot | 23.7 / 5.88 (4.0×) | 6.01 / 0.64 (9.3×) |
| 16 | 1e-6 | pot | 5.54 / 1.02 (5.4×) | 0.83 / 0.14 (6.1×) |
| 16 | 1e-9 | pot | 25.0 / 5.23 (4.8×) | 3.89 / 0.63 (6.2×) |
| 16 | 1e-6 | pot+grad | 28.1 / 2.70 (10×) | 2.43 / 0.31 (7.7×) |
| 16 | 1e-6 | pot+grad+hess | 47.1 / 5.29 (8.9×) | 4.18 / 0.58 (7.2×) |

Part of the single-thread gain comes from flushing subnormals to zero. With FTZ
as well, the Fortran solve of the 2081-box `norder = 16`, `eps = 1e-6` problem
takes 7.3 s instead of 13.3 s on the workstation; Julia takes 2.0 s.

## Periodic against C++

Against the C++ periodic solver (PeriodicDMK `bench_steps <cell> multi`):
four Gaussians down to width 0.007, `norder = 16`, one socket of a genoa node,
each case on its own node (`benchmark/periodic_cases.disbatch`, logs in
`benchmark/results/periodic/cases/`). Both codes evaluate the same density
point by point, apply the same work-saving rules and exact wide transforms,
and build the same forest.

The Julia time is split into the *sampling* stage, the adaptive refinement
that evaluates the density (its cost depends on how ``f`` is implemented), and
the *rest*, which does not depend on ``f`` (see
[Sampling the density once](@ref)). The columns are:

- *f everywhere*: `sample_once = false`, the C++ rules (the potentials agree
  with C++ to ``2\cdot10^{-13}`` at `eps = 1e-9`);
- *sampling + rest*: the default `sample_once = true` (agreement with C++ to
  about `eps`: ``8\cdot10^{-11}`` at `eps = 1e-9`);
- *SIMD density*: the same with the SIMD.jl batch density of
  `benchmark/periodic_vs_cpp.jl` (`DENSITY=simd`), for which the C++ code has
  no counterpart.

1 thread (s):

| cell | eps | boxes | C++ total | Julia, f everywhere | Julia: sampling + rest | Julia, SIMD density: sampling + rest |
|---|---|---|---|---|---|---|
| cubic | 1e-3 | 624 | 0.33 | 0.32 | 0.13 + 0.04 | 0.02 + 0.04 |
| cubic | 1e-6 | 1416 | 1.00 | 0.89 | 0.21 + 0.32 | 0.03 + 0.32 |
| cubic | 1e-9 | 2560 | 3.63 | 3.02 | 0.60 + 1.99 | 0.09 + 2.06 |
| sheared | 1e-3 | 720 | 0.87 | 0.76 | 0.18 + 0.19 | 0.03 + 0.18 |
| sheared | 1e-6 | 1960 | 2.37 | 1.80 | 0.26 + 0.64 | 0.04 + 0.65 |
| sheared | 1e-9 | 4048 | 7.05 | 5.25 | 0.74 + 3.02 | 0.12 + 3.01 |
| hexagonal | 1e-3 | 920 | 0.87 | 0.72 | 0.19 + 0.20 | 0.03 + 0.17 |
| hexagonal | 1e-6 | 2160 | 2.06 | 1.53 | 0.24 + 0.55 | 0.04 + 0.55 |
| hexagonal | 1e-9 | 5648 | 7.43 | 5.29 | 0.67 + 2.93 | 0.10 + 2.93 |

48 threads (ms):

| cell | eps | boxes | C++ total | Julia, f everywhere | Julia: sampling + rest | Julia, SIMD density: sampling + rest |
|---|---|---|---|---|---|---|
| cubic | 1e-3 | 624 | 57 | 32 | 11 + 25 | 3 + 23 |
| cubic | 1e-6 | 1416 | 171 | 111 | 11 + 59 | 3 + 77 |
| cubic | 1e-9 | 2560 | 330 | 222 | 24 + 152 | 9 + 194 |
| sheared | 1e-3 | 720 | 92 | 79 | 13 + 66 | 5 + 68 |
| sheared | 1e-6 | 1960 | 311 | 169 | 14 + 102 | 4 + 121 |
| sheared | 1e-9 | 4048 | 538 | 374 | 49 + 212 | 8 + 224 |
| hexagonal | 1e-3 | 920 | 105 | 86 | 13 + 66 | 3 + 68 |
| hexagonal | 1e-6 | 2160 | 297 | 140 | 15 + 121 | 8 + 103 |
| hexagonal | 1e-9 | 5648 | 654 | 383 | 39 + 241 | 9 + 259 |

Where the rest goes (hexagonal cell, `eps = 1e-9`): at one thread, the
GEMM-bound stages dominate: plane waves 0.92 s, near field 0.87 s, upward pass
0.38 s. Their kernels run at the FP64 peak. At 48 threads the plane
waves (≈ 75 ms) are bound by memory bandwidth: the expansions of a level are
hundreds of MB.

## Quasi-2D slab

`benchmark/q2d_bench.jl`: a 100 × 100 × 1 slab (periodic in x and y) with a
neutral mixture of 64 Gaussians (exponents 200–2000) near the mid-plane,
evaluated through a cell list. The defaults are used (`rc = 25`, so the slab
boxes are refined to side ≤ 1), `norder = 16`, genoa, one socket. "Points"
counts the nodes of the leaves in the slab. The error is measured at 40 points
near the charges against the 2-periodic Ewald sum
(`benchmark/results/q2d/genoa_100x100x1.log`).

| eps | points | 48 threads: sampling + rest | 1 thread: sampling + rest | rest, 1 / 48 threads | max error / max\|u\| |
|---|---|---|---|---|---|
| 1e-3 | 169 M | 0.55 + 0.96 s | 11.0 + 5.2 s | 32 / 175 Mpoints/s | 3.5e-6 |
| 1e-6 | 213 M | 0.59 + 1.74 s | 12.4 + 19.2 s | 11 / 122 Mpoints/s | 3.0e-9 |
| 1e-9 | 284 M | 0.82 + 4.42 s | 16.1 + 101.6 s | 2.8 / 64 Mpoints/s | 1.8e-10 |

The wide part (in-plane modes and z-kernel matrices) takes 13–28 ms of this at
48 threads. Most leaves are empty slab boxes, which is why the throughput is
higher than in the 3D benchmarks.
