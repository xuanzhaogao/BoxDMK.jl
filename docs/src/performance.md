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
four Gaussians down to width 0.007, `norder = 16`, genoa node, one socket. Both
codes evaluate the same density point by point, apply the same work-saving
rules and exact wide transforms, build the same forest, and agree to
``2\cdot10^{-13}`` at `eps = 1e-9`. The last column of each group uses the
SIMD.jl batch density of `benchmark/periodic_vs_cpp.jl` (`DENSITY=simd`)
instead; the C++ code has no counterpart. Log:
`benchmark/results/periodic/genoa_simd.log`.

| cell | eps | boxes | 1 thread (s): C++ / Julia / Julia + SIMD density | 48 threads (ms): C++ / Julia / Julia + SIMD density |
|---|---|---|---|---|
| cubic | 1e-3 | 624 | 0.32 / 0.31 / 0.08 | 57 / 51 / 31 |
| cubic | 1e-6 | 1416 | 1.00 / 0.87 / 0.40 | 169 / 107 / 75 |
| cubic | 1e-9 | 2560 | 3.61 / 3.03 / 2.14 | 328 / 256 / 202 |
| sheared | 1e-3 | 720 | 0.88 / 0.76 / 0.20 | 91 / 91 / 45 |
| sheared | 1e-6 | 1960 | 2.36 / 1.80 / 0.75 | 311 / 178 / 112 |
| sheared | 1e-9 | 4048 | 7.06 / 5.23 / 3.20 | 542 / 366 / 241 |
| hexagonal | 1e-3 | 920 | 0.88 / 0.72 / 0.21 | 105 / 60 / 62 |
| hexagonal | 1e-6 | 2160 | 2.06 / 1.53 / 0.67 | 299 / 134 / 100 |
| hexagonal | 1e-9 | 5648 | 7.43 / 5.45 / 3.19 | 657 / 361 / 262 |

Where the time goes (hexagonal cell, `eps = 1e-9`, SIMD density): at one
thread, the GEMM-bound stages dominate: plane waves 0.92 s, near field 0.87 s,
upward pass 0.38 s. Their kernels run at the FP64 peak. At 48 threads the plane
waves (≈ 75 ms) are bound by memory bandwidth: the expansions of a level are
hundreds of MB.
