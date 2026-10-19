# BoxDMK.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://xuanzhaogao.github.io/BoxDMK.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://xuanzhaogao.github.io/BoxDMK.jl/dev/)
[![Build Status](https://github.com/xuanzhaogao/BoxDMK.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/xuanzhaogao/BoxDMK.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/xuanzhaogao/BoxDMK.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/xuanzhaogao/BoxDMK.jl)

`BoxDMK.jl` evaluates volume potentials on adaptive box trees in 3D based on the `boxdmk` code by Shidong.

Given a source density `f` and a kernel `K`, it computes

```math
u(x) = \int_{\Omega} K(x, y) f(y)\,dy
```

on tensor-product box nodes, with optional interpolation to user targets and optional derivative recovery.

Everything — tree construction and the volume-potential solve — is pure Julia.
A vendored copy of the Fortran reference is kept only for parity testing and
benchmarking; it is optional and not needed at runtime.

## Current Status

1. `build_tree(...)` builds the adaptive tree in Julia; it produces exactly the same tree as the Fortran builder (same refinement, `zk` forcing, 4-phase level restriction, colleague construction — bit-for-bit across all kernel/basis/dimension configurations).
2. `bdmk(...)` solves with the pure-Julia box-DMK solver (`src/native/`), a port of the Fortran `bdmk` for 3D: Laplace, Yukawa and square-root Laplace kernels; Legendre and Chebyshev bases; potential, gradient and Hessian; arbitrary targets; any number of densities `nd`.
3. `bdmk_periodic(...)` computes the periodic Coulomb potential on any triclinic cell, for neutral or charged densities (pure-Julia port of the C++ periodic box-DMK of PeriodicDMK); see [Periodic Solver](#periodic-solver).

The solver reproduces the Fortran results to ~1e-13 relative (same SOG and
plane-wave tables, same interaction lists, same sparse/dense near-field rules)
while being several times faster; see [Performance](#performance).

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/xuanzhaogao/BoxDMK.jl")
```

Requires Julia `1.10+`.

## Optional: The Fortran Reference

Only needed for `bdmk_fortran` / `build_tree_fortran` (parity tests and
benchmarks). Build it from the package root with

```bash
julia --project deps/build_fortran_ref.jl
```

which produces `deps/usr/lib/libboxdmk{,_hot}.so` (requires `cmake`, `gfortran`
and OpenBLAS). Set `BOXDMK_FORTRAN_LIBDIR` to use libraries built elsewhere.

## Quick Start

```julia
using BoxDMK

f(x) = [exp(-100 * sum((x .- 0.5) .^ 2))]

tree, fvals = build_tree(
    f,
    LaplaceKernel(),
    LegendreBasis();
    ndim = 3,
    norder = 8,
    eps = 1e-6,
    boxlen = 1.0,
    nd = 1,
)

result = bdmk(tree, fvals, LaplaceKernel(); eps = 1e-6)

size(result.pot)
```

The returned `SolverResult` contains:

- `pot`
- `grad` or `nothing`
- `hess` or `nothing`
- `target_pot` or `nothing`
- `target_grad` or `nothing`
- `target_hess` or `nothing`

`bdmk(...)` is the normal entrypoint. Pass `timings = zeros(10)` to receive
the wall time of the solver stages (precomputation, local expansions, proxy
upward pass, plane waves, near field, …).

## Performance: Writing Fast User Functions

The user function `f` is called once per grid point per box during tree construction. At high polynomial orders (e.g., `norder=16`), this can be millions of calls. To get maximum performance:

**Avoid capturing non-const module-level variables.** This is the single most impactful optimization. Julia cannot infer types for non-const globals, causing ~50x overhead per function call:

```julia
# SLOW: captures non-const globals (type-unstable, ~800 ns/call)
sigma = 1e-4
f(x) = [exp(-sum(x.^2) / sigma)]

# FAST: close over a typed struct (~17 ns/call)
struct Params; sigma::Float64; end
p = Params(1e-4)
f(x) = exp(-sum(x.^2) / p.sigma)
```

**Return a scalar instead of `[val]` for `nd=1`.** Avoids heap allocation:

```julia
# OK:   f(x) = [val]      (allocates a Vector each call)
# Better: f(x) = val      (returns Float64, no allocation)
```

**Use the batch API for maximum throughput.** The `f_batch` keyword evaluates all points in a box at once, eliminating per-point function call overhead:

```julia
function my_f_batch!(values::AbstractMatrix, points::AbstractMatrix)
    @inbounds for i in axes(points, 2)
        x1, x2, x3 = points[1,i], points[2,i], points[3,i]
        values[1, i] = exp(-((x1-0.5)^2 + (x2-0.5)^2 + (x3-0.5)^2) * 100)
    end
end

tree, fvals = build_tree(
    LaplaceKernel(), LegendreBasis();
    ndim=3, norder=16, eps=1e-6, boxlen=1.0, nd=1,
    f_batch = my_f_batch!,
)
```

When `f_batch` is provided, the point-by-point `f` is never called and can be
omitted (as above) or passed as `nothing`.

With these practices, Julia tree construction runs within **1.1x** of a pure Fortran tree builder with a native Fortran callback.

## Target Evaluation

```julia
targets = rand(3, 100)

result = bdmk(
    tree,
    fvals,
    LaplaceKernel();
    eps = 1e-6,
    targets = targets,
)

result.target_pot
```

For Julia-facing APIs, the physical domain is `[0, boxlen]^3`.

## Evaluable Field Handle

`PotentialField` attaches the solver output to the tree and gives a callable
handle that can be evaluated anywhere in the domain (leaf-box lookup by tree
descent + tensor-product polynomial interpolation):

```julia
result = bdmk(tree, fvals, LaplaceKernel(); eps = 1e-6, grad = true, hess = true)

field = PotentialField(tree, result)

field([0.3, 0.6, 0.4])          # potential at a point (scalar for nd = 1)
field(rand(3, 1000))            # potential at many points

evaluate_potential(field, x)    # (nd,) vector / (nd, ntarg) matrix
evaluate_gradient(field, x)     # (nd, ndim) / (nd, ndim, ntarg); needs grad = true
evaluate_hessian(field, x)      # (nd, nhess) / (nd, nhess, ntarg); needs hess = true
```

Points on shared box faces are assigned to one of the adjacent leaves; the
interpolants there agree to solver accuracy (machine precision for the
potential, discretization error for derivatives).

The raw Fortran reference uses a box centered at the origin. The wrappers handle that coordinate shift internally for `build_tree_fortran(...)` and `bdmk_fortran(...)`.

## Periodic Solver

`bdmk_periodic` computes the periodic Coulomb potential of a cell-periodic
density on any (triclinic) cell `{A t : t ∈ [0,1)³}`, lattice vectors the columns of `A`:

```math
u(x) = \int_{\text{cell}} \sum_{n \in \mathbb{Z}^3} \frac{f(y)}{|x - y - A n|}\,dy \qquad (k = 0 \text{ mode dropped})
```

A density with nonzero mean is taken with a uniform neutralizing background (the
`k = 0` mode of the whole potential is dropped; for a neutral density this is the
usual periodic potential).

```julia
f(x) = ...                                       # periodic density
A = [1.0 0.35 0.15; 0 1 -0.2; 0 0 1]             # lattice vectors as columns
res = bdmk_periodic(f, A; eps = 1e-6)            # or a tuple (L₁, L₂, L₃), or a number for a cube

evaluate_periodic(res, [0.3, 0.7, 0.9])          # anywhere (wrapped into the cell)
res.pot                                          # (n, n, n, nboxes) on each box's Legendre grid
res.ft, res.istarget, res.grid                   # forest (rotated frame x_rot = Qᵀx, A = QU), boxes computed, geometry
```

`f` must be defined (periodically) slightly outside the cell; `f_batch!(values, points)`
may be passed instead, as for `build_tree`. Options: `norder = 16`, `rc_factor = 0.25`
(level-0 box side `rc = d_min / ⌈1/rc_factor⌉`, `d_min` the smallest distance between
lattice planes), `targets` (3 × N points: only the leaves holding them are computed,
values in `res.target_pot`), `skip_negligible`, `interpolate`, `near32` (below), `timings`.

Repeated solves on a fixed forest (e.g. time stepping) reuse the tree, the tables and
the buffers:

```julia
plan = periodic_plan(f0, A; eps = 1e-6)          # forest adapted to f0
for step in 1:nsteps
    res = periodic_solve!(plan, f_step)          # resamples f on the plan's forest
end
```

Work-saving rules (all on by default):

- **Halo copies / core interpolation**: when the level-0 cubes tile the cell, halo
  subtrees are copies of their lattice images (only the central forest samples `f`);
  otherwise only the boxes meeting the cell sample `f` and the rest interpolate their
  leaf polynomials at periodic images (`interpolate = false` reproduces the C++ forest).
- **Negligible sources** (`skip_negligible`): boxes whose density cannot change the
  potential by `eps` relative (bound `max|f| ∫K_narrow` against the potential scale of
  the strongest box) are left out of the plane-wave and near fields.
- **Single-precision near field** (`near32`, default for `eps ≥ 1e-6`, as in C++): the
  dense same-level near-field contractions run in Float32 (16-wide SIMD).

How it works (`src/native/periodic/`, after the C++ `bdmk_laplace3d_pot_periodic`):

- **Forest**: in the frame where the cell matrix is upper triangular, a grid of
  cubic boxes of side `rc` covering the cell plus a one-box image halo, each box
  refined adaptively (the C++ refinement test, which it reproduces box for box)
  and the whole forest 2:1 balanced. Only boxes that meet the cell are targets.
- **Kernel split**: `1/r` is the u-series sum of Gaussians; the Gaussians that
  decay within `rc` run through the free-space box machinery of `bdmk` on the
  forest (halo boxes carry the image density and act as sources only).
- **Wide Gaussians** are summed in reciprocal space, exactly per axis with
  `e^{-iαξ} = Σ (2l+1)(-i)^l j_l(α) P_l(ξ)`. Source: Legendre moments of one period, from
  the level-0 proxy charges when the cubes tile the cell (orthorhombic, sides
  multiple of `rc`), else from an adaptive tree in fractional coordinates, where
  `e^{-2πi m·t}` is separable. That tree refines with the same test and, in
  addition, wherever the main forest is much finer, so narrow features cannot hide
  between its sample points. Target: the Legendre coefficients of the wide
  potential on each level-0 cube (a separable product for orthorhombic cells, a
  slice-by-slice triangular contraction otherwise, as `k = 2πU⁻ᵀm` is lower
  triangular), injected into the level-0 proxy potentials before the downward
  pass. No FFT is needed.

Accuracy against an Ewald-summed Gaussian mixture is about `eps` relative to
max|u| (1.0–1.25 `eps` in the worst cases) at `eps = 1e-3, 1e-6, 1e-9` on cubic,
orthorhombic (1:1.5:2, 1:√2:√3), sheared, hexagonal and elongated (1:1:4)
triclinic cells, neutral and charged densities (`test/test_periodic.jl`). A lone
spike of width 1.4e-3 on a sheared cell (a known failure of the C++ wide source,
1.9e-5) is computed to 6e-10 at `eps = 1e-9`. On the C++ benchmark problems both
codes build the same forest and the potentials agree to 1.2e-10 at `eps = 1e-9`.

Wall time against the C++ solver (PeriodicDMK `bench_steps <cell> multi`: four
Gaussians down to width 0.007, `norder = 16`; same density callback; genoa node, one
socket, threads pinned; `benchmark/run_periodic_node.sh`,
`benchmark/results/periodic/genoa_cpp_worksaving.log`). The C++ code applies the
same work-saving rules and exact wide transforms (ported back from this package),
so both build the same forest and their potentials agree to 2e-13 at `eps = 1e-9`.

| cell | eps | boxes | 1 thread C++ / Julia | 48 threads C++ / Julia |
|---|---|---|---|---|
| cubic     | 1e-3 | 624  | 0.32 / 0.31 s | 55 / 73 ms |
| cubic     | 1e-6 | 1416 | 1.00 / 0.89 s | 168 / 132 ms |
| cubic     | 1e-9 | 2560 | 3.64 / 3.02 s | 326 / 277 ms |
| sheared   | 1e-3 | 720  | 0.87 / 0.81 s | 90 / 112 ms |
| sheared   | 1e-6 | 1960 | 2.36 / 1.97 s | 311 / 288 ms |
| sheared   | 1e-9 | 4048 | 7.09 / 5.58 s | 537 / 533 ms |
| hexagonal | 1e-3 | 920  | 0.88 / 0.83 s | 103 / 122 ms |
| hexagonal | 1e-6 | 2160 | 2.06 / 1.71 s | 298 / 210 ms |
| hexagonal | 1e-9 | 5648 | 7.41 / 5.89 s | 647 / 589 ms |

(Before these rules, C++ took 6.80 / 17.9 / 18.0 s single-threaded at `eps = 1e-9`.)
Both include the tree build with density sampling (about a third of the
single-thread time); triclinic cells sample the density a second time for the
fractional source tree.

## Kernels And Bases

Supported kernels:

```julia
LaplaceKernel()
YukawaKernel(beta)
SqrtLaplaceKernel()
```

Supported bases:

```julia
LegendreBasis()
ChebyshevBasis()
```

## Fortran Reference APIs

```julia
result = bdmk_fortran(tree, fvals, LaplaceKernel(); eps = 1e-6)   # Fortran solve, same API
ftree = build_tree_fortran(f, LaplaceKernel(), LegendreBasis(); ndim=3, norder=8, eps=1e-6, boxlen=1.0, nd=1)
```

## Performance

Wall time of `bdmk` vs `bdmk_fortran` on the same tree (Laplace kernel,
best of 3, Fortran built with `-O3 -march=native` + OpenBLAS + OpenMP). The
density is `f = -Δu/(4π)` for a sum of three Gaussians in a box of size 1.18,
so the exact potential is `u`; both codes reach the same error against `u`
and agree with each other to ~1e-13 (`benchmark/native_vs_fortran.jl`; full
tables in `benchmark/results/`).

Rusty genoa node (2× AMD EPYC 9474F, 96 cores), seconds:

| norder | eps | outputs | boxes | 1 thread F / Julia | 16 threads F / Julia | 48 threads F / Julia | 96 threads F / Julia |
|---|---|---|---|---|---|---|---|
| 8  | 1e-6 | pot | 14265 | 15.5 / 3.47 (4.5×) | 5.27 / 0.40 (13×) | 4.60 / 0.29 (16×) | 4.66 / 0.30 (15×) |
| 16 | 1e-3 | pot | 849   | 0.59 / 0.16 (3.6×) | 0.13 / 0.020 (6.3×) | 0.12 / 0.016 (7.3×) | 0.12 / 0.025 (4.8×) |
| 16 | 1e-6 | pot | 1225  | 3.05 / 0.88 (3.4×) | 0.52 / 0.10 (5.1×) | 0.45 / 0.066 (6.8×) | 0.46 / 0.082 (5.6×) |
| 16 | 1e-9 | pot | 2129  | 16.9 / 4.37 (3.9×) | 2.29 / 0.41 (5.7×) | 1.87 / 0.25 (7.5×) | 1.87 / 0.25 (7.5×) |
| 12 | 1e-9 | pot | 6553  | 41.6 / 9.87 (4.2×) | 6.80 / 0.84 (8.1×) | 5.57 / 0.45 (13×) | 5.56 / 0.41 (14×) |
| 16 | 1e-6 | pot+grad+hess | 1225 | 28.4 / 4.91 (5.8×) | 2.59 / 0.48 (5.4×) | 1.15 / 0.27 (4.3×) | 0.78 / 0.23 (3.3×) |

Workstation (2× Xeon Gold 6234, 16 cores), seconds:

| norder | eps | outputs | 1 thread F / Julia | 16 threads F / Julia |
|---|---|---|---|---|
| 8  | 1e-6 | pot | 23.7 / 5.88 (4.0×) | 6.01 / 0.64 (9.3×) |
| 16 | 1e-6 | pot | 5.54 / 1.02 (5.4×) | 0.83 / 0.14 (6.1×) |
| 16 | 1e-9 | pot | 25.0 / 5.23 (4.8×) | 3.89 / 0.63 (6.2×) |
| 16 | 1e-6 | pot+grad | 28.1 / 2.70 (10×) | 2.43 / 0.31 (7.7×) |
| 16 | 1e-6 | pot+grad+hess | 47.1 / 5.29 (8.9×) | 4.18 / 0.58 (7.2×) |

Part of the single-thread gain is flushing subnormals to zero (see below):
run with FTZ as well, the Fortran solve of the 2081-box n=16, eps=1e-6
problem takes 7.3 s instead of 13.3 s on the workstation; Julia takes 2.0 s.
Repeated solves reuse the internal buffers; call `BoxDMK.release_memory!()`
to return them to the OS.

## Implementation Notes

The solver (`src/native/`) follows the nine steps of the Fortran `bdmk` and
keeps its numerical choices (SOG nodes, plane-wave quadratures and lengths,
proxy orders, interaction lists, local tables and their sparsity patterns,
the dense-versus-sparse switch of the near field), which is why results agree
with Fortran to rounding. The speed comes from restructuring the work:

- **SIMD GEMM kernels** (`gemm.jl`, via SIMD.jl): every tensor-product stage
  is a small register-blocked matrix product, with no data transposes.
- **Near field**: same-level neighbours are evaluated target-centrically,
  `pot += w Σ_dz Tz(Σ_dy Ty(Σ_dx Tx f_s))`, sharing the y/z stages among
  sources (≈2× fewer flops); sparse pairs use GEMMs on pattern-zeroed tables.
- **Plane waves**: conjugate symmetry halves all complex transforms, which run
  as real GEMMs; leaves form/evaluate plane waves directly from/to grid values;
  the kernel's Fourier transform is folded into the outgoing expansions; the
  translation (M2L) runs over frequency blocks of all boxes, cache-blocked and
  partitioned over threads, with shift factors grouped by offset.
- **Tree passes**: parent↔child transfers share stages among the 8 children
  and fuse the leaf value↔proxy maps; proxy data is stored only for nonleaf
  boxes.
- **Local expansions**: the Taylor correction and the asymptotic expansion of
  the narrowest Gaussians are one linear combination per leaf.
- **Runtime**: subnormal numbers are flushed to zero during the solve (the
  Gaussian tails otherwise hit slow subnormal arithmetic); large buffers come
  from a malloc arena (no GC pressure); dynamic scheduling over boxes.

`scripts/generate_fortran_tables.jl` regenerates `src/native/tables/` from the
Fortran DATA statements.

## Repository Layout

- `src/core/`: types, utilities, basis/tensor primitives
- `src/tree/`: adaptive tree construction
- `src/native/`: the box-DMK solver (tables, flat tree and lists, local tables, GEMM and tensor kernels, plane waves, driver); `src/native/periodic/`: periodic solver (forest, fractional source tree, wide term, driver)
- `src/solver/`: public `bdmk` entry point, target helpers, `PotentialField`
- `src/fortran/`: optional Fortran reference wrappers
- `benchmark/`: `native_vs_fortran.jl` (speed and accuracy against an analytic solution), `periodic_vs_cpp.jl` (periodic solver vs the C++ code), cluster/node runners
- `test/`: regression tests (Fortran parity tests run when the reference is built)
- `deps/boxdmk_fortran/`: vendored Fortran reference sources

## Development Workflow

```bash
julia --project -t auto -e 'using Pkg; Pkg.test()'
OMP_NUM_THREADS=16 julia --project -t 16 benchmark/native_vs_fortran.jl results.md
```

## Limitations

- The native solver is 3D only (the tree builder also supports 1D/2D).
- `eps` must lie in `(1e-12, 1e-2]`, the range covered by the Fortran tables (SOG nodes exist for 3/6/9/12 digits, plane-wave tables for 2–11).

## License

MIT License. See [LICENSE](LICENSE) for details.
