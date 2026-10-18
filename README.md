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

PERFORMANCE_PLACEHOLDER

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
- `src/native/`: the box-DMK solver (tables, flat tree and lists, local tables, GEMM and tensor kernels, plane waves, driver)
- `src/solver/`: public `bdmk` entry point, target helpers, `PotentialField`
- `src/fortran/`: optional Fortran reference wrappers
- `benchmark/`: `native_vs_fortran.jl` (speed and accuracy against an analytic solution), cluster/node runners
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
