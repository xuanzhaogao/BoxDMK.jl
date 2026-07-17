# BoxDMK.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ArrogantGao.github.io/BoxDMK.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ArrogantGao.github.io/BoxDMK.jl/dev/)
[![Build Status](https://github.com/ArrogantGao/BoxDMK.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ArrogantGao/BoxDMK.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ArrogantGao/BoxDMK.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ArrogantGao/BoxDMK.jl)

`BoxDMK.jl` evaluates volume potentials on adaptive box trees in 3D based on the `boxdmk` code by Shidong.

Given a source density `f` and a kernel `K`, it computes

```math
u(x) = \int_{\Omega} K(x, y) f(y)\,dy
```

on tensor-product box nodes, with optional interpolation to user targets and optional derivative recovery.

This repository contains both:

- a pure Julia implementation producing exact Fortran-parity tree data,
- a vendored Fortran reference build used for the solve backend and parity testing.

## Current Status

The default workflow is:

1. Build the tree in Julia with `build_tree(...)` — produces exactly the same tree structure as the vendored Fortran builder
2. Solve through the public `bdmk(...)` API — uses the Fortran solve backend for Laplace

The Julia tree builder mirrors the Fortran algorithm exactly: same adaptive refinement with incremental `update_rints`, same `zk`-based forced refinement, same 4-phase level restriction (`flag`/`flag+`/`flag++` with `vol_tree_reorg`), and same `computecoll` colleague construction. Tree data matches Fortran bit-for-bit across all supported kernel/basis/dimension configurations.

Performance: with a type-stable user function, `build_tree(...)` runs within 1.1-1.2x of a pure Fortran tree builder with a native Fortran callback.

The direct Fortran entrypoints remain available for advanced use:

- `bdmk_fortran(tree, fvals, ...)` for parity/debug work
- `build_tree_fortran(...)` when you explicitly want the Fortran tree builder (slower for Julia-defined functions due to Fortran-to-Julia callbacks)

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/ArrogantGao/BoxDMK.jl")
```

Requires Julia `1.10+`.

## Build The Vendored Fortran Libraries

The public solver requires the vendored Fortran shared libraries at package load.

From the package root, build them with:

```bash
julia --project deps/build_fortran_ref.jl
```

This produces repo-local libraries under `deps/usr/lib/`.

If the solve library is missing, `using BoxDMK` will throw an error telling you to run that command.

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

`bdmk(...)` is the normal entrypoint. For Laplace solves, it uses the Fortran solve backend behind the public Julia API.

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

## Fortran And Hybrid APIs

The default public workflow is:

```julia
tree, fvals = build_tree(...)      # Julia tree builder (Fortran-parity)
result = bdmk(tree, fvals, kernel; eps = 1e-6)  # Fortran solve backend
```

The Fortran entrypoints remain available:

```julia
# Explicit Fortran solve (same as bdmk for Laplace)
result = bdmk_fortran(tree, fvals, LaplaceKernel(); eps = 1e-6)

# Fortran tree builder (slower for Julia functions due to callbacks)
ftree = build_tree_fortran(f, LaplaceKernel(), LegendreBasis(); ndim=3, norder=8, eps=1e-6, boxlen=1.0, nd=1)
```

Since `build_tree(...)` now produces identical tree data to the Fortran builder, `build_tree_fortran(...)` is mainly useful for parity testing.

## Repository Layout

The source tree is grouped by responsibility:

- `src/core/`: types, utilities, basis/tensor/kernel primitives
- `src/tree/`: adaptive tree construction, tree data transforms, interaction lists
- `src/solver/`: SOG tables, proxy/passes, plane-wave pipeline, local tables, derivatives, top-level solver
- `src/fortran/`: vendored-library paths, hotpaths, wrappers, debug snapshots
- `benchmark/`: performance and parity drivers
- `test/`: package regression tests
- `deps/boxdmk_fortran/`: vendored Fortran reference sources

Generated build outputs under `deps/usr/` and `deps/boxdmk_fortran/build*` are local artifacts and should not be committed.

## Development Workflow

Useful commands:

Run a focused test:

```bash
julia --project --threads=1 -e 'using BoxDMK; include("test/test_solver.jl")'
```

Run wrapper and hybrid parity checks:

```bash
julia --project --threads=1 -e 'using BoxDMK; include("test/test_fortran_wrapper.jl"); include("test/test_hybrid_parity.jl")'
```

Run the reference hybrid benchmark:

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --project benchmark/hybrid_parity.jl
```

## Limitations

- The vendored Fortran solve library is a required runtime dependency for the public solver.
- Tree construction is verified for exact Fortran parity across Laplace, Yukawa, and SqrtLaplace kernels, with Legendre and Chebyshev bases, in 1D/2D/3D.
- The native Julia solver internals remain in the repo for debugging and development and are still used for kernels outside the validated Laplace-backed path.

## License

MIT License. See [LICENSE](LICENSE) for details.
