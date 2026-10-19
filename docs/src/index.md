```@meta
CurrentModule = BoxDMK
```

# BoxDMK.jl

BoxDMK.jl computes volume potentials in 3D with the box dual-space multilevel
kernel-splitting method (box DMK), in pure Julia:

- **Free space**: ``u(x) = \int_\Omega K(x - y) f(y)\,dy`` on an adaptive box
  tree, for the Laplace, Yukawa and square-root Laplace kernels, with gradients,
  Hessians and evaluation at arbitrary targets. A port of the Fortran `bdmk`
  code that reproduces its results to about ``10^{-13}`` and runs 3–16 times
  faster.
- **Periodic**: the Coulomb potential of a cell-periodic density on any
  triclinic cell, neutral or charged (uniform neutralizing background). A port
  of the C++ periodic box DMK (PeriodicDMK), with exact reciprocal-space
  transforms of the long-range part, faster than the C++ code.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/xuanzhaogao/BoxDMK.jl")
```

Julia 1.10 or newer. Start Julia with threads (`julia -t auto`): every stage
of both solvers runs in parallel.

## Quick start

Free-space potential of a Gaussian:

```julia
using BoxDMK

f(x) = exp(-100 * sum(abs2, x .- 0.5))

tree, fvals = build_tree(f, LaplaceKernel(), LegendreBasis(); norder = 8, eps = 1e-6)
res = bdmk(tree, fvals, LaplaceKernel(); eps = 1e-6)

field = PotentialField(tree, res)
field([0.3, 0.6, 0.4])                 # potential at a point
```

Periodic potential on a sheared cell:

```julia
A = [1.0 0.35 0.15; 0 1 -0.2; 0 0 1]   # lattice vectors (columns)
g(x) = ...                             # density, periodic in A
res = bdmk_periodic(g, A; eps = 1e-6)
evaluate_periodic(res, [0.3, 0.7, 0.9])
```

## Contents

- [Free-space potentials](@ref): trees, solver options, field evaluation, fast
  density functions.
- [Periodic potentials](@ref): cells, plans for repeated solves, options, the
  method and its accuracy.
- [Performance](@ref): benchmarks against the Fortran and C++ codes.
- [Implementation notes](@ref): how the solvers are organized and where the
  speed comes from.
- [API reference](@ref).
