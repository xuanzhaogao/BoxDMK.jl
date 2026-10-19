# BoxDMK.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://xuanzhaogao.github.io/BoxDMK.jl/dev/)
[![Build Status](https://github.com/xuanzhaogao/BoxDMK.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/xuanzhaogao/BoxDMK.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/xuanzhaogao/BoxDMK.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/xuanzhaogao/BoxDMK.jl)

Volume potentials in 3D with the box dual-space multilevel kernel-splitting
method (box DMK), in pure Julia.

- **Free space**: `u(x) = ∫ K(x - y) f(y) dy` on an adaptive box tree for the
  Laplace, Yukawa and square-root Laplace kernels, with gradients, Hessians
  and arbitrary targets. A port of the Fortran `bdmk` code by Shidong Jiang:
  same results to ~1e-13, 3–16× faster.
- **Periodic**: the Coulomb potential of a periodic density on any triclinic
  cell, neutral or charged. A port of the C++ periodic box DMK (PeriodicDMK)
  with exact reciprocal-space transforms, faster than the C++ code.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/xuanzhaogao/BoxDMK.jl")
```

Julia 1.10 or newer; start Julia with threads (`julia -t auto`).

## Quick start

```julia
using BoxDMK

# free space
f(x) = exp(-100 * sum(abs2, x .- 0.5))
tree, fvals = build_tree(f, LaplaceKernel(), LegendreBasis(); norder = 8, eps = 1e-6)
res = bdmk(tree, fvals, LaplaceKernel(); eps = 1e-6)
PotentialField(tree, res)([0.3, 0.6, 0.4])

# periodic, on a sheared cell (lattice vectors as columns)
A = [1.0 0.35 0.15; 0 1 -0.2; 0 0 1]
res = bdmk_periodic(g, A; eps = 1e-6)          # g: a density periodic in A
evaluate_periodic(res, [0.3, 0.7, 0.9])
```

See the [documentation](https://xuanzhaogao.github.io/BoxDMK.jl/dev/) for
the options, plans for repeated periodic solves, writing fast (batch, SIMD)
density functions, the methods, and the benchmarks.

## Performance

Genoa node (AMD EPYC 9474F), `eps = 1e-9`, `norder = 16`:

| problem | 1 thread | 48 threads |
|---|---|---|
| free space, 2129 boxes: Fortran / Julia | 16.9 / 4.37 s | 1.87 / 0.25 s |
| periodic hexagonal cell, 5648 boxes: C++ / Julia | 7.43 / 5.45 s | 657 / 361 ms |

Full tables in the [docs](https://xuanzhaogao.github.io/BoxDMK.jl/dev/performance/).

## License

MIT, see [LICENSE](LICENSE).
