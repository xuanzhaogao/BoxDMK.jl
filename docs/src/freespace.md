# Free-space potentials

Given a density ``f`` on the box ``[0, L]^3`` and a kernel ``K``, `bdmk`
computes

```math
u(x) = \int_{[0, L]^3} K(x - y)\, f(y)\,dy
```

on the tensor-product nodes of an adaptive box tree, optionally with its
gradient and Hessian and at arbitrary target points.

## Building the tree

```julia
tree, fvals = build_tree(f, LaplaceKernel(), LegendreBasis();
                         ndim = 3, norder = 8, eps = 1e-6, boxlen = 1.0, nd = 1)
```

`build_tree` refines boxes until the polynomial tail of `f` on each box is
below `eps` (relative to the norm of `f`), then enforces the 2:1 level
restriction. It produces the same tree as the Fortran builder. `fvals` holds the
samples, `(nd, norder^3, nboxes)`; `nd > 1` solves several densities at once
(`f` then returns a vector of length `nd`).

Kernels: [`LaplaceKernel`](@ref) (``1/r``), [`YukawaKernel`](@ref)`(β)`
(``e^{-βr}/r``), [`SqrtLaplaceKernel`](@ref) (``1/r^2``). Bases:
[`LegendreBasis`](@ref), [`ChebyshevBasis`](@ref).

## Solving

```julia
res = bdmk(tree, fvals, LaplaceKernel(); eps = 1e-6, grad = true, hess = true,
           targets = rand(3, 100), timings = zeros(10))
```

The result holds `pot` `(nd, norder^3, nboxes)`, `grad` `(nd, 3, norder^3, nboxes)`
and `hess` `(nd, 6, norder^3, nboxes)` (or `nothing`), and `target_pot`,
`target_grad`, `target_hess` at the targets. `timings` receives the wall time
of the solver stages (precomputation, local expansions, upward pass, plane
waves, near field, …). Repeated solves reuse the internal buffers;
`BoxDMK.release_memory!()` returns them to the OS.

`eps` must lie in `(1e-12, 1e-2]` (the range of the Fortran tables).

## Evaluating the field anywhere

[`PotentialField`](@ref) attaches the solution to the tree and interpolates
it at any point of the domain:

```julia
field = PotentialField(tree, res)

field([0.3, 0.6, 0.4])          # potential at a point (a number for nd = 1)
field(rand(3, 1000))            # at many points
evaluate_gradient(field, x)     # needs grad = true
evaluate_hessian(field, x)      # needs hess = true
```

## Fast density functions

`f` is called once per node per box: millions of calls at `norder = 16`.

- Do not read non-constant globals in `f` (type-unstable, often 50× slower);
  close over a typed struct or constants instead.
- Return a number rather than `[value]` when `nd = 1`.
- For the best throughput pass `f_batch!(values, points)`, which fills
  `values (nd, npts)` at `points (3, npts)` for a whole box at once. It can be
  vectorized, for example with SIMD.jl (see [Periodic potentials](@ref) for an
  example), and `f` may then be omitted:

```julia
function g!(values, points)
    @inbounds for i in axes(points, 2)
        r2 = (points[1, i] - 0.5)^2 + (points[2, i] - 0.5)^2 + (points[3, i] - 0.5)^2
        values[1, i] = exp(-100r2)
    end
end
tree, fvals = build_tree(LaplaceKernel(), LegendreBasis(); norder = 16, eps = 1e-6, f_batch = g!)
```

## Fortran reference

The vendored Fortran code is optional and only used for parity tests and
benchmarks. Build it with `julia --project deps/build_fortran_ref.jl` (needs
`cmake`, `gfortran` and OpenBLAS); then [`bdmk_fortran`](@ref) and
[`build_tree_fortran`](@ref) take the same arguments as `bdmk` and
`build_tree`.
