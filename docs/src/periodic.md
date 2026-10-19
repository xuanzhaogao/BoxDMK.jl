# Periodic potentials

[`bdmk_periodic`](@ref) computes the periodic Coulomb potential of a density
``f`` that is periodic on the cell ``\{A t : t \in [0,1)^3\}`` (lattice vectors
the columns of ``A``, any triclinic cell):

```math
u(x) = \int_{\text{cell}} \sum_{n \in \mathbb{Z}^3} \frac{f(y)}{|x - y - A n|}\,dy
\qquad (k = 0 \text{ mode dropped}).
```

A density with nonzero mean is taken with a uniform neutralizing background
(the ``k = 0`` mode of the whole potential is dropped); for a neutral density
this is the usual periodic potential.

## Usage

```julia
A = [1.0 0.35 0.15; 0 1 -0.2; 0 0 1]       # or a tuple (L₁, L₂, L₃), or a number for a cube
res = bdmk_periodic(f, A; eps = 1e-6)

evaluate_periodic(res, [0.3, 0.7, 0.9])    # anywhere (wrapped into the cell)
res.pot                                    # (n, n, n, nboxes) on each box's Legendre grid
```

`f(x)` must be defined (periodically) slightly outside the cell. The forest
lives in the rotated frame ``x_\text{rot} = Q^T x`` of ``A = QU`` (``U`` upper
triangular); `res.ft` is the forest, `res.istarget` marks the boxes whose
potential is computed (those meeting the cell) and `res.grid` holds the
geometry.

Options (see the docstring for all): `norder = 16`; `rc_factor = 0.25` (the
level-0 boxes have side ``r_c = d_\min / \lceil 1/\text{rc\_factor} \rceil``,
``d_\min`` the smallest distance between lattice planes); `targets` (3 × N
points: only the leaves holding them are computed, values in
`res.target_pot`); `f_batch`; `timings`.

## Repeated solves

A plan keeps the forest, the tables and the buffers; each solve resamples the
new density on the plan's forest:

```julia
plan = periodic_plan(f0, A; eps = 1e-6)    # forest adapted to f0
for step in 1:nsteps
    res = periodic_solve!(plan, f_step)
end
```

## Work-saving rules

All are on by default and change the result by less than `eps`:

- **Halo copies, core interpolation.** When the level-0 cubes tile the cell,
  the subtrees of the image halo are copies of their lattice images and only
  the central forest samples ``f``. Otherwise only the boxes that meet the cell
  sample ``f``, and the others interpolate the leaf polynomials at the periodic
  images of their nodes (`interpolate = false` gives the forest of the C++
  builder).
- **Negligible sources** (`skip_negligible`). Boxes whose density cannot change
  the potential by `eps` relative are left out of the real-space part.
- **Single-precision near field** (`near32`, default for `eps ≥ 1e-6`). The
  dense same-level near field runs in Float32.

## Sampling the density once

With `sample_once = true` (the default), ``f`` is evaluated only by the
adaptive refinement of the boxes that meet the cell. That is the one stage
that needs ``f``, to decide the forest. Everything after it works on the
polynomials, which resolve ``f`` to `eps`:

- the boxes added for the 2:1 balance take their parent's polynomial;
- the boxes outside the cell copy their lattice images or interpolate the
  core leaves (see above);
- the wide source tree of non-tiled cells interpolates the forest leaves
  instead of sampling ``f`` again. Its nodes are an affine image of a tensor
  grid, upper triangular in the rotated frame, so on each leaf the polynomial
  is contracted along z, then y, then x (about 90 ns per node);
- a plan re-solve evaluates ``f`` on the leaves of the adaptive refinement
  only and derives the other boxes in the same way.

`timings[1]` is the sampling stage; the rest of the time does not depend on
the cost of ``f``. Compared with sampling ``f`` everywhere
(`sample_once = false`, the C++ rules), ``f`` is called 2–4 times less often,
the potential changes by less than `eps` (at most 0.4 `eps` in the tests),
and the accuracy against Ewald is unchanged.

## Fast densities

At high accuracy, evaluating a point-by-point density can take half of the
single-thread time. A batch density `f_batch!(values, points)` (`values (1, npts)`,
`points (3, npts)`) evaluates a whole box at once and can use SIMD. For
example, `density_batch!` in `benchmark/periodic_vs_cpp.jl` evaluates a
periodized Gaussian mixture on 8 points at a time with SIMD.jl and a
vectorized `exp`. It is about 30 times faster per point than the scalar
version. With `sample_once` it shortens only the sampling stage, for example
0.67 → 0.10 s on the hexagonal cell at `eps = 1e-9` (see [Performance](@ref)):

```julia
res = bdmk_periodic(nothing, A; eps = 1e-9, f_batch = (v, p) -> density_batch!(v, p, mixture))
```

## Method

The implementation in `src/native/periodic/` follows the C++
`bdmk_laplace3d_pot_periodic`.

- **Forest.** In the rotated frame, a grid of cubes of side ``r_c`` covers the
  cell plus a one-box image halo. Each cube is refined adaptively, and the
  forest is 2:1 balanced.
- **Kernel split.** ``1/r`` is written as a sum of Gaussians (the u-series).
  The Gaussians that decay within ``r_c`` go through the free-space box
  machinery on the forest (halo boxes act as sources only).
- **Wide Gaussians** are summed in reciprocal space, exactly per axis with
  ``e^{-i\alpha\xi} = \sum_l (2l+1)(-i)^l j_l(\alpha) P_l(\xi)``.
  - Source: Legendre moments of one period. They come from the level-0 proxy
    charges when the cubes tile the cell, and otherwise from an adaptive tree
    in fractional coordinates, where ``e^{-2\pi i m\cdot t}`` is separable.
    That tree also refines wherever the main forest is much finer, so narrow
    features cannot hide between its sample points.
  - Target: the Legendre coefficients of the wide potential on each level-0
    cube. This is a separable product for orthorhombic cells. Otherwise
    ``k = 2\pi U^{-T} m`` is lower triangular, and the contraction runs slice
    by slice. The coefficients are injected into the level-0 proxy potentials
    before the downward pass.
  - The density is real, so only half of the modes are computed. No FFT is
    needed.

## Accuracy

Against an Ewald-summed Gaussian mixture, the max error is at most `eps`
relative to ``\max|u|`` (0.3–1.0 `eps` over 3000 nodes per case), and the
relative ``L^2`` error is 1–4 `eps`, at `eps = 1e-3, 1e-6, 1e-9`. This holds
on cubic, orthorhombic (1:1.5:2, 1:√2:√3), sheared, hexagonal and
elongated (1:1:4) triclinic cells, for neutral and charged densities
(`test/test_periodic.jl`). A lone spike of width ``1.4\cdot10^{-3}`` on a
sheared cell, a known failure of the C++ wide source (error
``1.9\cdot10^{-5}``), is computed to ``6.5\cdot10^{-10}`` at `eps = 1e-9`. On
the C++ benchmark problems both codes build the same forest, and the
potentials agree to ``2\cdot10^{-13}`` at `eps = 1e-9`.
