# Quasi-2D (slab) potentials

[`bdmk_q2d`](@ref) computes the Coulomb potential of a density in a slab that
is periodic in x and y (in-plane lattice ``A = [a_1\ a_2]``) and free in z:

```math
u(x) = \int \sum_{n \in \mathbb{Z}^2} \frac{f(y)}{|x - y - (An, 0)|}\,dy .
```

The density must vanish outside the slab ``z_0 \le z \le z_1``. A neutral
density is the default (checked). With `charged = true` a net charge ``Q``
per cell is allowed, and the potential is fixed by the gauge
``u + 2\pi Q|z|/|A| \to 0`` far from the slab. The potential is computed
inside the slab.

```julia
L = (100.0, 100.0)                       # rectangle; a number for a square
res = bdmk_q2d(f, L, (0.0, 1.0); eps = 1e-6)
evaluate_periodic(res, [12.3, 45.6, 0.5])

A = [10.0 5.0; 0.0 5sqrt(3)]             # oblique (here hexagonal) cell, columns a₁, a₂
res = bdmk_q2d(f, A, (-7.0, 7.0); eps = 1e-6, charged = true)

plan = q2d_plan(f0, L, (0.0, 1.0); eps = 1e-6)   # repeated solves
res = periodic_solve!(plan, f1)
```

Options are those of [`bdmk_periodic`](@ref), plus:
- `rc`, the side of the level-0 boxes, with ``r_c \le \min(d, H)``. Here ``d``
  is the in-plane spacing of the lattice lines (``\min(L_1, L_2)`` for a
  rectangle) and ``H = z_1 - z_0``.
- The default `rc`: for a rectangle, the largest value dividing both sides,
  so that the cubes tile the cell (``r_c = H`` for a thin slab); otherwise
  ``\min(d, H)``.

The forest and `res.pot` are in the frame rotated so that ``a_1`` lies along
x.

### Densities given by values, one-period integrals

For densities known only through a basis, as in ISDF:
- `P = sample_points(plan)` gives the points where a solve samples the density.
- `periodic_solve!(plan, v)` takes the values `v[j]` at those points.

[`period_quadrature`](@ref) gives a quadrature over exactly one period, with
nodes `Q.x` and weights `Q.w`. It is built on parallelepiped cells in
fractional coordinates and refined to the forest's resolution.
`period_values!(out, Q, res)` interpolates the potential to its nodes. For
example, ``\int_{\text{cell}} g\,u`` is `sum(Q.w .* gq .* out)`.

## Method

The narrow Gaussians of the u-series (those that decay within `rc`) go
through the real-space box machinery, as in the periodic solver, with image
boxes in x and y only. The wide Gaussians are summed after Poisson summation
over the in-plane lattice:

```math
\sum_n e^{-|x - y - n\circ L|^2/\delta}
  = \frac{\pi\delta}{L_1 L_2} \sum_{\dot k} e^{-\delta|\dot k|^2/4}
    e^{i\dot k\cdot(\dot r - \dot r')} \, e^{-(z - z')^2/\delta},
\qquad \dot k = 2\pi(m_1/L_1, m_2/L_2).
```

- **In the plane**, the transforms between Legendre moments of the source
  cells and the modes ``\dot k = 2\pi A^{-T} m`` are exact per-axis
  Legendre–spherical-Bessel expansions (half space ``m_1 \ge 0``), as in the
  periodic solver. The source cells tile one period: the level-0 cubes when
  they tile a rectangle, else the cells of a tree in fractional coordinates. They are
  factored through the uniform grid of cells: with ``c_s = (s+\tfrac12) r_c``
  and ``k = 2\pi m/(m_t r_c)``,
  ``e^{-ikc_s} = e^{-i\pi m/m_t} e^{-2\pi i (m \bmod m_t) s/m_t}``. The sum over
  cells is then a (real) FFT of length ``m_t`` per Legendre degree, and the
  modes fold onto its frequencies. This is exact, with no window or
  oversampling as in a NUFFT, and costs O(N log N) instead of
  O(modes × cells) per axis. For an oblique cell the cubes do not tile the
  period, and two back-transforms (modes to cube coefficients) are available.
  - **Dense:** the triangular transform of the periodic solver (``k_2``
    depends on ``m_1``). It is the exact projection of each plane wave, done
    with GEMMs, at a cost cubic in the number of cubes per axis.
  - **NUFFT:** a type-2 NUFFT evaluates each z component of the potential,
    a real 2D Fourier series in ``\xi = 2\pi A^{-1}x``, at
    ``q = p_e + 8`` Gauss nodes per axis of each target level-0 cube.
    - Kernel: exponential of semicircle, fine grid twice the band, tolerance
      0.01 `eps`.
    - The values are projected onto the cube's Legendre basis (degree
      ``< p_e``), the same coefficients the dense path injects into the proxy
      potentials, so the downward pass and level-0 leaves are unchanged.
    - The number of targets is proportional to the number of cubes, and the
      cost is about linear in it.
  - The option `wide_nufft` (`true`, `false`, or `nothing` = automatic, by a
    calibrated cost model) chooses between them. The NUFFT pays off from about
    50 cubes per axis: 4× faster at 192 × 111 cubes and `eps = 1e-6`, with
    0.6× the memory. The two agree to about 0.1 `eps`
    (`benchmark/q2d_nufft.jl`, `benchmark/results/q2d_nufft/`).
- **In z**, there is no Fourier transform and no zero padding. On a level-0
  z-cell (side `rc`) every wide Gaussian is smooth, so its z-convolution is a
  small exact matrix in the Legendre basis of the cells, computed once by
  quadrature. For each mode the z-part is ``\sum_\ell c_\ell(\dot k) G_\ell``,
  applied to all modes by one product per Gaussian. This is the idea of the
  Fourier–Chebyshev solver of the spectral SOG method for quasi-2D systems
  (Gao, Jiang, Liang, Xu and Zhou), made exact in the cell basis.
- **The mode ``\dot k = 0``** takes the kernel of all wide Gaussians,
  ``D(z) = \sum_\ell (\pi w_\ell \delta_\ell/L_1L_2)(e^{-z^2/\delta_\ell} - 1)``,
  which converges (the tail beyond the u-series is summed in closed form) and
  tends to the exact ``-2\pi|z|/|A|``. The constants left out cancel for a
  neutral density. For a charged one they set the gauge: the narrow images
  add ``Q\sum_{\rm narrow}\pi w\delta/|A|``. The u-series stops resolving
  ``1/r`` below ``2^{-16} r_c`` (where it saturates), which shifts the whole
  kernel by ``1.3\times10^{-5}\,r_c/|A|``. Both constants are subtracted. It plays the role of the separate ``k = 0`` term of
  2-periodic Ewald summation (Lindbo and Tornberg).

## Accuracy and cost

Against a 2-periodic Ewald sum (`test/q2d_oracle.jl`), the max error is
0.03–0.23 `eps` relative to ``\max|u|`` at `eps = 1e-3, 1e-6, 1e-9`. The
tests (`test/test_q2d.jl`) cover:
- a square cell, a 2 × 3 cell and a thin 4 × 4 × 0.5 slab;
- an oblique cell and a 1 × √2 rectangle (cubes not tiling);
- charged slabs, rectangular and oblique, with rel. L2 error 3e-10 at
  `eps = 1e-9`.

For the 100 × 100 × 1 slab at `eps = 1e-6` (`rc = 1`, 389k modes), the wide
part takes 133 ms of 0.84 s at 48 threads. See [Performance](@ref).
