# Quasi-2D (slab) potentials

[`bdmk_q2d`](@ref) computes the Coulomb potential of a neutral density in a
slab that is periodic in x and y (rectangle ``L_1 \times L_2``) and free in z:

```math
u(x) = \int \sum_{n \in \mathbb{Z}^2} \frac{f(y)}{|x - y - (n_1 L_1, n_2 L_2, 0)|}\,dy .
```

The density must vanish outside the slab ``z_0 \le z \le z_1`` and have zero
total charge (otherwise the 2-periodic sum diverges). The potential is
computed inside the slab.

```julia
L = (100.0, 100.0)                       # or a number for a square cell
res = bdmk_q2d(f, L, (0.0, 1.0); eps = 1e-6)
evaluate_periodic(res, [12.3, 45.6, 0.5])

plan = q2d_plan(f0, L, (0.0, 1.0); eps = 1e-6)   # repeated solves
res = periodic_solve!(plan, f1)
```

Options are those of [`bdmk_periodic`](@ref), and `rc`, the side of the
level-0 boxes: ``r_c \le \min(L_1, L_2, H)`` (``H = z_1 - z_0``), dividing
both in-plane sides; by default the largest such value (``r_c = H`` for a thin
slab, so the slab is one level-0 z-cell).

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

- **In the plane**, the transforms between Legendre moments of the level-0
  cubes and the modes ``\dot k`` are exact per-axis Legendre–spherical-Bessel
  expansions (half space ``m_1 \ge 0``), as in the periodic solver. They are
  factored through the uniform grid of cells: with ``c_s = (s+\tfrac12) r_c``
  and ``k = 2\pi m/(m_t r_c)``,
  ``e^{-ikc_s} = e^{-i\pi m/m_t} e^{-2\pi i (m \bmod m_t) s/m_t}``. The sum over
  cells is then a (real) FFT of length ``m_t`` per Legendre degree, and the
  modes fold onto its frequencies. This is exact, with no window or
  oversampling as in a NUFFT, and costs O(N log N) instead of
  O(modes × cells) per axis.
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
  tends to the exact ``-2\pi|z|/(L_1L_2)``. The constants left out cancel for
  a neutral density. It plays the role of the separate ``k = 0`` term of
  2-periodic Ewald summation (Lindbo and Tornberg).

## Accuracy and cost

Against a 2-periodic Ewald sum (`test/q2d_oracle.jl`), the max error is
0.03–0.23 `eps` relative to ``\max|u|`` at `eps = 1e-3, 1e-6, 1e-9`. The
tests cover a square cell, a 2 × 3 cell and a thin 4 × 4 × 0.5 slab
(`test/test_q2d.jl`).

For the 100 × 100 × 1 slab at `eps = 1e-6` (`rc = 1`, 389k modes), the wide
part takes 133 ms of 0.84 s at 48 threads. See [Performance](@ref).
