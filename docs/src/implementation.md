# Implementation notes

## Free-space solver

The solver (`src/native/`) follows the nine steps of the Fortran `bdmk`. It
keeps all of Fortran's numerical choices: SOG nodes, plane-wave quadratures
and lengths, proxy orders, interaction lists, local tables and their sparsity
patterns, and the dense-versus-sparse switch of the near field. That is why
its results agree with Fortran to rounding. The speed comes from restructuring
the work:

- **SIMD GEMM kernels** (`gemm.jl`, SIMD.jl). Every tensor-product stage is a
  small register-blocked matrix product with no data transposes. The kernel
  runs at about the FP64 peak of one core (≈ 60 GFLOP/s on Zen 4), at or above
  OpenBLAS for these sizes.
- **Near field.** Same-level neighbours are evaluated target-centrically,
  `pot += w Σ_dz Tz(Σ_dy Ty(Σ_dx Tx f_s))`, sharing the y/z stages among
  sources (about 2× fewer flops). Sparse pairs use GEMMs on pattern-zeroed
  tables.
- **Plane waves.** Conjugate symmetry halves all complex transforms, which run
  as real GEMMs, and leaves form and evaluate plane waves directly from and to
  their grid values. The kernel's Fourier transform is folded into the
  outgoing expansions. The translation (M2L) runs over frequency blocks of all
  boxes, cache-blocked and partitioned over threads.
- **Tree passes.** Parent↔child transfers share stages among the 8 children
  and fuse the leaf value↔proxy maps. Proxy data is stored only for nonleaf
  boxes.
- **Runtime.** Subnormals are flushed to zero during the solve (Gaussian tails
  otherwise fall into slow subnormal arithmetic). Large buffers come from a
  malloc arena that is reused across solves, so they cause no GC pressure.
  Boxes are scheduled dynamically.

## Periodic solver

`src/native/periodic/`:

- `forest.jl`: `PeriodicGrid` (QR frame, level-0 grid and image halo) and
  `build_periodic_forest`. The forest uses Legendre tail refinement and 2:1
  balance, with halo copies or core interpolation. Interpolated nodes reuse
  the previous node's core leaf when it holds their image, and are evaluated
  per leaf by three small products.
- `fractional.jl`: the source tree in fractional coordinates for cells whose
  level-0 cubes do not tile the cell.
- `wide.jl`: the wide-Gaussian transforms, moments → modes → Legendre
  coefficients. They only compute the half space ``m_1 \ge 0`` (the density is
  real) and keep real and imaginary parts as separate arrays. Every
  contraction is a set of real GEMMs threaded over slices or row blocks, so
  OpenBLAS threads never compete with Julia's.
- `slab.jl`: the wide part of quasi-2D slabs. In-plane transforms as in
  `wide.jl`, exact Gaussian z-kernel matrices between the level-0 z-cells per
  wide Gaussian, and the ``\dot k = 0`` kernel ``D(z)`` of all wide
  Gaussians.
- `periodic.jl`: the u-series SOG, the plan and the solve (narrow Gaussians
  through the free-space machinery, the wide term injected before the
  downward pass, the neutralizing background).

## Repository layout

- `src/core/`, `src/tree/`: types, bases, adaptive tree construction.
- `src/native/`: the box-DMK solver; `src/native/periodic/`: the periodic
  solver.
- `src/solver/`: `bdmk`, targets, `PotentialField`.
- `src/fortran/`: optional wrappers of the Fortran reference
  (`deps/boxdmk_fortran/`).
- `benchmark/`: `native_vs_fortran.jl`, `periodic_vs_cpp.jl` and node
  runners; `benchmark/results/` holds the logs.
- `test/`: the test suite. The Fortran parity tests run when the reference is
  built.

## Development

```bash
julia --project -t auto -e 'using Pkg; Pkg.test()'
julia --project -t 16 benchmark/native_vs_fortran.jl results.md
julia --project -t 48 benchmark/periodic_vs_cpp.jl hex 9 multi 3
```

`scripts/generate_fortran_tables.jl` regenerates `src/native/tables/` from the
Fortran DATA statements.

## Limitations

- The solvers are 3D only (the tree builder also supports 1D and 2D).
- `eps` must lie in `(1e-12, 1e-2]`, the range covered by the Fortran tables.
