# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Run full test suite
julia --project -e 'using Pkg; Pkg.test()'

# Run a single test file
julia --project -e 'using BoxDMK; include("test/test_solver.jl")'

# Run tests with threading
julia --project --threads=auto -e 'using Pkg; Pkg.test()'

# Add a dependency
julia -e 'using Pkg; Pkg.activate("."); Pkg.add("PackageName")'

# Precompile
julia --project -e 'using BoxDMK'

# Run benchmark
julia --project --threads=auto benchmark/benchmark.jl
```

## Architecture

BoxDMK.jl computes **volume potentials** (convolution of a source density with a kernel) on adaptive hierarchical box trees in 3D. It uses a kernel-independent approach: any kernel is decomposed as a **Sum-of-Gaussians (SOG)**, then evaluated via a **Box Fast Gauss Transform (FGT)** combined with a multi-level FMM-like hierarchy.

This is a Julia translation of the Fortran BoxDMK library (`/mnt/home/xgao1/codes/boxdmk/`).

### Public API

```julia
tree, fvals = build_tree(f, kernel, basis; ndim=3, norder=6, eps=1e-6, boxlen=1.0, nd=1)
result = bdmk(tree, fvals, kernel; eps=1e-6, grad=false, hess=false, targets=nothing)
# result.pot, result.grad, result.hess, result.target_pot

field = PotentialField(tree, result)   # callable handle: field(x) evaluates anywhere
# evaluate_potential / evaluate_gradient / evaluate_hessian (src/solver/field.jl)
```

### Type Dispatch

Kernel and basis selection uses Julia's type system instead of integer flags:
- `AbstractKernel` → `LaplaceKernel`, `YukawaKernel{T}`, `SqrtLaplaceKernel`
- `AbstractBasis` → `LegendreBasis`, `ChebyshevBasis`

### 9-Step Solver Pipeline (`src/solver.jl` → `bdmk()`)

1. **Precomputation** — SOG nodes, proxy data, interaction lists, local tables, PW data
2. **Taylor corrections** — kernel-dependent residual correction using Laplacian/biLaplacian
3. **Upward pass** — leaf density → proxy charges, anterpolate up tree via `c2p_transmat`
4. **Form multipole PW** — proxy charges → plane wave expansions (complex)
5. **M2L translation** — shift + kernel FT multiply between well-separated boxes
6. **Local PW → proxy potential** + downward pass via `p2c_transmat`
7. **Direct local interactions** — near-field via precomputed sparse 1D tables + tensor product
8. **Asymptotic expansion** — small-delta SOG components handled analytically
9. **Output conversion** — proxy potential → user grid, optional grad/hess/targets

### Key Data Flow

- `fvals (nd, npbox, nboxes)` — source density on tensor grid per box
- `proxy_charges/proxy_pot (ncbox, nd, nboxes)` — proxy grid (always Legendre, order from `select_porder(eps)`)
- `PlaneWaveData.rmlexp` — complex PW expansion workspace
- `LocalTables.tab (norder, norder, nloctab2, ndeltas, nlevels+1)` — 5D sparse interaction tables

### Module Include Order

`types.jl` → `utils.jl` → `sog.jl` → `basis.jl` → `tensor.jl` → `kernels.jl` → `proxy.jl` → `tree.jl` → `tree_data.jl` → `passes.jl` → `interaction_lists.jl` → `derivatives.jl` → `local_tables.jl` → `local.jl` → `planewave.jl` → `boxfgt.jl` → `solver.jl`

### SOG Data

Pre-computed Sum-of-Gaussians approximation tables are stored as JLD2 files in `data/sog/`. These were extracted from Fortran source via `scripts/extract_sog_tables.jl`. Each file maps precision level ("1e-2" through "1e-12") to `(weights, deltas, r0)`.

### Fortran Reference

The original Fortran code is at `/mnt/home/xgao1/codes/boxdmk/`. Key files:
- `src/bdmk/bdmk4.f` — main 9-step solver
- `src/bdmk/boxfgt_md.f` — Box FGT
- `src/bdmk/bdmk_local.f` / `bdmk_local_tables.f` — near-field
- `src/common/tree_vol_coeffs.f` — tree construction
- `src/common/dmk_routs.f` — proxy/PW transforms
- `build/libboxdmk.so` — compiled shared library with C API
