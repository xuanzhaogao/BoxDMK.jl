module BoxDMK

using LinearAlgebra
using Libdl
using SIMD
using FFTW

"""
    build_tree(f, kernel, basis; ndim=3, norder=6, eps=1e-6, boxlen=1.0, nd=1, eta=1.0, f_batch=nothing)
        -> (tree::BoxTree, fvals)

Adaptive tree on the box `[0, boxlen]^ndim` resolving the density `f` to
relative tolerance `eps` (the Fortran builder: Legendre/Chebyshev tail test,
2:1 level restriction, colleague lists). `f(x)` returns the `nd` density
values at the point `x` (a number when `nd == 1`); alternatively
`f_batch!(values, points)` fills `values (nd, npts)` at `points (ndim, npts)`
for a whole box at once, and `f` may then be omitted. `fvals` holds the
samples, `(nd, norder^ndim, nboxes)`.
"""
function build_tree end
function bdmk end

const _SRC_ROOT = @__DIR__
const _PACKAGE_ROOT = normpath(joinpath(_SRC_ROOT, ".."))

# Core
include("core/types.jl")
include("core/utils.jl")
include("core/basis.jl")
include("core/tensor.jl")

# Tree construction
include("tree/tree.jl")

# Pure-Julia box-DMK solver
include("native/basis1d.jl")
include("native/pw.jl")
include("native/sog.jl")
include("native/flattree.jl")
include("native/loctab.jl")
include("native/gemm.jl")
include("native/tensor3.jl")
include("native/local3.jl")
include("native/pwops.jl")
include("native/driver.jl")
include("native/periodic/forest.jl")
include("native/periodic/fftaxis.jl")
include("native/periodic/wide.jl")
include("native/periodic/slab.jl")
include("native/periodic/fractional.jl")
include("native/periodic/periodic.jl")
include("solver/solver.jl")
include("solver/field.jl")

# Optional Fortran reference (requires the vendored library, see deps/)
include("fortran/fortran_paths.jl")
include("fortran/fortran_wrapper.jl")

export LaplaceKernel, YukawaKernel, SqrtLaplaceKernel
export LegendreBasis, ChebyshevBasis
export build_tree, bdmk, bdmk_native
export bdmk_periodic, evaluate_periodic, PeriodicResult, PeriodicPlan, periodic_plan, periodic_solve!
export bdmk_q2d, q2d_plan
export PotentialField, evaluate_potential, evaluate_gradient, evaluate_hessian
export FortranTreeData, build_tree_fortran, pack_tree_fortran, bdmk_fortran

end
