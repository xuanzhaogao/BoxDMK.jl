module BoxDMK

using LinearAlgebra
using JLD2
using Libdl

function build_tree end
function bdmk end

const _SRC_ROOT = @__DIR__
const _PACKAGE_ROOT = normpath(joinpath(_SRC_ROOT, ".."))

# Core
include("core/types.jl")
include("core/utils.jl")
include("solver/sog.jl")
include("solver/proxy.jl")
include("solver/passes.jl")
include("core/kernels.jl")
include("core/basis.jl")
include("core/tensor.jl")

# Tree construction and geometry
include("tree/tree.jl")
include("tree/tree_data.jl")
include("solver/derivatives.jl")
include("tree/interaction_lists.jl")

# Solver pipeline
include("solver/local_tables.jl")
include("fortran/fortran_paths.jl")
include("solver/planewave.jl")
include("solver/boxfgt.jl")
include("solver/local.jl")
include("solver/solver.jl")
include("solver/field.jl")

# Fortran integration
include("fortran/fortran_hotpaths.jl")
include("fortran/fortran_wrapper.jl")
include("fortran/fortran_debug_wrapper.jl")

export LaplaceKernel, YukawaKernel, SqrtLaplaceKernel
export LegendreBasis, ChebyshevBasis
export build_tree, build_interaction_lists, bdmk
export PotentialField, evaluate_potential, evaluate_gradient, evaluate_hessian
export load_sog_nodes
export taylor_coefficients, taylor_coefficients_grad, taylor_coefficients_hess
export select_porder, build_proxy_data, density_to_proxy!, proxy_to_potential!
export compute_laplacian!, compute_bilaplacian!, compute_gradient_density!
export compute_hessian_density!, eval_asymptotic!, apply_asymptotic!
export compute_gradient!, compute_hessian!
export upward_pass!, downward_pass!
export build_local_tables
export get_pw_term_count, get_pw_nodes, build_pw_conversion_tables
export build_pw_shift_matrices, kernel_fourier_transform, setup_planewave_data
export DeltaGroups, get_delta_cutoff_level, group_deltas_by_level
export boxfgt!, handle_fat_gaussian!
export apply_local!
export FortranTreeData, build_tree_fortran, pack_tree_fortran, bdmk_fortran
export reset_fortran_debug!, get_fortran_debug_snapshot

end
