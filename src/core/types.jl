# Kernels
abstract type AbstractKernel end
struct LaplaceKernel <: AbstractKernel end
struct YukawaKernel{T<:Real} <: AbstractKernel
    beta::T
end
struct SqrtLaplaceKernel <: AbstractKernel end

# Polynomial bases
abstract type AbstractBasis end
struct LegendreBasis <: AbstractBasis end
struct ChebyshevBasis <: AbstractBasis end

# Adaptive tree (geometry only — no function data)
struct BoxTree{T<:Real, B<:AbstractBasis}
    ndim::Int
    nlevels::Int
    centers::Matrix{T}           # (ndim, nboxes)
    boxsize::Vector{T}           # one per level (0:nlevels)
    parent::Vector{Int}
    children::Matrix{Int}        # (2^ndim, nboxes); 0 = no child
    colleagues::Vector{Vector{Int}}
    level::Vector{Int}
    basis::B
    norder::Int
end

# Solver output
struct SolverResult{T<:Real}
    pot::Array{T,3}                              # (nd, npbox, nboxes)
    grad::Union{Nothing, Array{T,4}}             # (nd, ndim, npbox, nboxes)
    hess::Union{Nothing, Array{T,4}}             # (nd, nhess, npbox, nboxes)
    target_pot::Union{Nothing, Matrix{T}}        # (nd, ntarg)
    target_grad::Union{Nothing, Array{T,3}}      # (nd, ndim, ntarg)
    target_hess::Union{Nothing, Array{T,3}}      # (nd, nhess, ntarg)
end
