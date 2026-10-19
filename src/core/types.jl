# Kernels
abstract type AbstractKernel end

"""
    LaplaceKernel()

The Laplace kernel `K(r) = 1/r` (3D), so that `u = ∫ K f` solves `-Δu = 4π f`.
"""
struct LaplaceKernel <: AbstractKernel end

"""
    YukawaKernel(beta)

The Yukawa (screened Coulomb) kernel `K(r) = exp(-β r)/r` (3D).
"""
struct YukawaKernel{T<:Real} <: AbstractKernel
    beta::T
end

"""
    SqrtLaplaceKernel()

The square-root Laplace kernel `K(r) = 1/r²` (3D), the inverse of `(-Δ)^{1/2}` up to a constant.
"""
struct SqrtLaplaceKernel <: AbstractKernel end

# Polynomial bases
abstract type AbstractBasis end

"""
    LegendreBasis()

Tensor-product Legendre polynomials on Gauss-Legendre nodes in each box.
"""
struct LegendreBasis <: AbstractBasis end

"""
    ChebyshevBasis()

Tensor-product Chebyshev polynomials on Chebyshev nodes in each box.
"""
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
