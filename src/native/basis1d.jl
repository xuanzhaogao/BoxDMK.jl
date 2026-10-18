# One-dimensional polynomial tables (Fortran `legeexps2` / `chebexps2`).

struct Ortho1D
    n::Int
    x::Vector{Float64}      # nodes on [-1, 1], ascending
    w::Vector{Float64}      # quadrature weights
    U::Matrix{Float64}      # values -> coefficients
    V::Matrix{Float64}      # coefficients -> values
    Vp::Matrix{Float64}     # coefficients -> first-derivative values
    Vpp::Matrix{Float64}    # coefficients -> second-derivative values
end

function ortho1d(::LegendreBasis, n::Integer)
    x, w, U, V, Vp, Vpp = _legendre_gauss_data(Int(n))
    return Ortho1D(Int(n), x, w, Matrix(U), Matrix(V), Matrix(Vp), Matrix(Vpp))
end

function ortho1d(::ChebyshevBasis, n::Integer)
    return setprecision(BigFloat, 256) do
        x, w, U, V, Vp, Vpp = _chebyshev_gauss_data(BigFloat, Int(n))
        Ortho1D(Int(n), Float64.(x), Float64.(w), Float64.(U), Float64.(V), Float64.(Vp), Float64.(Vpp))
    end
end

# Values of P_0..P_{n-1} (Legendre) at x.
function legepols!(p::AbstractVector, x::Real, n::Integer)
    p[1] = 1.0
    n == 1 && return p
    p[2] = x
    for k in 2:(n - 1)
        p[k + 1] = ((2k - 1) * x * p[k] - (k - 1) * p[k - 1]) / k
    end
    return p
end

# Values of T_0..T_{n-1} (Chebyshev) at x.
function chebpols!(p::AbstractVector, x::Real, n::Integer)
    p[1] = 1.0
    n == 1 && return p
    p[2] = x
    for k in 2:(n - 1)
        p[k + 1] = 2x * p[k] - p[k - 1]
    end
    return p
end

polyvals!(p, ::LegendreBasis, x, n) = legepols!(p, x, n)
polyvals!(p, ::ChebyshevBasis, x, n) = chebpols!(p, x, n)

# Parent->child interpolation and child->parent anterpolation matrices on
# coefficients (Fortran `dmk_get_coefs_translation_matrices`).  Returns
# (p2c, c2p), each indexed [side] with side 1 = minus child, 2 = plus child.
function coefs_translation_matrices(basis::AbstractBasis, n::Int)
    t = ortho1d(basis, n)
    p = zeros(n)
    trans = ntuple(2) do side
        sgn = side == 1 ? -1.0 : 1.0
        polyv = zeros(n, n)        # polyv[j, i] = P_{j-1}(x_i/2 ± 1/2)
        for i in 1:n
            polyvals!(p, basis, t.x[i] / 2 + sgn * 0.5, n)
            polyv[:, i] .= p
        end
        # trans(i, j) = sum_k U(i, k) * polyv(j, k)
        t.U * transpose(polyv)
    end
    p2c = (trans[1], trans[2])
    c2p = (Matrix(transpose(trans[1])), Matrix(transpose(trans[2])))
    return p2c, c2p
end
