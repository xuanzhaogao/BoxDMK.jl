# Analytic periodic Coulomb potential of a periodized Gaussian mixture
# f(x) = Σ_j c_j Σ_n exp(-α_j |x - x_j - A n|²) with a uniform neutralizing
# background (k = 0 mode dropped), by Ewald summation.

using LinearAlgebra

struct GaussMixture
    A::Matrix{Float64}                  # lattice vectors (columns)
    x::Vector{NTuple{3,Float64}}
    alpha::Vector{Float64}
    c::Vector{Float64}
    Ainv::Matrix{Float64}
    m::Vector{Int}                      # image half range around the nearest image
end

_cellmat(L::Tuple) = Matrix(Diagonal(collect(Float64.(L))))
_cellmat(A::AbstractMatrix) = Matrix{Float64}(A)
_dmin(A) = minimum(1 / norm(inv(A)[i, :]) for i in 1:3)

function GaussMixture(cell, x, alpha, c)
    A = _cellmat(cell)
    m = [max(1, ceil(Int, sqrt(46 / a) / _dmin(A) - 0.5)) for a in alpha]
    return GaussMixture(A, x, collect(Float64, alpha), collect(Float64, c), inv(A), m)
end

# Mixture with the last coefficient set for charge neutrality.
function neutral_mixture(cell, x, alpha, c)
    c = collect(Float64, c)
    q = c .* (pi ./ alpha) .^ 1.5
    c[end] = -sum(q[1:end-1]) / (pi / alpha[end])^1.5
    return GaussMixture(cell, x, alpha, c)
end

function (g::GaussMixture)(p::AbstractVector)
    s = 0.0
    A = g.A
    for j in eachindex(g.alpha)
        a = g.alpha[j]; m = g.m[j]
        d = (p[1] - g.x[j][1], p[2] - g.x[j][2], p[3] - g.x[j][3])
        t = ntuple(i -> round(g.Ainv[i, 1] * d[1] + g.Ainv[i, 2] * d[2] + g.Ainv[i, 3] * d[3]), 3)
        for n3 in -m:m, n2 in -m:m, n1 in -m:m
            nn = (t[1] + n1, t[2] + n2, t[3] + n3)
            r2 = 0.0
            for i in 1:3
                e = d[i] - (A[i, 1] * nn[1] + A[i, 2] * nn[2] + A[i, 3] * nn[3])
                r2 += e * e
            end
            r2 * a < 46 && (s += g.c[j] * exp(-a * r2))
        end
    end
    return s
end

erfc_(x::Float64) = ccall((:erfc, Base.Math.libm), Float64, (Float64,), x)

function ewald_potential(g::GaussMixture, p::AbstractVector; beta = 40.0 / _dmin(g.A)^2)
    A = g.A
    V = abs(det(A))
    dmin = _dmin(A)
    Bk = 2pi * transpose(g.Ainv)                  # k = Bk m
    u = 0.0
    for j in eachindex(g.alpha)
        a = g.alpha[j]
        q = g.c[j] * (pi / a)^1.5
        gam = a * beta / (a + beta)
        d = [p[i] - g.x[j][i] for i in 1:3]
        d .-= A * round.(g.Ainv * d)
        # real space: (erf(√α r) - erf(√γ r)) / r over images
        nimg = ceil(Int, 7 / sqrt(gam) / dmin) + 1
        for n3 in -nimg:nimg, n2 in -nimg:nimg, n1 in -nimg:nimg
            e = d - A * [n1, n2, n3]
            r = norm(e)
            u += q * (r < 1e-12 ? 2 * (sqrt(a) - sqrt(gam)) / sqrt(pi) :
                      (erfc_(sqrt(gam) * r) - erfc_(sqrt(a) * r)) / r)
        end
        u -= q * pi / (beta * V)
        # reciprocal space: (4π/V) e^{-k²/4γ}/k² e^{ik·d}
        kcut = 2 * sqrt(gam) * 7
        mm = [ceil(Int, kcut * norm(A[:, i]) / 2pi) for i in 1:3]
        for m3 in -mm[3]:mm[3], m2 in -mm[2]:mm[2], m1 in -mm[1]:mm[1]
            k = Bk * [m1, m2, m3]
            k2 = dot(k, k)
            (k2 == 0 || k2 > kcut^2) && continue
            u += q * 4pi / V * exp(-k2 / (4gam)) / k2 * cos(dot(k, d))
        end
    end
    return u
end
