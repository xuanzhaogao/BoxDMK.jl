# Quasi-2D (periodic in x, y with the in-plane lattice A = [a₁ a₂], free in z)
# Coulomb potential of a neutral Gaussian mixture
# f(x) = Σ_j c_j Σ_n exp(-α_j |x - x_j - (An, 0)|²), by 2-periodic Ewald
# summation with parameter ξ < √α_j:
#   erf(√α r)/r = (erfc(ξr) - erfc(√α r))/r [real space] + erf(ξr)/r [Fourier, in plane],
# the k̇ = 0 term of the latter -(2√π/|A|)(e^{-ξ²z²}/ξ + √π z erf(ξz)) (Lindbo & Tornberg 2012),
# k̇ = 2πA⁻ᵀm.

using LinearAlgebra

struct Q2DMixture
    A::Matrix{Float64}                  # in-plane lattice vectors (columns)
    Ainv::Matrix{Float64}
    x::Vector{NTuple{3,Float64}}
    alpha::Vector{Float64}
    c::Vector{Float64}
    m::Vector{NTuple{2,Int}}            # in-plane image half ranges for the density
end

_lattice2(L::Real) = Matrix{Float64}(L * I, 2, 2)
_lattice2(L::Tuple) = Matrix{Float64}(Diagonal([L[1], L[2]]))
_lattice2(L::AbstractMatrix) = Matrix{Float64}(L)

# image half range along each lattice vector covering distance r
_img_range(Ainv, r) = ntuple(d -> ceil(Int, r * norm(view(Ainv, d, :))) + 1, 2)

# Mixture with charges q_j = c_j (π/α_j)^{3/2}; `neutral` sets the last for charge neutrality.
function q2d_mixture(L, x, alpha, c; neutral = true)
    A = _lattice2(L)
    Ainv = inv(A)
    c = collect(Float64, c)
    if neutral
        q = c .* (pi ./ alpha) .^ 1.5
        c[end] = -sum(q[1:end-1]) / (pi / alpha[end])^1.5
    end
    m = [_img_range(Ainv, sqrt(46 / a)) for a in alpha]
    return Q2DMixture(A, Ainv, x, collect(Float64, alpha), c, m)
end
neutral_q2d_mixture(L, x, alpha, c) = q2d_mixture(L, x, alpha, c)

total_charge(g::Q2DMixture) = sum(g.c .* (pi ./ g.alpha) .^ 1.5)

# in-plane difference reduced to the nearest image (fractional round)
function _reduce2(g::Q2DMixture, d1, d2)
    t1 = g.Ainv[1, 1] * d1 + g.Ainv[1, 2] * d2
    t2 = g.Ainv[2, 1] * d1 + g.Ainv[2, 2] * d2
    t1 -= round(t1); t2 -= round(t2)
    return g.A[1, 1] * t1 + g.A[1, 2] * t2, g.A[2, 1] * t1 + g.A[2, 2] * t2
end

function (g::Q2DMixture)(p::AbstractVector)
    s = 0.0
    A = g.A
    for j in eachindex(g.alpha)
        a = g.alpha[j]; m1, m2 = g.m[j]
        e1, e2 = _reduce2(g, p[1] - g.x[j][1], p[2] - g.x[j][2])
        z = p[3] - g.x[j][3]
        for n2 in -m2:m2, n1 in -m1:m1
            r2 = (e1 - A[1, 1] * n1 - A[1, 2] * n2)^2 + (e2 - A[2, 1] * n1 - A[2, 2] * n2)^2 + z^2
            r2 * a < 46 && (s += g.c[j] * exp(-a * r2))
        end
    end
    return s
end

_erfc(x::Float64) = ccall((:erfc, Base.Math.libm), Float64, (Float64,), x)
_erf(x::Float64) = ccall((:erf, Base.Math.libm), Float64, (Float64,), x)

_dmin2(g::Q2DMixture) = minimum(1 / norm(view(g.Ainv, d, :)) for d in 1:2)

# Potential at p. Each charge's sum is independent of ξ, so a non-neutral
# mixture gets the gauge u + 2πQ|z|/|A| → 0 far from the slab.
function q2d_potential(g::Q2DMixture, p::AbstractVector; xi = 6.0 / _dmin2(g))
    A = g.A
    area = abs(det(A))
    B = 2pi * transpose(g.Ainv)                    # k̇ = B m
    rcut = 6.5 / xi
    n1i, n2i = _img_range(g.Ainv, rcut)
    kcut = 2 * xi * 6.5
    M1 = ceil(Int, kcut * norm(view(A, :, 1)) / 2pi); M2 = ceil(Int, kcut * norm(view(A, :, 2)) / 2pi)
    u = 0.0
    for j in eachindex(g.alpha)
        a = g.alpha[j]
        q = g.c[j] * (pi / a)^1.5
        dx, dy = _reduce2(g, p[1] - g.x[j][1], p[2] - g.x[j][2])
        z = p[3] - g.x[j][3]
        # real space
        for n2 in -n2i:n2i, n1 in -n1i:n1i
            r = sqrt((dx - A[1, 1] * n1 - A[1, 2] * n2)^2 + (dy - A[2, 1] * n1 - A[2, 2] * n2)^2 + z^2)
            u += q * (r < 1e-12 ? 2 * (sqrt(a) - xi) / sqrt(pi) : (_erfc(xi * r) - _erfc(sqrt(a) * r)) / r)
        end
        # k̇ = 0
        u -= q * 2 * sqrt(pi) / area * (exp(-xi^2 * z^2) / xi + sqrt(pi) * z * _erf(xi * z))
        # k̇ ≠ 0
        for m2 in -M2:M2, m1 in -M1:M1
            (m1 == 0 && m2 == 0) && continue
            k1 = B[1, 1] * m1 + B[1, 2] * m2; k2 = B[2, 1] * m1 + B[2, 2] * m2
            k = sqrt(k1^2 + k2^2)
            k > kcut && continue
            h = k / (2xi)
            v = exp(k * z) * _erfc(h + xi * z) + exp(-k * z) * _erfc(h - xi * z)
            u += q * pi / (area * k) * v * cos(k1 * dx + k2 * dy)
        end
    end
    return u
end
