# Quasi-2D (periodic in x, y with the rectangle L₁ × L₂, free in z) Coulomb
# potential of a neutral Gaussian mixture f(x) = Σ_j c_j Σ_n exp(-α_j |x - x_j - n∘L|²),
# n = (n₁, n₂, 0), by 2-periodic Ewald summation with parameter ξ < √α_j:
#   erf(√α r)/r = (erfc(ξr) - erfc(√α r))/r [real space] + erf(ξr)/r [Fourier, in plane],
# the k̇ = 0 term of the latter -(2√π/A)(e^{-ξ²z²}/ξ + √π z erf(ξz)) (Lindbo & Tornberg 2012).

struct Q2DMixture
    L::NTuple{2,Float64}
    x::Vector{NTuple{3,Float64}}
    alpha::Vector{Float64}
    c::Vector{Float64}
    m::Vector{Int}                      # in-plane image half range for the density
end

# Mixture with the last coefficient set for charge neutrality.
function neutral_q2d_mixture(L, x, alpha, c)
    c = collect(Float64, c)
    q = c .* (pi ./ alpha) .^ 1.5
    c[end] = -sum(q[1:end-1]) / (pi / alpha[end])^1.5
    m = [max(1, ceil(Int, sqrt(46 / a) / minimum(L) + 0.5)) for a in alpha]
    return Q2DMixture((Float64(L[1]), Float64(L[2])), x, collect(Float64, alpha), c, m)
end

function (g::Q2DMixture)(p::AbstractVector)
    s = 0.0
    for j in eachindex(g.alpha)
        a = g.alpha[j]; m = g.m[j]
        d = (p[1] - g.x[j][1], p[2] - g.x[j][2], p[3] - g.x[j][3])
        t1 = round(d[1] / g.L[1]); t2 = round(d[2] / g.L[2])
        for n2 in -m:m, n1 in -m:m
            e1 = d[1] - (t1 + n1) * g.L[1]; e2 = d[2] - (t2 + n2) * g.L[2]
            r2 = e1^2 + e2^2 + d[3]^2
            r2 * a < 46 && (s += g.c[j] * exp(-a * r2))
        end
    end
    return s
end

_erfc(x::Float64) = ccall((:erfc, Base.Math.libm), Float64, (Float64,), x)
_erf(x::Float64) = ccall((:erf, Base.Math.libm), Float64, (Float64,), x)

function q2d_potential(g::Q2DMixture, p::AbstractVector; xi = 6.0 / minimum(g.L))
    L1, L2 = g.L
    A = L1 * L2
    u = 0.0
    for j in eachindex(g.alpha)
        a = g.alpha[j]
        q = g.c[j] * (pi / a)^1.5
        dx = p[1] - g.x[j][1]; dy = p[2] - g.x[j][2]; z = p[3] - g.x[j][3]
        dx -= L1 * round(dx / L1); dy -= L2 * round(dy / L2)
        # real space
        nimg = ceil(Int, 6.5 / xi / minimum(g.L)) + 1
        for n2 in -nimg:nimg, n1 in -nimg:nimg
            r = sqrt((dx - n1 * L1)^2 + (dy - n2 * L2)^2 + z^2)
            u += q * (r < 1e-12 ? 2 * (sqrt(a) - xi) / sqrt(pi) : (_erfc(xi * r) - _erfc(sqrt(a) * r)) / r)
        end
        # k̇ = 0
        u -= q * 2 * sqrt(pi) / A * (exp(-xi^2 * z^2) / xi + sqrt(pi) * z * _erf(xi * z))
        # k̇ ≠ 0
        kcut = 2 * xi * 6.5
        M1 = ceil(Int, kcut * L1 / 2pi); M2 = ceil(Int, kcut * L2 / 2pi)
        for m2 in -M2:M2, m1 in -M1:M1
            (m1 == 0 && m2 == 0) && continue
            k1 = 2pi * m1 / L1; k2 = 2pi * m2 / L2
            k = sqrt(k1^2 + k2^2)
            k > kcut && continue
            h = k / (2xi)
            v = exp(k * z) * _erfc(h + xi * z) + exp(-k * z) * _erfc(h - xi * z)
            u += q * pi / (A * k) * v * cos(k1 * dx + k2 * dy)
        end
    end
    return u
end
