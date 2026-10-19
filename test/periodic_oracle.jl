# Analytic periodic Coulomb potential of a periodized Gaussian mixture
# f(x) = Σ_j c_j Σ_n exp(-α_j |x - x_j - n∘L|²) (k = 0 mode dropped), by
# Ewald summation with splitting parameter β.

struct GaussMixture
    L::NTuple{3,Float64}
    x::Vector{NTuple{3,Float64}}
    alpha::Vector{Float64}
    c::Vector{Float64}
end

# Mixture with the last coefficient set for charge neutrality.
function neutral_mixture(L, x, alpha, c)
    c = copy(c)
    q = c .* (pi ./ alpha) .^ 1.5
    c[end] = -sum(q[1:end-1]) / (pi / alpha[end])^1.5
    return GaussMixture(L, x, alpha, c)
end

function (g::GaussMixture)(p::AbstractVector)
    s = 0.0
    for j in eachindex(g.alpha)
        a = g.alpha[j]
        prod_ = 1.0
        for d in 1:3
            dx = p[d] - g.x[j][d]
            dx -= g.L[d] * round(dx / g.L[d])
            acc = 0.0
            for n in -2:2
                acc += exp(-a * (dx - n * g.L[d])^2)
            end
            prod_ *= acc
        end
        s += g.c[j] * prod_
    end
    return s
end

function ewald_potential(g::GaussMixture, p::AbstractVector; beta = 40.0 / minimum(g.L)^2, nimg = 3, mmax = 20)
    L = g.L
    V = prod(L)
    u = 0.0
    E = [zeros(ComplexF64, 2mmax + 1) for _ in 1:3]
    for j in eachindex(g.alpha)
        a = g.alpha[j]
        q = g.c[j] * (pi / a)^1.5
        gam = a * beta / (a + beta)
        dx = ntuple(d -> p[d] - g.x[j][d], 3)
        dx = ntuple(d -> dx[d] - L[d] * round(dx[d] / L[d]), 3)
        # real space
        for n3 in -nimg:nimg, n2 in -nimg:nimg, n1 in -nimg:nimg
            r = sqrt((dx[1] - n1 * L[1])^2 + (dx[2] - n2 * L[2])^2 + (dx[3] - n3 * L[3])^2)
            u += q * (r < 1e-12 ? 2 * (sqrt(a) - sqrt(gam)) / sqrt(pi) :
                      (erfc_(sqrt(gam) * r) - erfc_(sqrt(a) * r)) / r)
        end
        u -= q * pi / (beta * V)
        # reciprocal space
        for d in 1:3, m in -mmax:mmax
            E[d][m + mmax + 1] = cis(2pi * m / L[d] * dx[d])
        end
        for m3 in -mmax:mmax, m2 in -mmax:mmax
            k23 = (2pi * m2 / L[2])^2 + (2pi * m3 / L[3])^2
            e23 = E[2][m2 + mmax + 1] * E[3][m3 + mmax + 1]
            for m1 in -mmax:mmax
                k2 = (2pi * m1 / L[1])^2 + k23
                k2 == 0 && continue
                u += q * 4pi / V * exp(-k2 / (4gam)) / k2 * real(E[1][m1 + mmax + 1] * e23)
            end
        end
    end
    return u
end

erfc_(x::Float64) = ccall((:erfc, Base.Math.libm), Float64, (Float64,), x)
