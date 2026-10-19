# Wide (long-range) part of the periodic Coulomb potential. The Gaussians of
# the SOG wider than rc are summed in reciprocal space:
#
#   u_wide(x) = Σ_{m≠0} Ŵ(k) ρ̂(k) e^{ik·x},   k = 2π m ./ L,  |k| ≤ kmax,
#   Ŵ(k) = (1/V) Σ_wide w (πδ)^{3/2} exp(-δ|k|²/4),   ρ̂(k) = ∫_cell f e^{-ik·y} dy.
#
# The central level-0 boxes tile the cell exactly, and their proxy charges
# are the Legendre moments M_l = ∫_box f P_l(ξ) dy, so with
# e^{-iαξ} = Σ_l (2l+1)(-i)^l j_l(α) P_l(ξ) the transform from moments to
# modes is separable per axis, and so is the transform from modes back to the
# Legendre coefficients of u_wide on each central box (injected into the
# level-0 proxy potentials). Both are exact up to the truncation of the
# Legendre series at porder, whose terms decay like j_porder(kmax rc / 2).

# j_0..j_lmax at x ≥ 0 (Miller's downward recurrence, normalised by j_0 or j_1).
function sphbesselj!(j::AbstractVector{Float64}, lmax::Int, x::Float64)
    fill!(j, 0.0)
    if x == 0.0
        j[1] = 1.0
        return j
    end
    lm = max(lmax, 1)
    buf = zeros(lm + 1)
    a = 0.0; b = 1e-280          # j_{l+1}, j_l (unnormalised)
    for l in (lm + 30 + ceil(Int, 1.5x)):-1:1
        c = (2l + 1) / x * b - a
        a = b; b = c
        l - 1 <= lm && (buf[l] = c)
        if abs(b) > 1e250
            a *= 1e-250; b *= 1e-250
            for i in l:(lm + 1)
                buf[i] *= 1e-250
            end
        end
    end
    j0 = sin(x) / x
    j1 = sin(x) / x^2 - cos(x) / x
    sc = abs(j0) >= abs(j1) ? j0 / buf[1] : j1 / buf[2]
    for l in 0:lmax
        j[l + 1] = buf[l + 1] * sc
    end
    return j
end

struct WidePlan
    p::Int                          # Legendre order per axis (porder)
    Kh::NTuple{3,Int}               # mode half extents
    W::Array{Float64,3}             # Ŵ on the (2Kh+1)³ mode grid, zero outside |k| ≤ kmax and at k = 0
    F::NTuple{3,Matrix{ComplexF64}} # F[d][m, l + p s + 1] = (2l+1)(-i)^l j_l(k rc/2) e^{-ik c_s}
end

function WidePlan(g::PeriodicGrid, eps::Float64, p::Int, tsw::Vector{Float64}, wsw::Vector{Float64})
    rc = g.rc
    kmax = 2 * log(1 / eps) / rc
    Kh = ntuple(d -> ceil(Int, kmax * g.L[d] / 2pi), 3)
    V = prod(g.L)
    W = zeros(2Kh[1] + 1, 2Kh[2] + 1, 2Kh[3] + 1)
    for m3 in -Kh[3]:Kh[3], m2 in -Kh[2]:Kh[2], m1 in -Kh[1]:Kh[1]
        k2 = (2pi * m1 / g.L[1])^2 + (2pi * m2 / g.L[2])^2 + (2pi * m3 / g.L[3])^2
        (k2 == 0 || k2 > kmax^2) && continue
        acc = 0.0
        for i in eachindex(tsw)
            acc += wsw[i] * (pi * tsw[i])^1.5 * exp(-tsw[i] * k2 / 4)
        end
        W[m1 + Kh[1] + 1, m2 + Kh[2] + 1, m3 + Kh[3] + 1] = acc / V
    end
    jl = zeros(p)
    F = ntuple(3) do d
        Fd = zeros(ComplexF64, 2Kh[d] + 1, p * g.mgrid[d])
        for m in -Kh[d]:Kh[d]
            k = 2pi * m / g.L[d]
            sphbesselj!(jl, p - 1, abs(k) * rc / 2)
            for l in 0:(p - 1)
                # (-i)^l j_l(k rc/2) with j_l(-α) = (-1)^l j_l(α)
                a = (2l + 1) * jl[l + 1] * (k < 0 && isodd(l) ? -1 : 1) * (-im)^l
                for s in 0:(g.mgrid[d] - 1)
                    Fd[m + Kh[d] + 1, l + p * s + 1] = a * cis(-k * (s + 0.5) * rc)
                end
            end
        end
        Fd
    end
    return WidePlan(p, Kh, W, F)
end

# Legendre coefficients C (Q₁, Q₂, Q₃) of u_wide on the central boxes from
# their moments M (same layout, Q_d = p * mgrid[d], index l + p s).
function wide_coefficients(P::WidePlan, M::Array{Float64,3})
    Q1, Q2, Q3 = size(M)
    K1, K2, K3 = size(P.W)
    F1, F2, F3 = P.F
    # moments -> modes: S = (F1 ⊗ F2 ⊗ F3) M
    M2 = reshape(M, Q1, Q2 * Q3)
    A = complex.(real.(F1) * M2, imag.(F1) * M2)            # (K1, Q2 Q3)
    A3 = reshape(A, K1, Q2, Q3)
    B = Array{ComplexF64}(undef, K1, K2, Q3)
    F2t = transpose(F2)
    for q in 1:Q3
        mul!(view(B, :, :, q), view(A3, :, :, q), F2t)
    end
    S = reshape(reshape(B, K1 * K2, Q3) * transpose(F3), K1, K2, K3)
    S .*= P.W
    # modes -> coefficients: C = Re (F1ᴴ ⊗ F2ᴴ ⊗ F3ᴴ) S
    T1 = reshape(reshape(S, K1 * K2, K3) * conj.(F3), K1, K2, Q3)
    T2 = Array{ComplexF64}(undef, K1, Q2, Q3)
    F2c = conj.(F2)
    for q in 1:Q3
        mul!(view(T2, :, :, q), view(T1, :, :, q), F2c)
    end
    T22 = reshape(T2, K1, Q2 * Q3)
    C = transpose(real.(F1)) * real.(T22) .+ transpose(imag.(F1)) * imag.(T22)
    return reshape(C, Q1, Q2, Q3)
end
