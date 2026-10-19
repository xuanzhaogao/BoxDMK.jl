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
    pf::Int                         # moments kept per axis (degree < pf)
    pe::Int                         # u_wide coefficients computed per axis (degree < pe)
    Kh::NTuple{3,Int}               # mode half extents
    W::Array{Float64,3}             # Ŵ on the (2Kh+1)³ mode grid, zero outside |k| ≤ kmax and at k = 0
    F::NTuple{3,Matrix{ComplexF64}} # moments -> modes: F[d][m, l + pf s + 1] = (2l+1)(-i)^l j_l(k rc/2) e^{-ik c_s}
    E::NTuple{3,Matrix{ComplexF64}} # modes -> coefficients: same with pe, conjugated
end

# Truncation orders (C++ `p_wide` and `wide_eval_order`): the moments of
# degree ≥ 1.25 ln(1/ε) reach the kept band only through j_l(k rc/2) weighted
# by the decay of Ŵ, and u_wide on an rc box is resolved at degree
# kmax rc / 1.7 + 4.
function wide_orders(eps::Float64, porder::Int, norder::Int, rc::Float64)
    kmax = 2 * log(1 / eps) / rc
    pf = min(porder, round(Int, 1.25 * log(1 / eps)))
    pe = min(porder, max(norder, ceil(Int, kmax * rc / 1.7) + 4))
    return pf, pe
end

function WidePlan(g::PeriodicGrid, eps::Float64, pf::Int, pe::Int, tsw::Vector{Float64}, wsw::Vector{Float64})
    rc = g.rc
    kmax = 2 * log(1 / eps) / rc
    Kh = ntuple(d -> ceil(Int, kmax * g.L[d] / 2pi), 3)
    K = 2 .* Kh .+ 1
    # Ŵ = Σ_s c_s e1_s ⊗ e2_s ⊗ e3_s with e_d,s[m] = exp(-δ_s k_m²/4)
    ns = length(tsw)
    kv = ntuple(d -> [2pi * m / g.L[d] for m in -Kh[d]:Kh[d]], 3)
    E1 = [exp(-tsw[s] * kv[1][m]^2 / 4) for m in 1:K[1], s in 1:ns]
    E2c = [wsw[s] * (pi * tsw[s])^1.5 / prod(g.L) * exp(-tsw[s] * kv[2][m]^2 / 4) for s in 1:ns, m in 1:K[2]]
    W = zeros(K...)
    B = similar(E2c)
    for m3 in 1:K[3]
        for m in 1:K[2], s in 1:ns
            B[s, m] = E2c[s, m] * exp(-tsw[s] * kv[3][m3]^2 / 4)
        end
        mul!(view(W, :, :, m3), E1, B)
    end
    for m3 in 1:K[3], m2 in 1:K[2], m1 in 1:K[1]
        k2 = kv[1][m1]^2 + kv[2][m2]^2 + kv[3][m3]^2
        (k2 == 0 || k2 > kmax^2) && (W[m1, m2, m3] = 0.0)
    end
    pmax = max(pf, pe)
    jl = zeros(pmax)
    tabs = ntuple(3) do d
        Fd = zeros(ComplexF64, K[d], pf * g.mgrid[d])
        Ed = zeros(ComplexF64, K[d], pe * g.mgrid[d])
        for m in -Kh[d]:Kh[d]
            k = kv[d][m + Kh[d] + 1]
            sphbesselj!(jl, pmax - 1, abs(k) * rc / 2)
            for l in 0:(pmax - 1)
                # (-i)^l j_l(k rc/2) with j_l(-α) = (-1)^l j_l(α)
                a = (2l + 1) * jl[l + 1] * (k < 0 && isodd(l) ? -1 : 1) * (-im)^l
                for s in 0:(g.mgrid[d] - 1)
                    v = a * cis(-k * (s + 0.5) * rc)
                    l < pf && (Fd[m + Kh[d] + 1, l + pf * s + 1] = v)
                    l < pe && (Ed[m + Kh[d] + 1, l + pe * s + 1] = conj(v))
                end
            end
        end
        (Fd, Ed)
    end
    return WidePlan(pf, pe, Kh, W, ntuple(d -> tabs[d][1], 3), ntuple(d -> tabs[d][2], 3))
end

# Legendre coefficients C (pe mgrid₁, pe mgrid₂, pe mgrid₃) of u_wide on the
# central boxes from their moments M (pf mgrid₁, pf mgrid₂, pf mgrid₃), index
# l + p s along each axis. Each axis is one large GEMM (the middle one after a
# permutation); buffers come from the arena `ar`.
function wide_coefficients(P::WidePlan, M::Array{Float64,3}, ar::Arena)
    Q1, Q2, Q3 = size(M)
    K1, K2, K3 = size(P.W)
    F1, F2, F3 = P.F
    E1, E2, E3 = P.E
    R1, R2, R3 = size(E1, 2), size(E2, 2), size(E3, 2)
    # moments -> modes: S = (F1 ⊗ F2 ⊗ F3) M, in (m2, m1, m3) order
    M2 = reshape(M, Q1, Q2 * Q3)
    Xr = alloc(ar, K1, Q2, Q3); Xi = alloc(ar, K1, Q2, Q3)
    mul!(reshape(Xr, K1, Q2 * Q3), real.(F1), M2)
    mul!(reshape(Xi, K1, Q2 * Q3), imag.(F1), M2)
    X1 = allocc(ar, Q2, K1, Q3)
    @inbounds for q3 in 1:Q3, k1 in 1:K1, q2 in 1:Q2
        X1[q2, k1, q3] = complex(Xr[k1, q2, q3], Xi[k1, q2, q3])
    end
    X2 = allocc(ar, K2, K1, Q3)
    mul!(reshape(X2, K2, K1 * Q3), F2, reshape(X1, Q2, K1 * Q3))
    S = allocc(ar, K2 * K1, K3)
    mul!(S, reshape(X2, K2 * K1, Q3), transpose(F3))
    W = P.W
    @inbounds for m3 in 1:K3, m1 in 1:K1, m2 in 1:K2
        S[m2 + K2 * (m1 - 1), m3] *= W[m1, m2, m3]
    end
    # modes -> coefficients: C = Re (E1ᵀ ⊗ E2ᵀ ⊗ E3ᵀ) S
    Y1 = allocc(ar, K2, K1, R3)
    mul!(reshape(Y1, K2 * K1, R3), S, E3)
    Y2 = allocc(ar, R2, K1, R3)
    mul!(reshape(Y2, R2, K1 * R3), transpose(E2), reshape(Y1, K2, K1 * R3))
    Yr = alloc(ar, K1, R2, R3); Yi = alloc(ar, K1, R2, R3)
    @inbounds for r3 in 1:R3, k1 in 1:K1, r2 in 1:R2
        y = Y2[r2, k1, r3]
        Yr[k1, r2, r3] = real(y); Yi[k1, r2, r3] = imag(y)
    end
    C = alloc(ar, R1, R2, R3)
    C2 = reshape(C, R1, R2 * R3)
    mul!(C2, transpose(real.(E1)), reshape(Yr, K1, R2 * R3))
    mul!(C2, transpose(imag.(E1)), reshape(Yi, K1, R2 * R3), -1.0, 1.0)
    return C
end
