# Wide (long-range) part of the periodic Coulomb potential. The Gaussians of
# the SOG wider than rc are summed in reciprocal space:
#
#   u_wide(x) = Σ_{m≠0} Ŵ(k) ρ̂(k) e^{ik·x},   k = B m,  B = 2π U⁻ᵀ,  |k| ≤ kmax,
#   Ŵ(k) = (1/V) Σ_wide w (πδ)^{3/2} exp(-δ|k|²/4),   ρ̂(k) = V ∫_{[0,1)³} f(U t) e^{-2πi m·t} dt.
#
# Source side: one period is split into mt[d] root cells per fractional axis,
# whose Legendre moments M_l = ∫_cell f P_l(ξ) du are known (u = mt ∘ t, so a
# root cell is a unit cube). With e^{-iαξ} = Σ_l (2l+1)(-i)^l j_l(α) P_l(ξ)
# the moments -> modes transform is separable per fractional axis and exact
# up to the truncation of the series. Target side: the Legendre coefficients
# of u_wide on each central level-0 cube (side rc, rotated frame) use the same
# expansion along the physical axes; k is lower triangular in m (k₁ depends on
# m₁, k₂ on m₁, m₂, ...), so the contraction runs m₃, m₂, m₁ with tables built
# per slice, and is a plain separable product for orthorhombic cells.

# j_0..j_lmax at x ≥ 0 into j (length ≥ max(lmax, 1) + 1): Miller's downward
# recurrence, normalised by j_0 or j_1.
function sphbesselj!(j::AbstractVector{Float64}, lmax::Int, x::Float64)
    lm = max(lmax, 1)
    @inbounds for i in 1:(lm + 1)
        j[i] = 0.0
    end
    if x == 0.0
        j[1] = 1.0
        return j
    end
    a = 0.0; b = 1e-280          # j_{l+1}, j_l (unnormalised)
    @inbounds for l in (lm + 30 + ceil(Int, 1.5x)):-1:1
        c = (2l + 1) / x * b - a
        a = b; b = c
        l - 1 <= lm && (j[l] = c)
        if abs(b) > 1e250
            a *= 1e-250; b *= 1e-250
            for i in l:(lm + 1)
                j[i] *= 1e-250
            end
        end
    end
    j0 = sin(x) / x
    j1 = sin(x) / x^2 - cos(x) / x
    sc = abs(j0) >= abs(j1) ? j0 / j[1] : j1 / j[2]
    @inbounds for i in 1:(lm + 1)
        j[i] *= sc
    end
    return j
end

# t[l+1] = (2l+1) (sgn i)^l j_l(κ h), l < p, for κ of either sign
# (j_l(-x) = (-1)^l j_l(x)); jl is scratch of length ≥ p + 1.
@inline function _legexp!(t::AbstractVector{ComplexF64}, jl::Vector{Float64}, p::Int, κ::Float64, h::Float64, sgn::Int)
    sphbesselj!(jl, p - 1, abs(κ) * h)
    ph = complex(1.0)
    @inbounds for l in 0:(p - 1)
        t[l + 1] = (2l + 1) * jl[l + 1] * (κ < 0 && isodd(l) ? -1 : 1) * ph
        ph *= sgn * im
    end
    return t
end

struct WidePlan
    pf::Int                         # source moments per axis (degree < pf)
    pe::Int                         # u_wide coefficients per axis (degree < pe)
    Kh::NTuple{3,Int}               # mode half extents
    mt::NTuple{3,Int}               # source root cells per fractional axis
    scale::Float64                  # V / prod(mt) (u-measure -> volume)
    W::Array{Float64,3}             # Ŵ on the (2Kh+1)³ grid of m, zero outside |k| ≤ kmax and at m = 0
    B::Matrix{Float64}              # k = B m
    F::NTuple{3,Matrix{ComplexF64}} # F[d][m, l + pf s + 1] = (2l+1)(-i)^l j_l(π m/mt) e^{-2πi m (s+½)/mt}
    E::Union{Nothing,NTuple{3,Matrix{ComplexF64}}}  # orthorhombic: E[d][m, l + pe s + 1] = (2l+1) i^l j_l(k rc/2) e^{ik c_s}
    rc::Float64
    xgrid0::NTuple{3,Float64}
    mgrid::NTuple{3,Int}
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

# Source root cells per fractional axis: edges |a_d| / mt[d] ≤ rc (C++ `mrc`);
# the central cubes themselves when they tile the cell.
source_cells(g::PeriodicGrid) = g.tiled ? g.mgrid :
    ntuple(d -> max(1, ceil(Int, norm(view(g.U, :, d)) / g.rc * (1 - 1e-12))), 3)

function WidePlan(g::PeriodicGrid, eps::Float64, pf::Int, pe::Int, tsw::Vector{Float64}, wsw::Vector{Float64})
    rc = g.rc
    kmax = 2 * log(1 / eps) / rc
    Kh = ntuple(d -> ceil(Int, kmax * norm(view(g.U, :, d)) / 2pi), 3)
    K = 2 .* Kh .+ 1
    B = 2pi * Matrix(transpose(g.Uinv))
    mt = source_cells(g)
    W = _wide_weights(g, B, Kh, kmax, tsw, wsw)
    jl = zeros(max(pf, pe) + 1)
    tl = zeros(ComplexF64, max(pf, pe))
    F = ntuple(3) do d
        Fd = zeros(ComplexF64, K[d], pf * mt[d])
        for m in -Kh[d]:Kh[d]
            _legexp!(tl, jl, pf, pi * m / mt[d], 1.0, -1)
            for s in 0:(mt[d] - 1)
                e = cis(-2pi * m * (s + 0.5) / mt[d])
                for l in 0:(pf - 1)
                    Fd[m + Kh[d] + 1, l + pf * s + 1] = tl[l + 1] * e
                end
            end
        end
        Fd
    end
    E = if g.diagonal
        ntuple(3) do d
            Ed = zeros(ComplexF64, K[d], pe * g.mgrid[d])
            for m in -Kh[d]:Kh[d]
                k = B[d, d] * m
                _legexp!(tl, jl, pe, k, rc / 2, 1)
                for s in 0:(g.mgrid[d] - 1)
                    e = cis(k * (g.xgrid0[d] + (s + 0.5) * rc))
                    for l in 0:(pe - 1)
                        Ed[m + Kh[d] + 1, l + pe * s + 1] = tl[l + 1] * e
                    end
                end
            end
            Ed
        end
    else
        nothing
    end
    return WidePlan(pf, pe, Kh, mt, g.V / prod(mt), W, B, F, E, rc, g.xgrid0, g.mgrid)
end

# Ŵ on the mode grid. Orthorhombic: Σ_s c_s e1_s ⊗ e2_s ⊗ e3_s, one GEMM per
# z-slice; otherwise per mode, summing the Gaussians by increasing δ until
# exp(-δ|k|²/4) is negligible.
function _wide_weights(g::PeriodicGrid, B::Matrix{Float64}, Kh, kmax::Float64, tsw::Vector{Float64}, wsw::Vector{Float64})
    K = 2 .* Kh .+ 1
    ns = length(tsw)
    c = [wsw[s] * (pi * tsw[s])^1.5 / g.V for s in 1:ns]
    W = zeros(K...)
    kvec(m1, m2, m3) = (B[1, 1] * m1, B[2, 1] * m1 + B[2, 2] * m2, B[3, 1] * m1 + B[3, 2] * m2 + B[3, 3] * m3)
    if g.diagonal
        kv = ntuple(d -> [B[d, d] * m for m in -Kh[d]:Kh[d]], 3)
        E1 = [exp(-tsw[s] * kv[1][m]^2 / 4) for m in 1:K[1], s in 1:ns]
        E2c = [c[s] * exp(-tsw[s] * kv[2][m]^2 / 4) for s in 1:ns, m in 1:K[2]]
        Bm = similar(E2c)
        for m3 in 1:K[3]
            for m in 1:K[2], s in 1:ns
                Bm[s, m] = E2c[s, m] * exp(-tsw[s] * kv[3][m3]^2 / 4)
            end
            mul!(view(W, :, :, m3), E1, Bm)
        end
    else
        ord = sortperm(tsw)
        parallel_foreach(collect(1:K[3]), fill(nothing, _nthreads())) do i3, _
            m3 = i3 - Kh[3] - 1
            for i2 in 1:K[2], i1 in 1:K[1]
                k = kvec(i1 - Kh[1] - 1, i2 - Kh[2] - 1, m3)
                q = k[1]^2 + k[2]^2 + k[3]^2
                (q == 0 || q > kmax^2) && continue
                acc = 0.0
                for s in ord
                    x = tsw[s] * q / 4
                    x > 60 && break
                    acc += c[s] * exp(-x)
                end
                W[i1, i2, i3] = acc
            end
        end
    end
    for i3 in 1:K[3], i2 in 1:K[2], i1 in 1:K[1]
        k = kvec(i1 - Kh[1] - 1, i2 - Kh[2] - 1, i3 - Kh[3] - 1)
        q = k[1]^2 + k[2]^2 + k[3]^2
        (q == 0 || q > kmax^2) && (W[i1, i2, i3] = 0.0)
    end
    return W
end

# Modes G = Ŵ ρ̂ (K₁, K₂, K₃) from the root-cell moments M (pf mt₁, pf mt₂,
# pf mt₃), index l + pf s per axis, u-measure. Each axis is one large GEMM
# (the middle one after a permutation); buffers come from the arena `ar`.
function wide_modes(P::WidePlan, M::Array{Float64,3}, ar::Arena)
    Q1, Q2, Q3 = size(M)
    K1, K2, K3 = size(P.W)
    F1, F2, F3 = P.F
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
    S = allocc(ar, K2, K1, K3)
    mul!(reshape(S, K2 * K1, K3), reshape(X2, K2 * K1, Q3), transpose(F3))
    G = allocc(ar, K1, K2, K3)
    W = P.W
    sc = P.scale
    @inbounds for m3 in 1:K3, m2 in 1:K2, m1 in 1:K1
        G[m1, m2, m3] = sc * W[m1, m2, m3] * S[m2, m1, m3]
    end
    return G
end

# Legendre coefficients C (pe mgrid₁, pe mgrid₂, pe mgrid₃) of u_wide on the
# central level-0 cubes from the modes G.
wide_coefficients(P::WidePlan, G::Array{ComplexF64,3}, ar::Arena) =
    P.E === nothing ? _wide_coefs_triangular(P, G, ar) : _wide_coefs_separable(P, G, ar)

function _wide_coefs_separable(P::WidePlan, G::Array{ComplexF64,3}, ar::Arena)
    K1, K2, K3 = size(G)
    E1, E2, E3 = P.E
    R1, R2, R3 = size(E1, 2), size(E2, 2), size(E3, 2)
    Y1 = allocc(ar, K1, K2, R3)
    mul!(reshape(Y1, K1 * K2, R3), reshape(G, K1 * K2, K3), E3)
    Y1p = allocc(ar, K2, K1, R3)
    permutedims!(Y1p, Y1, (2, 1, 3))
    Y2 = allocc(ar, R2, K1, R3)
    mul!(reshape(Y2, R2, K1 * R3), transpose(E2), reshape(Y1p, K2, K1 * R3))
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

# Triclinic: per m₁ slice, m₃ (tables per mode, k₃ depends on all of m) then
# m₂ (tables per (m₁, m₂)), then one GEMM over m₁.
function _wide_coefs_triangular(P::WidePlan, G::Array{ComplexF64,3}, ar::Arena)
    K1, K2, K3 = size(G)
    Kh = P.Kh; B = P.B; pe = P.pe; rc = P.rc
    R1, R2, R3 = pe .* P.mgrid
    cs = ntuple(d -> [P.xgrid0[d] + (s + 0.5) * rc for s in 0:(P.mgrid[d] - 1)], 3)
    A2r = alloc(ar, R2, R3, K1); A2i = alloc(ar, R2, R3, K1)
    wks = make_workspaces(() -> (A1r = zeros(K2, R3), A1i = zeros(K2, R3), E2r = zeros(R2, K2), E2i = zeros(R2, K2),
                                 jl = zeros(pe + 1), tl = zeros(ComplexF64, pe)))
    parallel_foreach(collect(1:K1), wks) do i1, w
        m1 = i1 - Kh[1] - 1
        fill!(w.A1r, 0.0); fill!(w.A1i, 0.0)
        any2 = false
        for i2 in 1:K2
            m2 = i2 - Kh[2] - 1
            for i3 in 1:K3
                gv = G[i1, i2, i3]
                gv == 0 && continue
                any2 = true
                κ = B[3, 1] * m1 + B[3, 2] * m2 + B[3, 3] * (i3 - Kh[3] - 1)
                _legexp!(w.tl, w.jl, pe, κ, rc / 2, 1)
                for s in 0:(P.mgrid[3] - 1)
                    e = gv * cis(κ * cs[3][s + 1])
                    @inbounds for l in 1:pe
                        v = e * w.tl[l]
                        w.A1r[i2, l + pe * s] += real(v)
                        w.A1i[i2, l + pe * s] += imag(v)
                    end
                end
            end
        end
        if !any2
            fill!(view(A2r, :, :, i1), 0.0); fill!(view(A2i, :, :, i1), 0.0)
            return
        end
        for i2 in 1:K2
            κ = B[2, 1] * m1 + B[2, 2] * (i2 - Kh[2] - 1)
            _legexp!(w.tl, w.jl, pe, κ, rc / 2, 1)
            for s in 0:(P.mgrid[2] - 1)
                e = cis(κ * cs[2][s + 1])
                @inbounds for l in 1:pe
                    v = e * w.tl[l]
                    w.E2r[l + pe * s, i2] = real(v)
                    w.E2i[l + pe * s, i2] = imag(v)
                end
            end
        end
        # A2 = E2 A1 (complex, as four real products)
        GC.@preserve A2r A2i w begin
            pr = pointer(A2r, 1 + R2 * R3 * (i1 - 1)); pim = pointer(A2i, 1 + R2 * R3 * (i1 - 1))
            gemm!(pr, R2, pointer(w.E2r), R2, pointer(w.A1r), K2, R2, R3, K2, 1.0, false)
            gemm!(pr, R2, pointer(w.E2i), R2, pointer(w.A1i), K2, R2, R3, K2, -1.0, true)
            gemm!(pim, R2, pointer(w.E2r), R2, pointer(w.A1i), K2, R2, R3, K2, 1.0, false)
            gemm!(pim, R2, pointer(w.E2i), R2, pointer(w.A1r), K2, R2, R3, K2, 1.0, true)
        end
    end
    E1r = zeros(R1, K1); E1i = zeros(R1, K1)
    jl = zeros(pe + 1); tl = zeros(ComplexF64, pe)
    for i1 in 1:K1
        κ = B[1, 1] * (i1 - Kh[1] - 1)
        _legexp!(tl, jl, pe, κ, rc / 2, 1)
        for s in 0:(P.mgrid[1] - 1)
            e = cis(κ * cs[1][s + 1])
            for l in 1:pe
                v = e * tl[l]
                E1r[l + pe * s, i1] = real(v); E1i[l + pe * s, i1] = imag(v)
            end
        end
    end
    C = alloc(ar, R1, R2, R3)
    C2 = reshape(C, R1, R2 * R3)
    mul!(C2, E1r, transpose(reshape(A2r, R2 * R3, K1)))
    mul!(C2, E1i, transpose(reshape(A2i, R2 * R3, K1)), -1.0, 1.0)
    return C
end
