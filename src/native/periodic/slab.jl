# Wide part of the quasi-2D (slab) potential: periodic in x, y, free in z.
# Poisson summation over the in-plane lattice turns each wide Gaussian into
#
#   Σ_n e^{-|x - y - An|²/δ} = (πδ/A) Σ_k̇ e^{-δ|k̇|²/4} e^{ik̇·(ṙ - ṙ')} e^{-(z - z')²/δ},
#
# A = L₁L₂ the in-plane area, k̇ = (2πm₁/L₁, 2πm₂/L₂). In the plane the
# transforms are those of the periodic solver (exact Legendre-Bessel per axis,
# half space m₁ ≥ 0). In z the kernel is applied directly: on the level-0
# z-cells (side rc) every wide Gaussian is smooth, so its z-convolution is a
# small exact matrix in the Legendre basis of the cells (no Fourier transform,
# no padding). The mode k̇ = 0 gets the kernel of all wide Gaussians,
#
#   D(z) = Σ_wide (πwδ/A) (e^{-z²/δ} - 1)  →  -2π|z|/A  (far tail in closed form),
#
# where the constants dropped vanish for a neutral density.

struct SlabWide
    pf::Int                          # source moments per axis (degree < pf)
    pe::Int                          # u_wide coefficients per axis (degree < pe)
    Kh::NTuple{2,Int}                # in-plane mode half extents
    F1::NTuple{2,Matrix{Float64}}    # as WidePlan: rows m₁ ≥ 0, (Kh₁+1) × pf mgrid₁
    F2t::NTuple{2,Matrix{Float64}}   # pf mgrid₂ × K₂
    E1::NTuple{2,Matrix{Float64}}    # (Kh₁+1) × pe mgrid₁, half-space weights
    E2::NTuple{2,Matrix{Float64}}    # K₂ × pe mgrid₂
    GzT::Vector{Matrix{Float64}}     # per wide Gaussian: transpose of its z matrix (pf mz × pe mz)
    cm::Matrix{Float64}              # (Kh₁+1)K₂ × #Gaussians: rc³ (πwδ/A) e^{-δ|k̇|²/4} (0 at k̇ = 0, beyond kmax)
    D0T::Matrix{Float64}             # transpose of the k̇ = 0 z matrix (times rc³)
    mgrid::NTuple{3,Int}
end

# z matrix of the kernel K(z - z') between the level-0 z-cells: entry
# [(l, i), (l', j)] = ((2l+1)/2)((2l'+1)/2) ∫∫ K(z_i(ξ) - z_j(ξ')) P_l(ξ) P_l'(ξ') dξ dξ',
# so that the Legendre coefficients (degree < pe) on cell i of ∫ ρ(z') K(z - z') dz'
# follow from the moments ∫_cell j ρ P_l' dz' (degree < pf).
function _zkernel_matrix(K, g::PeriodicGrid, pe::Int, pf::Int)
    mz, rc, z0 = g.mgrid[3], g.rc, g.xgrid0[3]
    q = max(pe, pf) + 24
    leg = ortho1d(LegendreBasis(), q)
    ξ, w = leg.x, leg.w
    pv = zeros(max(pe, pf))
    Pe = zeros(pe, q); Pf = zeros(pf, q)
    for a in 1:q
        legepols!(pv, ξ[a], max(pe, pf))
        for l in 1:pe
            Pe[l, a] = (2l - 1) / 2 * w[a] * pv[l]
        end
        for l in 1:pf
            Pf[l, a] = (2l - 1) / 2 * w[a] * pv[l]
        end
    end
    G = zeros(pe * mz, pf * mz)
    Kq = zeros(q, q)
    for j in 0:(mz - 1), i in 0:(mz - 1)
        for b in 1:q, a in 1:q
            Kq[a, b] = K((i - j) * rc + (ξ[a] - ξ[b]) * rc / 2)
        end
        G[pe * i .+ (1:pe), pf * j .+ (1:pf)] .= Pe * Kq * transpose(Pf)
    end
    return G
end

function SlabWide(g::PeriodicGrid, eps::Float64, pf::Int, pe::Int, tsw::Vector{Float64}, wsw::Vector{Float64})
    rc = g.rc
    L = (g.U[1, 1], g.U[2, 2])
    area = L[1] * L[2]
    kmax = 2 * log(1 / eps) / rc
    Kh = ntuple(d -> ceil(Int, kmax * L[d] / 2pi), 2)
    mt = g.mgrid
    B = (2pi / L[1], 2pi / L[2])
    F(d, ms) = _legtable(ms, pf, mt[d], m -> pi * m / mt[d], 1.0, -1, (m, s) -> -2pi * m * (s + 0.5) / mt[d])
    E(d, ms, wt = m -> 1.0) = _legtable(ms, pe, mt[d], m -> B[d] * m, rc / 2, 1,
                                        (m, s) -> B[d] * m * (g.xgrid0[d] + (s + 0.5) * rc), wt)
    full = (-Kh[2]):Kh[2]
    H1, K2 = Kh[1] + 1, 2Kh[2] + 1
    sc = rc^3                                   # u-measure moments -> volume
    # Gaussians that reach a mode k̇ ≠ 0, with their per-mode weights
    kmin2 = minimum(B)^2
    keep = [s for s in eachindex(tsw) if wsw[s] * pi * tsw[s] * exp(-tsw[s] * kmin2 / 4) / area > 1e-18 * maximum(wsw) * rc]
    cm = zeros(H1 * K2, length(keep))
    for (c, s) in enumerate(keep), i2 in 1:K2, i1 in 1:H1
        k2 = (B[1] * (i1 - 1))^2 + (B[2] * (i2 - Kh[2] - 1))^2
        (k2 == 0 || k2 > kmax^2) && continue
        cm[i1 + H1 * (i2 - 1), c] = sc * wsw[s] * pi * tsw[s] / area * exp(-tsw[s] * k2 / 4)
    end
    GzT = [Matrix(transpose(_zkernel_matrix(z -> exp(-z^2 / tsw[s]), g, pe, pf))) for s in keep]
    # k̇ = 0: all wide Gaussians, (e^{-z²/δ} - 1); beyond the widest, -z² Σ w (geometric tail)
    ord = sortperm(tsw)
    b = sqrt(tsw[ord[end]] / tsw[ord[end - 1]])
    wtail = wsw[ord[end]] / (b - 1)
    D(z) = (sum(wsw[s] * pi * tsw[s] * expm1(-z^2 / tsw[s]) for s in eachindex(tsw)) - pi * wtail * z^2) / area
    D0T = Matrix(transpose(sc .* _zkernel_matrix(D, g, pe, pf)))
    return SlabWide(pf, pe, Kh, F(1, 0:Kh[1]), _tr(F(2, full)), E(1, 0:Kh[1], m -> m == 0 ? 1.0 : 2.0),
                    E(2, full), GzT, cm, D0T, mt)
end

"""
    slab_coefficients(S::SlabWide, M, ar) -> Ct

Legendre coefficients of the wide slab potential on the level-0 cubes,
`Ct[l₃ + pe s₃, l₂ + pe s₂, l₁ + pe s₁]` (as `wide_coefficients`), from the
level-0 moments `M (pf mgrid₁, pf mgrid₂, pf mgrid₃)` (u-measure).
"""
function slab_coefficients(S::SlabWide, M::Array{Float64,3}, ar::Arena)
    Q1, Q2, Q3 = size(M)
    H1, K2 = S.Kh[1] + 1, 2S.Kh[2] + 1
    nm = H1 * K2
    (F1r, F1i), (F2r, F2i) = S.F1, S.F2t
    # in plane, per z moment q₃: Z[:, :, q₃] = F₁ M[:, :, q₃] F₂ᵀ (as wide_modes)
    Zr = alloc(ar, H1, K2, Q3); Zi = alloc(ar, H1, K2, Q3)
    wks = make_workspaces(() -> (alloc(ar, H1, Q2), alloc(ar, H1, Q2)))
    parallel_foreach(collect(1:Q3), wks) do q3, (Yr, Yi)
        GC.@preserve M Zr Zi Yr Yi F1r F1i F2r F2i begin
            pm = pointer(M) + 8 * Q1 * Q2 * (q3 - 1)
            o = 8 * nm * (q3 - 1)
            gemm!(pointer(Yr), H1, pointer(F1r), H1, pm, Q1, H1, Q2, Q1, 1.0, false)
            gemm!(pointer(Yi), H1, pointer(F1i), H1, pm, Q1, H1, Q2, Q1, 1.0, false)
            _zgemm!(pointer(Zr) + o, pointer(Zi) + o, H1, pointer(Yr), pointer(Yi), H1,
                    pointer(F2r), pointer(F2i), Q2, H1, K2, Q2, false)
        end
    end
    # in z, per mode: Out = Σ_g (c_g ⊙ Z) G_gᵀ, plus the k̇ = 0 kernel (rows: modes)
    R3 = S.pe * S.mgrid[3]
    Or = alloc(ar, nm, R3); Oi = alloc(ar, nm, R3)
    blk = max(16, 16 * cld(cld(nm, 4 * _nthreads()), 16))
    wsc = make_workspaces(() -> (alloc(ar, blk, Q3), alloc(ar, blk, Q3)))
    parallel_foreach(collect(0:blk:(nm - 1)), wsc) do r0, (Tr, Ti)
        len = min(blk, nm - r0)
        GC.@preserve Zr Zi Or Oi Tr Ti S begin
            for c in eachindex(S.GzT)
                cmc = view(S.cm, :, c)
                @inbounds for q in 1:Q3, r in 1:len
                    w = cmc[r0 + r]
                    Tr[r + blk * (q - 1)] = w * Zr[r0 + r + nm * (q - 1)]
                    Ti[r + blk * (q - 1)] = w * Zi[r0 + r + nm * (q - 1)]
                end
                G = S.GzT[c]
                gemm!(pointer(Or) + 8r0, nm, pointer(Tr), blk, pointer(G), Q3, len, R3, Q3, 1.0, c > 1)
                gemm!(pointer(Oi) + 8r0, nm, pointer(Ti), blk, pointer(G), Q3, len, R3, Q3, 1.0, c > 1)
            end
            if isempty(S.GzT)
                fill!(view(Or, (r0 + 1):(r0 + len), :), 0.0); fill!(view(Oi, (r0 + 1):(r0 + len), :), 0.0)
            end
        end
    end
    r0 = 1 + H1 * S.Kh[2]                          # k̇ = 0 (m₁ = 0, m₂ = 0)
    GC.@preserve Zr Zi Or Oi S begin
        gemm!(pointer(Or) + 8(r0 - 1), nm, pointer(Zr) + 8(r0 - 1), nm, pointer(S.D0T), Q3, 1, R3, Q3, 1.0, true)
        gemm!(pointer(Oi) + 8(r0 - 1), nm, pointer(Zi) + 8(r0 - 1), nm, pointer(S.D0T), Q3, 1, R3, Q3, 1.0, true)
    end
    # back in plane: Y₁[r₃, m₂, m₁] = Out, then E₂ per m₁ and E₁ (as the orthorhombic case)
    Y1r = alloc(ar, R3, K2, H1); Y1i = alloc(ar, R3, K2, H1)
    permutedims!(Y1r, reshape(Or, H1, K2, R3), (3, 2, 1)); permutedims!(Y1i, reshape(Oi, H1, K2, R3), (3, 2, 1))
    (E2r, E2i) = S.E2
    R2 = size(E2r, 2)
    Y2r = alloc(ar, R3, R2, H1); Y2i = alloc(ar, R3, R2, H1)
    parallel_foreach(collect(1:H1), fill(nothing, _nthreads())) do i1, _
        GC.@preserve Y1r Y1i Y2r Y2i E2r E2i begin
            o1 = 8 * R3 * K2 * (i1 - 1); o2 = 8 * R3 * R2 * (i1 - 1)
            _zgemm!(pointer(Y2r) + o2, pointer(Y2i) + o2, R3, pointer(Y1r) + o1, pointer(Y1i) + o1, R3,
                    pointer(E2r), pointer(E2i), K2, R3, R2, K2, false)
        end
    end
    return _wide_coefs_final(S, Y2r, Y2i, ar)
end

# Total charge of the level-0 moments (degree-0 entries, u-measure × rc³).
_slab_charge(S::SlabWide, M::Array{Float64,3}, rc::Float64) =
    rc^3 * sum(M[1 + S.pf * s1, 1 + S.pf * s2, 1 + S.pf * s3]
               for s1 in 0:(S.mgrid[1] - 1), s2 in 0:(S.mgrid[2] - 1), s3 in 0:(S.mgrid[3] - 1))
