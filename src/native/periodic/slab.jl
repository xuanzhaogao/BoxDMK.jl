# Wide part of the quasi-2D (slab) potential: periodic in x, y, free in z.
# Poisson summation over the in-plane lattice turns each wide Gaussian into
#
#   Σ_n e^{-|x - y - An|²/δ} = (πδ/A) Σ_k̇ e^{-δ|k̇|²/4} e^{ik̇·(ṙ - ṙ')} e^{-(z - z')²/δ},
#
# A the in-plane cell area, k̇ = B m with B = 2π U₂⁻ᵀ (in-plane cell A₂ = Q₂U₂,
# rotated frame). In the plane each axis is the exact Legendre-Bessel expansion
# of the periodic solver (half space m₁ ≥ 0). The source moments are those of
# cells that tile one period in fractional coordinates (the level-0 cubes of a
# rectangular cell tiled by them, or the cells of the fractional source tree),
# so the sum over cells is an FFT (fftaxis.jl). The coefficients on the
# Cartesian level-0 cubes are an FFT too when the cubes tile the cell; for
# oblique cells (k₂ = B₂₁m₁ + B₂₂m₂) the triangular transform of the periodic
# solver, per m₁ slice. In z the kernel is applied directly: on the level-0
# z-cells (side rc) every wide Gaussian is smooth, so its z-convolution is a
# small exact matrix in the Legendre basis of the cells (no Fourier transform,
# no padding). The mode k̇ = 0 gets the kernel of all wide Gaussians,
#
#   D(z) = Σ_wide (πwδ/A) (e^{-z²/δ} - 1)  →  -2π|z|/A  (far tail in closed form),
#
# where the constants dropped vanish for a neutral density (a charged one gets
# the gauge u + 2πQ|z|/A → 0 far from the slab, see `_periodic_solve!`).

struct SlabWide
    pf::Int                          # source moments per axis (degree < pf)
    pe::Int                          # u_wide coefficients per axis (degree < pe)
    Kh::NTuple{2,Int}                # in-plane mode half extents
    mt::NTuple{3,Int}                # source cells per axis (fractional; z: slab / mt₃)
    mgrid::NTuple{3,Int}             # target level-0 cubes per axis
    tiled::Bool                      # the level-0 cubes tile the cell (FFT both ways)
    scale::Float64                   # u-measure moments -> volume
    T1t::Matrix{ComplexF64}          # pf × (Kh₁+1): (2l+1)(-i)^l j_l(πm/mt) e^{-iπm/mt}, m₁ ≥ 0
    T2t::Matrix{ComplexF64}          # pf × K₂, m₂ = -Kh₂ … Kh₂
    r1::Vector{Int}                  # m₁ mod mt₁
    r2::Vector{Int}                  # m₂ mod mt₂
    # tiled: inverse FFT tables (as fftaxis.jl); oblique: dense E₁ (half-space weights) and B
    Et1::Matrix{ComplexF64}          # (Kh₁+1) × pe
    Et2::Matrix{ComplexF64}          # K₂ × pe
    E1::NTuple{2,Matrix{Float64}}    # (Kh₁+1) × pe mgrid₁ (oblique)
    B::Matrix{Float64}               # k̇ = B m (2 × 2, lower triangular)
    rc::Float64
    xgrid0::NTuple{3,Float64}
    GzT::Vector{Matrix{Float64}}     # per wide Gaussian: transpose of its z matrix (pf mt₃ × pe mgrid₃)
    ord::Vector{Int}                 # modes (index m₁ + H₁ m₂) by increasing |k̇|
    nact::Vector{Int}                # per Gaussian: its modes are ord[1:nact] (weight below eps/1e3 beyond)
    cm::Matrix{Float64}              # (sorted modes) × #Gaussians: scale (πwδ/A) e^{-δ|k̇|²/4} (0 at k̇ = 0, beyond kmax)
    D0T::Matrix{Float64}             # transpose of the k̇ = 0 z matrix (times scale)
    charged::Bool                    # nonzero total charge allowed (gauge u + 2πQ|z|/A → 0)
    qconst::Float64                  # potential per unit charge to subtract for that gauge
end

# z matrix of the kernel K(z - z') from source z-cells (side hs from zs) to
# target z-cells (side ht from zt): entry
# [(l, i), (l', j)] = ((2l+1)/2)((2l'+1)/2) ∫∫ K(z_i(ξ) - z'_j(ξ')) P_l(ξ) P_l'(ξ') dξ dξ',
# so that the Legendre coefficients (degree < pe) on target cell i of
# ∫ ρ(z') K(z - z') dz' follow from the moments ∫_cell j ρ P_l' dz' (degree < pf).
function _zkernel_matrix(K, src::NTuple{3,Any}, tgt::NTuple{3,Any}, pe::Int, pf::Int)
    (hs, ns, zs), (ht, nt, zt) = src, tgt
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
    G = zeros(pe * nt, pf * ns)
    Kq = zeros(q, q)
    for j in 0:(ns - 1), i in 0:(nt - 1)
        for b in 1:q, a in 1:q
            Kq[a, b] = K((zt + (i + (ξ[a] + 1) / 2) * ht) - (zs + (j + (ξ[b] + 1) / 2) * hs))
        end
        G[pe * i .+ (1:pe), pf * j .+ (1:pf)] .= Pe * Kq * transpose(Pf)
    end
    return G
end

function SlabWide(g::PeriodicGrid, eps::Float64, pf::Int, pe::Int, tsw::Vector{Float64}, wsw::Vector{Float64};
                  charged::Bool = false, narrow = (Float64[], Float64[]))
    rc = g.rc
    U2 = g.U[1:2, 1:2]
    H = g.U[3, 3]
    area = abs(det(U2))
    B = 2pi * Matrix(transpose(inv(U2)))
    kmax = 2 * log(1 / eps) / rc
    Kh = ntuple(d -> ceil(Int, kmax * norm(view(U2, :, d)) / 2pi), 2)
    mt = source_cells(g)
    mg = g.mgrid
    m1s = 0:Kh[1]; m2s = (-Kh[2]):Kh[2]
    H1, K2 = Kh[1] + 1, 2Kh[2] + 1
    sc = g.tiled ? rc^3 : area * H / prod(mt)     # u-measure moments -> volume
    src = g.tiled ? (rc, mg[3], g.xgrid0[3]) : (H / mt[3], mt[3], 0.0)
    tgt = (rc, mg[3], g.xgrid0[3])
    # Gaussians that reach a mode k̇ ≠ 0, with their per-mode weights
    kk = [(B[1, 1] * (i1 - 1))^2 + (B[2, 1] * (i1 - 1) + B[2, 2] * (i2 - Kh[2] - 1))^2 for i1 in 1:H1, i2 in 1:K2]
    kmin2 = minimum(k for k in kk if k > 0)
    keep = [s for s in eachindex(tsw) if wsw[s] * pi * tsw[s] * exp(-tsw[s] * kmin2 / 4) / area > 1e-18 * maximum(wsw) * rc]
    ord = sortperm(vec(kk))
    cm = zeros(H1 * K2, length(keep))
    nact = zeros(Int, length(keep))
    for (c, s) in enumerate(keep)
        kc2 = min(kmax^2, 4 * log(1e3 / eps) / tsw[s])  # e^{-δk²/4} < eps/1e3 beyond
        for (r, m) in enumerate(ord)
            k2 = kk[m]
            k2 > kc2 && break
            nact[c] = r
            k2 == 0 && continue
            cm[r, c] = sc * wsw[s] * pi * tsw[s] / area * exp(-tsw[s] * k2 / 4)
        end
    end
    GzT = [Matrix(transpose(_zkernel_matrix(z -> exp(-z^2 / tsw[s]), src, tgt, pe, pf))) for s in keep]
    # k̇ = 0: all wide Gaussians, (e^{-z²/δ} - 1); beyond the widest, -z² Σ w (geometric tail)
    byw = sortperm(tsw)
    b = sqrt(tsw[byw[end]] / tsw[byw[end - 1]])
    wtail = wsw[byw[end]] / (b - 1)
    D(z) = (sum(wsw[s] * pi * tsw[s] * expm1(-z^2 / tsw[s]) for s in eachindex(tsw)) - pi * wtail * z^2) / area
    D0T = Matrix(transpose(sc .* _zkernel_matrix(D, src, tgt, pe, pf)))
    # Gauge constant: at k̇ = 0 the narrow images give Σ_narrow (πwδ/A) e^{-z²/δ},
    # so the whole kernel is K(z) = Σ_narrow (πwδ/A) e^{-z²/δ} + D(z), which
    # is -2π|z|/A + const for |z| above the scale 2⁻¹⁶ rc where the u-series
    # stops resolving 1/r: the constant, taken at z* = 2⁻¹¹ rc (still ≪ rc,
    # where the error ε 2π|z|/A of the series is negligible).
    zs = rc / 2048
    tsn, wsn = narrow
    qconst = D(zs) + (sum(wsn[s] * pi * tsn[s] * exp(-zs^2 / tsn[s]) for s in eachindex(tsn); init = 0.0) + 2pi * zs) / area
    wt1 = m -> m == 0 ? 1.0 : 2.0
    Et1 = g.tiled ? _axis_E(m1s, pe, mg[1], rc, g.xgrid0[1], wt1) : zeros(ComplexF64, 0, 0)
    Et2 = g.tiled ? _axis_E(m2s, pe, mg[2], rc, g.xgrid0[2]) : zeros(ComplexF64, 0, 0)
    E1 = g.tiled ? (zeros(0, 0), zeros(0, 0)) :
         _legtable(m1s, pe, mg[1], m -> B[1, 1] * m, rc / 2, 1, (m, s) -> B[1, 1] * m * (g.xgrid0[1] + (s + 0.5) * rc), wt1)
    return SlabWide(pf, pe, Kh, mt, mg, g.tiled, sc, _axis_T(m1s, pf, mt[1]), _axis_T(m2s, pf, mt[2]),
                    _axis_r(m1s, mt[1]), _axis_r(m2s, mt[2]), Et1, Et2, E1, B, rc, g.xgrid0,
                    GzT, ord, nact, cm, D0T, charged, qconst)
end

"""
    slab_coefficients(S::SlabWide, M, ar) -> C

Legendre coefficients of the wide slab potential on the level-0 cubes from
the source moments `M (pf mt₁, pf mt₂, pf mt₃)` (u-measure): in natural order
`C[l₁ + pe s₁, l₂ + pe s₂, l₃ + pe s₃]` when the cubes tile the cell, else in
reversed order `C[l₃ + pe s₃, l₂ + pe s₂, l₁ + pe s₁]`.
"""
function slab_coefficients(S::SlabWide, M::Array{Float64,3}, ar::Arena)
    pf, pe = S.pf, S.pe
    mt1, mt2, mt3 = S.mt
    mg1, mg2, mg3 = S.mgrid
    H1, K2 = S.Kh[1] + 1, 2S.Kh[2] + 1
    nm = H1 * K2
    Q2, Q3 = pf * mt2, pf * mt3
    R2, R3 = pe * mg2, pe * mg3
    nt = fill(nothing, _nthreads())
    T1t, T2t, Et1, Et2, r1, r2 = S.T1t, S.T2t, S.Et1, S.Et2, S.r1, S.r2
    # in plane (x, then y): Z[m₁, m₂, q₃]
    X = _axis_fwd(M, 1, Q2 * Q3, pf, mt1, T1t, r1, ar)                  # (H₁, Q₂, Q₃)
    Z = _axis_fwd(X, H1, Q3, pf, mt2, T2t, r2, ar)                        # (H₁, K₂, Q₃)
    # z, per mode: Out = Σ_g (c_g ⊙ Z) G_gᵀ plus the k̇ = 0 kernel, on (re, im)
    # rows of the modes sorted by |k̇|, where Gaussian g acts on the first nact[g]
    ord = S.ord
    Zs = alloc(ar, 2nm, Q3)
    parallel_foreach(collect(1:Q3), nt) do q, _
        @inbounds for (r, m) in enumerate(ord)
            z = Z[m + nm * (q - 1)]
            Zs[2r - 1, q] = real(z); Zs[2r, q] = imag(z)
        end
    end
    Os = alloc(ar, 2nm, R3)
    nm2 = 2nm
    blk = max(32, 32 * cld(cld(nm2, 4 * _nthreads()), 32))
    wsc = make_workspaces(() -> alloc(ar, blk, Q3))
    parallel_foreach(collect(0:blk:(nm2 - 1)), wsc) do r0, Tm
        len = min(blk, nm2 - r0)
        GC.@preserve Zs Os Tm S begin
            fill!(view(Os, (r0 + 1):(r0 + len), :), 0.0)
            for c in eachindex(S.GzT)
                la = min(len, 2 * S.nact[c] - r0)
                la <= 0 && continue
                cmc = view(S.cm, :, c)
                @inbounds for q in 1:Q3, r in 1:la
                    Tm[r + blk * (q - 1)] = cmc[(r0 + r + 1) >> 1] * Zs[r0 + r, q]
                end
                gemm!(pointer(Os) + 8r0, nm2, pointer(Tm), blk, pointer(S.GzT[c]), Q3, la, R3, Q3, 1.0, true)
            end
        end
    end
    i0 = findfirst(==(1 + H1 * S.Kh[2]), ord)            # k̇ = 0 (m₁ = 0, m₂ = 0)
    GC.@preserve Zs Os S begin
        o = 8 * 2 * (i0 - 1)
        gemm!(pointer(Os) + o, nm2, pointer(Zs) + o, nm2, pointer(S.D0T), Q3, 2, R3, Q3, 1.0, true)
    end
    Out = allocc(ar, H1, K2, R3)
    parallel_foreach(collect(1:R3), nt) do q, _
        @inbounds for (r, m) in enumerate(ord)
            Out[m + nm * (q - 1)] = complex(Os[2r - 1, q], Os[2r, q])
        end
    end
    S.tiled || return _slab_coefs_oblique(S, Out, ar)
    # back in plane (y, then x with the real part): C[l₁ + pe s₁, r₂, r₃]
    C2 = _axis_inv(Out, H1, R3, pe, mg2, Et2, r2, ar)                     # (H₁, R₂, R₃)
    C = reshape(_axis_inv_real(C2, 1, R2 * R3, pe, mg1, Et1, r1, ar), pe * mg1, R2, R3)
    return C
end

# Oblique cells: from Out (H₁, K₂, R₃), per m₁ slice A2[:, :, m₁] = Out[m₁, :, :]ᵀ E₂(m₁)
# with k₂ = B₂₁m₁ + B₂₂m₂ (tables per (m₁, m₂)), then one product over m₁ (as the
# triangular transform of the periodic solver). Reversed order Ct[r₃, r₂, r₁].
function _slab_coefs_oblique(S::SlabWide, Out::Array{ComplexF64,3}, ar::Arena)
    H1, K2, R3 = size(Out)
    pe, rc, B, Kh = S.pe, S.rc, S.B, S.Kh
    mg2 = S.mgrid[2]
    R2 = pe * mg2
    cs2 = [S.xgrid0[2] + (s + 0.5) * rc for s in 0:(mg2 - 1)]
    A2r = alloc(ar, R3, R2, H1); A2i = alloc(ar, R3, R2, H1)
    wks = make_workspaces(() -> (A1r = alloc(ar, R3, K2), A1i = alloc(ar, R3, K2), E2r = alloc(ar, K2, R2),
                                 E2i = alloc(ar, K2, R2), jl = zeros(pe + 1), tr = zeros(pe), ti = zeros(pe)))
    parallel_foreach(collect(1:H1), wks) do i1, w
        m1 = i1 - 1
        @inbounds for i2 in 1:K2, r3 in 1:R3
            v = Out[i1, i2, r3]
            w.A1r[r3, i2] = real(v); w.A1i[r3, i2] = imag(v)
        end
        for i2 in 1:K2
            κ = B[2, 1] * m1 + B[2, 2] * (i2 - Kh[2] - 1)
            _legexp!(w.tr, w.ti, w.jl, pe, κ, rc / 2, 1)
            for s in 0:(mg2 - 1)
                e = cis(κ * cs2[s + 1])
                @inbounds for l in 1:pe
                    v = e * complex(w.tr[l], w.ti[l])
                    w.E2r[i2, l + pe * s] = real(v)
                    w.E2i[i2, l + pe * s] = imag(v)
                end
            end
        end
        o2 = 1 + R3 * R2 * (i1 - 1)
        GC.@preserve A2r A2i w begin
            _zgemm!(pointer(A2r, o2), pointer(A2i, o2), R3, pointer(w.A1r), pointer(w.A1i), R3,
                    pointer(w.E2r), pointer(w.E2i), K2, R3, R2, K2, false)
        end
    end
    return _wide_coefs_final(S, A2r, A2i, ar)
end

# Total charge of the source moments (degree-0 entries times the cell volume).
_slab_charge(S::SlabWide, M::Array{Float64,3}) =
    S.scale * sum(M[1 + S.pf * s1, 1 + S.pf * s2, 1 + S.pf * s3]
                  for s1 in 0:(S.mt[1] - 1), s2 in 0:(S.mt[2] - 1), s3 in 0:(S.mt[3] - 1))
