# Wide part of the quasi-2D (slab) potential: periodic in x, y, free in z.
# Poisson summation over the in-plane lattice turns each wide Gaussian into
#
#   Σ_n e^{-|x - y - An|²/δ} = (πδ/A) Σ_k̇ e^{-δ|k̇|²/4} e^{ik̇·(ṙ - ṙ')} e^{-(z - z')²/δ},
#
# A = L₁L₂ the in-plane area, k̇ = (2πm₁/L₁, 2πm₂/L₂). In the plane each
# axis is the exact Legendre-Bessel expansion of the periodic solver (half
# space m₁ ≥ 0), factored through the uniform cell grid: with c_s = (s+½) rc
# and k = 2πm/(mt rc), e^{-ikc_s} = e^{-iπm/mt} e^{-2πi (m mod mt) s/mt}, so the
# sum over cells is an FFT of length mt per Legendre degree, and the modes
# fold onto its frequencies (exact, O(N log N)). In z the kernel is applied directly: on the level-0
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
    T1t::Matrix{ComplexF64}          # pf × (Kh₁+1): (2l+1)(-i)^l j_l(πm/mt) e^{-iπm/mt}, m₁ ≥ 0
    T2t::Matrix{ComplexF64}          # pf × K₂, m₂ = -Kh₂ … Kh₂
    Et1::Matrix{ComplexF64}          # (Kh₁+1) × pe: (2l+1) i^l j_l(k rc/2) e^{iπm/mt} × half-space weight
    Et2::Matrix{ComplexF64}          # K₂ × pe
    r1::Vector{Int}                  # 1 + (m₁ mod mt₁)
    r2::Vector{Int}                  # 1 + (m₂ mod mt₂)
    GzT::Vector{Matrix{Float64}}     # per wide Gaussian: transpose of its z matrix (pf mz × pe mz)
    ord::Vector{Int}                 # modes (index m₁ + H₁ m₂) by increasing |k̇|
    nact::Vector{Int}                # per Gaussian: its modes are ord[1:nact] (weight below eps/1e3 beyond)
    cm::Matrix{Float64}              # (sorted modes) × #Gaussians: rc³ (πwδ/A) e^{-δ|k̇|²/4} (0 at k̇ = 0, beyond kmax)
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
    cplx(T) = complex.(T[1], T[2])
    T(d, ms) = cplx(_legtable(ms, pf, 1, m -> pi * m / mt[d], 1.0, -1, (m, s) -> -pi * m / mt[d]))
    E(d, ms, wt = m -> 1.0) = cplx(_legtable(ms, pe, 1, m -> B[d] * m, rc / 2, 1, (m, s) -> pi * m / mt[d], wt))
    m1s = 0:Kh[1]; m2s = (-Kh[2]):Kh[2]
    H1, K2 = Kh[1] + 1, 2Kh[2] + 1
    sc = rc^3                                   # u-measure moments -> volume
    # Gaussians that reach a mode k̇ ≠ 0, with their per-mode weights
    kmin2 = minimum(B)^2
    keep = [s for s in eachindex(tsw) if wsw[s] * pi * tsw[s] * exp(-tsw[s] * kmin2 / 4) / area > 1e-18 * maximum(wsw) * rc]
    kk = [(B[1] * (i1 - 1))^2 + (B[2] * (i2 - Kh[2] - 1))^2 for i1 in 1:H1, i2 in 1:K2]   # index i1 + H1 (i2 - 1)
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
    GzT = [Matrix(transpose(_zkernel_matrix(z -> exp(-z^2 / tsw[s]), g, pe, pf))) for s in keep]
    # k̇ = 0: all wide Gaussians, (e^{-z²/δ} - 1); beyond the widest, -z² Σ w (geometric tail)
    byw = sortperm(tsw)
    b = sqrt(tsw[byw[end]] / tsw[byw[end - 1]])
    wtail = wsw[byw[end]] / (b - 1)
    D(z) = (sum(wsw[s] * pi * tsw[s] * expm1(-z^2 / tsw[s]) for s in eachindex(tsw)) - pi * wtail * z^2) / area
    D0T = Matrix(transpose(sc .* _zkernel_matrix(D, g, pe, pf)))
    return SlabWide(pf, pe, Kh, Matrix(transpose(T(1, m1s))), Matrix(transpose(T(2, m2s))),
                    E(1, m1s, m -> m == 0 ? 1.0 : 2.0), E(2, m2s), [mod(m, mt[1]) + 1 for m in m1s],
                    [mod(m, mt[2]) + 1 for m in m2s], GzT, ord, nact, cm, D0T, mt)
end

# FFTs along one dimension on all threads (forward = e^{-2πi rs/n}, inverse
# unnormalised): complex in place, real -> half spectrum, half spectrum -> real.
function _with_fftw_threads(f)
    old = FFTW.get_num_threads()
    FFTW.set_num_threads(_nthreads())
    try
        return f()
    finally
        FFTW.set_num_threads(old)
    end
end
_fft_dim!(A::AbstractArray{ComplexF64}, d::Int, forward::Bool) = _with_fftw_threads() do
    (forward ? FFTW.plan_fft!(A, d; flags = FFTW.ESTIMATE) : FFTW.plan_bfft!(A, d; flags = FFTW.ESTIMATE)) * A
end
_rfft_dim!(out, A::AbstractArray{Float64}, d::Int) =
    _with_fftw_threads(() -> mul!(out, FFTW.plan_rfft(A, d; flags = FFTW.ESTIMATE), A))
_brfft_dim!(out, A::AbstractArray{ComplexF64}, n::Int, d::Int) =
    _with_fftw_threads(() -> mul!(out, FFTW.plan_brfft(A, n, d; flags = FFTW.ESTIMATE), A))

"""
    slab_coefficients(S::SlabWide, M, ar) -> C

Legendre coefficients of the wide slab potential on the level-0 cubes,
`C[l₁ + pe s₁, l₂ + pe s₂, l₃ + pe s₃]`, from the level-0 moments
`M (pf mgrid₁, pf mgrid₂, pf mgrid₃)` (u-measure).
"""
function slab_coefficients(S::SlabWide, M::Array{Float64,3}, ar::Arena)
    pf, pe = S.pf, S.pe
    mt1, mt2, mz = S.mgrid
    H1, K2 = S.Kh[1] + 1, 2S.Kh[2] + 1
    nm = H1 * K2
    Q2, Q3 = pf * mt2, pf * mz
    R2, R3 = pe * mt2, pe * mz
    nt = fill(nothing, _nthreads())
    T1t, T2t, Et1, Et2, r1, r2 = S.T1t, S.T2t, S.Et1, S.Et2, S.r1, S.r2
    # x: real FFT over the cells (half spectrum; frequency r > mt/2 is the
    # conjugate of mt - r), then the modes m₁ from their frequency: X[m₁, (l₂ s₂), q₃]
    N1 = Q2 * Q3
    h1 = mt1 ÷ 2 + 1
    A1 = allocc(ar, pf, h1, N1)
    _rfft_dim!(A1, reshape(M, pf, mt1, N1), 2)
    X = allocc(ar, H1, Q2, Q3)
    parallel_foreach(collect(1:N1), nt) do n, _
        @inbounds for m in 1:H1
            r = r1[m] - 1
            acc = zero(ComplexF64)
            if r < h1
                @simd for l in 1:pf
                    acc = muladd(T1t[l, m], A1[l, r + 1, n], acc)
                end
            else
                @simd for l in 1:pf
                    acc = muladd(T1t[l, m], conj(A1[l, mt1 - r + 1, n]), acc)
                end
            end
            X[m + H1 * (n - 1)] = acc
        end
    end
    # y: the same along (l₂, s₂): Z[m₁, m₂, q₃]
    X4 = reshape(X, H1, pf, mt2, Q3)
    _fft_dim!(X4, 3, true)
    Z = allocc(ar, H1, K2, Q3)
    parallel_foreach([(m2, q3) for q3 in 1:Q3 for m2 in 1:K2], nt) do (m2, q3), _
        r = r2[m2]
        @inbounds begin
            for i in 1:H1
                Z[i, m2, q3] = 0
            end
            for l in 1:pf
                t = T2t[l, m2]
                @simd for i in 1:H1
                    Z[i, m2, q3] = muladd(t, X4[i, l, r, q3], Z[i, m2, q3])
                end
            end
        end
    end
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
    # back along y: W[m₁, l₂, frequency, r₃] = Σ_{m₂} Ẽ₂ Out, inverse FFT over the cells -> (m₁, l₂ + pe s₂, r₃)
    W2 = allocc(ar, H1, pe, mt2, R3)
    parallel_foreach([(l, r3) for r3 in 1:R3 for l in 1:pe], nt) do (l, r3), _
        @inbounds begin
            for f in 1:mt2, i in 1:H1
                W2[i, l, f, r3] = 0
            end
            for m2 in 1:K2
                e = Et2[m2, l]; f = r2[m2]
                @simd for i in 1:H1
                    W2[i, l, f, r3] = muladd(e, Out[i, m2, r3], W2[i, l, f, r3])
                end
            end
        end
    end
    _fft_dim!(W2, 3, false)
    # back along x, real part: Re Σ_r W_r e^{2πi rs/mt} is the inverse real FFT
    # of the Hermitian half (W_r + conj W_{mt-r})/2, accumulated directly:
    # V[l₁, frequency ≤ mt/2, (r₂, r₃)] from Σ_{m₁} Ẽ₁ C₂, then C[l₁ + pe s₁, r₂, r₃]
    N2 = R2 * R3
    C2 = reshape(W2, H1, N2)
    W1 = allocc(ar, pe, h1, N2)
    parallel_foreach(collect(1:N2), nt) do n, _
        @inbounds begin
            for f in 1:h1, l in 1:pe
                W1[l, f, n] = 0
            end
            for m in 1:H1
                c = C2[m, n] / 2; r = r1[m] - 1; rb = mod(mt1 - r, mt1)
                if r < h1
                    @simd for l in 1:pe
                        W1[l, r + 1, n] = muladd(Et1[m, l], c, W1[l, r + 1, n])
                    end
                end
                if rb < h1
                    cc = conj(c)
                    @simd for l in 1:pe
                        W1[l, rb + 1, n] = muladd(conj(Et1[m, l]), cc, W1[l, rb + 1, n])
                    end
                end
            end
        end
    end
    C = alloc(ar, pe * mt1, R2, R3)
    _brfft_dim!(reshape(C, pe, mt1, N2), W1, mt1, 2)
    return C
end

# Total charge of the level-0 moments (degree-0 entries, u-measure × rc³).
_slab_charge(S::SlabWide, M::Array{Float64,3}, rc::Float64) =
    rc^3 * sum(M[1 + S.pf * s1, 1 + S.pf * s2, 1 + S.pf * s3]
               for s1 in 0:(S.mgrid[1] - 1), s2 in 0:(S.mgrid[2] - 1), s3 in 0:(S.mgrid[3] - 1))
