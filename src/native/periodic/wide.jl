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
#
# f is real, so ρ̂(-m) = conj ρ̂(m): only the half space m₁ ≥ 0 is computed
# (weight 2 off the plane m₁ = 0, real part at the end). The sums over the
# uniform cells are FFTs (fftaxis.jl); the dense coefficient transforms of
# non-tiled cells keep real and imaginary parts as separate arrays and run as
# real products (`gemm!`) on all threads over slices or row blocks.

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
    ix = 1 / x
    @inbounds for l in (lm + 30 + ceil(Int, 1.5x)):-1:1
        c = (2l + 1) * ix * b - a
        a = b; b = c
        l - 1 <= lm && (j[l] = c)
        if abs(b) > 1e250
            a *= 1e-250; b *= 1e-250
            for i in l:(lm + 1)
                j[i] *= 1e-250
            end
        end
    end
    # normalise by j_0, or by j_1 near the zeros of j_0 (x ≥ 1, where the
    # closed form of j_1 does not cancel)
    j0 = sin(x) / x
    j1 = x < 1 ? 0.0 : sin(x) / x^2 - cos(x) / x
    sc = abs(j0) >= abs(j1) ? j0 / j[1] : j1 / j[2]
    @inbounds for i in 1:(lm + 1)
        j[i] *= sc
    end
    return j
end

# Real and imaginary parts of t[l+1] = (2l+1) (sgn i)^l j_l(κ h), l < p, for κ
# of either sign (j_l(-x) = (-1)^l j_l(x)); jl is scratch of length ≥ p + 1.
@inline function _legexp!(tr::AbstractVector{Float64}, ti::AbstractVector{Float64}, jl::Vector{Float64},
                          p::Int, κ::Float64, h::Float64, sgn::Int)
    sphbesselj!(jl, p - 1, abs(κ) * h)
    @inbounds for l in 0:(p - 1)
        a = (2l + 1) * jl[l + 1] * (isodd(l) ? (κ < 0 ? -sgn : sgn) : 1)
        r = l & 3                                    # i^l = 1, i, -1, -i
        tr[l + 1] = r == 0 ? a : r == 2 ? -a : 0.0
        ti[l + 1] = r == 1 ? a : r == 3 ? -a : 0.0
    end
    return tr, ti
end

# Real and imaginary parts of T[i, l + p s + 1] = wt(m) (2l+1) (sgn i)^l j_l(κ(m) h) e^{i φ(m, s)},
# m = ms[i], s < ns, l < p.
function _legtable(ms, p::Int, ns::Int, κ, h::Float64, sgn::Int, φ, wt = m -> 1.0)
    Tr = zeros(length(ms), p * ns); Ti = zeros(length(ms), p * ns)
    jl = zeros(p + 1); tr = zeros(p); ti = zeros(p)
    for (i, m) in enumerate(ms)
        _legexp!(tr, ti, jl, p, κ(m), h, sgn)
        for s in 0:(ns - 1)
            e = wt(m) * cis(φ(m, s))
            for l in 1:p
                v = e * complex(tr[l], ti[l])
                Tr[i, l + p * s] = real(v); Ti[i, l + p * s] = imag(v)
            end
        end
    end
    return Tr, Ti
end

_tr(T::NTuple{2,Matrix{Float64}}) = (Matrix(transpose(T[1])), Matrix(transpose(T[2])))

struct WidePlan
    pf::Int                         # source moments per axis (degree < pf)
    pe::Int                         # u_wide coefficients per axis (degree < pe)
    Kh::NTuple{3,Int}               # mode half extents
    mt::NTuple{3,Int}               # source root cells per fractional axis
    scale::Float64                  # V / prod(mt) (u-measure -> volume)
    W::Array{Float64,3}             # Ŵ on the half grid m₁ ≥ 0 (Kh₁+1, K₂, K₃), zero outside |k| ≤ kmax and at m = 0
    B::Matrix{Float64}              # k = B m
    # moments -> modes. Many cells per axis: FFT over the source cells
    # (fftaxis.jl), per axis T[l, m] = (2l+1)(-i)^l j_l(π m/mt) e^{-iπm/mt} and the
    # frequencies m mod mt. Few cells (dense products are faster): (real, imaginary)
    # parts of F_d[m, l + pf s + 1] = (2l+1)(-i)^l j_l(π m/mt) e^{-2πi m (s+½)/mt}
    fT::Union{Nothing,NTuple{3,Matrix{ComplexF64}}}
    fr::Union{Nothing,NTuple{3,Vector{Int}}}
    F1::Union{Nothing,NTuple{2,Matrix{Float64}}}    # rows m₁ ≥ 0
    F2t::Union{Nothing,NTuple{2,Matrix{Float64}}}   # transpose(F₂)
    F3t::Union{Nothing,NTuple{2,Matrix{Float64}}}   # transpose(F₃)
    # modes -> coefficients by FFT over the level-0 cubes (many cells, cubes tiling the cell)
    iE::Union{Nothing,NTuple{3,Matrix{ComplexF64}}}
    ir::Union{Nothing,NTuple{3,Vector{Int}}}
    # otherwise dense, E_d[m, l + pe s + 1] = (2l+1) i^l j_l(k_d rc/2) e^{ik_d c_s}
    E1::NTuple{2,Matrix{Float64}}   # rows m₁ ≥ 0 times the half-space weight (k₁ = B₁₁ m₁ for any cell)
    E2::Union{Nothing,NTuple{2,Matrix{Float64}}}   # orthorhombic: E₂
    E3t::Union{Nothing,NTuple{2,Matrix{Float64}}}  # orthorhombic: transpose(E₃)
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
    B = 2pi * Matrix(transpose(g.Uinv))
    mt = source_cells(g)
    W = _wide_weights(g, B, Kh, kmax, tsw, wsw)
    E(d, ms, wt = m -> 1.0) = _legtable(ms, pe, g.mgrid[d], m -> B[d, d] * m, rc / 2, 1,
                                        (m, s) -> B[d, d] * m * (g.xgrid0[d] + (s + 0.5) * rc), wt)
    full(d) = (-Kh[d]):Kh[d]
    ms = (0:Kh[1], full(2), full(3))
    wt1 = m -> m == 0 ? 1.0 : 2.0
    fft = maximum(mt) >= _FFT_CELLS
    F(d) = _legtable(ms[d], pf, mt[d], m -> pi * m / mt[d], 1.0, -1, (m, s) -> -2pi * m * (s + 0.5) / mt[d])
    fT = fft ? ntuple(d -> _axis_T(ms[d], pf, mt[d]), 3) : nothing
    fr = fft ? ntuple(d -> _axis_r(ms[d], mt[d]), 3) : nothing
    tiledfft = g.tiled && maximum(g.mgrid) >= _FFT_CELLS
    iE = tiledfft ? ntuple(d -> _axis_E(ms[d], pe, g.mgrid[d], rc, g.xgrid0[d], d == 1 ? wt1 : (m -> 1.0)), 3) : nothing
    ir = tiledfft ? ntuple(d -> _axis_r(ms[d], g.mgrid[d]), 3) : nothing
    return WidePlan(pf, pe, Kh, mt, g.V / prod(mt), W, B, fT, fr, fft ? nothing : F(1), fft ? nothing : _tr(F(2)),
                    fft ? nothing : _tr(F(3)), iE, ir,
                    E(1, 0:Kh[1], wt1),
                    g.diagonal ? E(2, full(2)) : nothing, g.diagonal ? _tr(E(3, full(3))) : nothing,
                    rc, g.xgrid0, g.mgrid)
end

# Ŵ on the half mode grid m₁ ≥ 0. Orthorhombic: Σ_s c_s e1_s ⊗ e2_s ⊗ e3_s,
# one GEMM per z-slice; otherwise per mode, summing the Gaussians by
# increasing δ until exp(-δ|k|²/4) is negligible.
function _wide_weights(g::PeriodicGrid, B::Matrix{Float64}, Kh, kmax::Float64, tsw::Vector{Float64}, wsw::Vector{Float64})
    K = (Kh[1] + 1, 2Kh[2] + 1, 2Kh[3] + 1)
    mof = (-1, -Kh[2] - 1, -Kh[3] - 1)              # m = i + mof
    ns = length(tsw)
    c = [wsw[s] * (pi * tsw[s])^1.5 / g.V for s in 1:ns]
    W = zeros(K...)
    kvec(m1, m2, m3) = (B[1, 1] * m1, B[2, 1] * m1 + B[2, 2] * m2, B[3, 1] * m1 + B[3, 2] * m2 + B[3, 3] * m3)
    if g.diagonal
        kv = ntuple(d -> [B[d, d] * (i + mof[d]) for i in 1:K[d]], 3)
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
            for i2 in 1:K[2], i1 in 1:K[1]
                k = kvec(i1 + mof[1], i2 + mof[2], i3 + mof[3])
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
        k = kvec(i1 + mof[1], i2 + mof[2], i3 + mof[3])
        q = k[1]^2 + k[2]^2 + k[3]^2
        (q == 0 || q > kmax^2) && (W[i1, i2, i3] = 0.0)
    end
    return W
end

# Complex C (+)= A B from real and imaginary parts (pointers, column major):
# Cr = Ar Br - Ai Bi, Ci = Ar Bi + Ai Br; pCi = C_NULL keeps the real part only.
function _zgemm!(pCr::Ptr{Float64}, pCi::Ptr{Float64}, ldc::Int, pAr::Ptr{Float64}, pAi::Ptr{Float64}, lda::Int,
                 pBr::Ptr{Float64}, pBi::Ptr{Float64}, ldb::Int, M::Int, N::Int, K::Int, add::Bool)
    gemm!(pCr, ldc, pAr, lda, pBr, ldb, M, N, K, 1.0, add)
    gemm!(pCr, ldc, pAi, lda, pBi, ldb, M, N, K, -1.0, true)
    pCi == C_NULL && return nothing
    gemm!(pCi, ldc, pAr, lda, pBi, ldb, M, N, K, 1.0, add)
    gemm!(pCi, ldc, pAi, lda, pBr, ldb, M, N, K, 1.0, true)
    return nothing
end

# C = A B (M × N, K inner, all contiguous) on all threads: row blocks
# (bycols = false) or column blocks. Ci = nothing keeps the real part only.
function _zgemm_par!(Cr::Array{Float64}, Ci, Ar::Array{Float64}, Ai::Array{Float64},
                     Br::Array{Float64}, Bi::Array{Float64}, M::Int, N::Int, K::Int; bycols::Bool = false)
    nt = _nthreads()
    L = bycols ? N : M
    blk = bycols ? max(4, cld(L, 4nt)) : max(32, 16 * cld(cld(L, 4nt), 16))
    parallel_foreach(collect(0:blk:(L - 1)), fill(nothing, nt)) do i0, _
        len = min(blk, L - i0)
        GC.@preserve Cr Ci Ar Ai Br Bi begin
            pci = Ci === nothing ? Ptr{Float64}(C_NULL) : pointer(Ci)
            if bycols
                o, ob = M * i0, K * i0
                _zgemm!(pointer(Cr) + 8o, pci == C_NULL ? pci : pci + 8o, M, pointer(Ar), pointer(Ai), M,
                        pointer(Br) + 8ob, pointer(Bi) + 8ob, K, M, len, K, false)
            else
                _zgemm!(pointer(Cr) + 8i0, pci == C_NULL ? pci : pci + 8i0, M, pointer(Ar) + 8i0, pointer(Ai) + 8i0, M,
                        pointer(Br), pointer(Bi), K, len, N, K, false)
            end
        end
    end
    return nothing
end

# Cells per axis from which the sums over cells run as FFTs (below, dense
# products at the GEMM peak are faster).
const _FFT_CELLS = 24

# Modes G = Ŵ ρ̂ on the half grid (Kh₁+1, K₂, K₃) from the root-cell moments
# M (pf mt₁, pf mt₂, pf mt₃), index l + pf s per axis, u-measure: one FFT axis
# transform per fractional axis (fftaxis.jl), or dense products for few cells
# (then as (real, imaginary) parts). Buffers come from the arena `ar`.
wide_modes(P::WidePlan, M::Array{Float64,3}, ar::Arena) =
    P.fT === nothing ? _wide_modes_dense(P, M, ar) : _wide_modes_fft(P, M, ar)

function _wide_modes_fft(P::WidePlan, M::Array{Float64,3}, ar::Arena)
    Q1, Q2, Q3 = size(M)
    H1, K2, K3 = size(P.W)
    pf, mt = P.pf, P.mt
    X = _axis_fwd(M, 1, Q2 * Q3, pf, mt[1], P.fT[1], P.fr[1], ar)          # (H₁, Q₂, Q₃)
    Y = _axis_fwd(X, H1, Q3, pf, mt[2], P.fT[2], P.fr[2], ar)              # (H₁, K₂, Q₃)
    G = reshape(_axis_fwd(Y, H1 * K2, 1, pf, mt[3], P.fT[3], P.fr[3], ar), H1, K2, K3)
    W = P.W; sc = P.scale
    @inbounds @simd for i in eachindex(G)
        G[i] *= sc * W[i]
    end
    return G
end

# Per source slice q₃, Z[:, :, q₃] = F₁ M[:, :, q₃] F₂ᵀ, then G = Ŵ Z F₃ᵀ.
function _wide_modes_dense(P::WidePlan, M::Array{Float64,3}, ar::Arena)
    Q1, Q2, Q3 = size(M)
    H1, K2, K3 = size(P.W)
    (F1r, F1i), (F2r, F2i), (F3r, F3i) = P.F1, P.F2t, P.F3t
    Zr = alloc(ar, H1, K2, Q3); Zi = alloc(ar, H1, K2, Q3)
    wks = make_workspaces(() -> (alloc(ar, H1, Q2), alloc(ar, H1, Q2)))
    parallel_foreach(collect(1:Q3), wks) do q3, (Yr, Yi)
        GC.@preserve M Zr Zi Yr Yi F1r F1i F2r F2i begin
            pm = pointer(M) + 8 * Q1 * Q2 * (q3 - 1)
            o = 8 * H1 * K2 * (q3 - 1)
            gemm!(pointer(Yr), H1, pointer(F1r), H1, pm, Q1, H1, Q2, Q1, 1.0, false)
            gemm!(pointer(Yi), H1, pointer(F1i), H1, pm, Q1, H1, Q2, Q1, 1.0, false)
            _zgemm!(pointer(Zr) + o, pointer(Zi) + o, H1, pointer(Yr), pointer(Yi), H1,
                    pointer(F2r), pointer(F2i), Q2, H1, K2, Q2, false)
        end
    end
    Gr = alloc(ar, H1, K2, K3); Gi = alloc(ar, H1, K2, K3)
    _zgemm_par!(Gr, Gi, Zr, Zi, F3r, F3i, H1 * K2, K3, Q3)
    W = P.W; sc = P.scale
    @inbounds @simd for i in eachindex(Gr)
        w = sc * W[i]
        Gr[i] *= w; Gi[i] *= w
    end
    return Gr, Gi
end

"""
    wide_coefficients(P, G, ar) -> C

Legendre coefficients (degree < pe) of u_wide on the central level-0 cubes
from the half-grid modes `G` of `wide_modes`. By FFT over the cubes (many
cubes tiling the cell) in natural order, C[l₁ + pe s₁, l₂ + pe s₂, l₃ + pe s₃];
otherwise in reversed order, C[l₃ + pe s₃, l₂ + pe s₂, l₁ + pe s₁], by chirp-z
transforms (very many cubes, chirpz.jl) or dense products.
"""
function wide_coefficients(P::WidePlan, G::Array{ComplexF64,3}, ar::Arena)
    H1, K2, K3 = size(G)
    pe = P.pe
    if P.iE !== nothing
        mg = P.mgrid
        R2, R3 = pe * mg[2], pe * mg[3]
        Y3 = _axis_inv(G, H1 * K2, 1, pe, mg[3], P.iE[3], P.ir[3], ar)       # (H₁ K₂, R₃)
        Y2 = _axis_inv(Y3, H1, R3, pe, mg[2], P.iE[2], P.ir[2], ar)         # (H₁, R₂, R₃)
        return reshape(_axis_inv_real(Y2, 1, R2 * R3, pe, mg[1], P.iE[1], P.ir[1], ar), pe * mg[1], R2, R3)
    end
    maximum(P.mgrid) >= _CHIRP_MIN_CUBES && return _wide_coefs_chirp(P, G, ar)   # very many cubes, not tiling
    Gr = alloc(ar, H1, K2, K3); Gi = alloc(ar, H1, K2, K3)
    @inbounds for i in eachindex(G)
        Gr[i] = real(G[i]); Gi[i] = imag(G[i])
    end
    return wide_coefficients(P, (Gr, Gi), ar)
end
function wide_coefficients(P::WidePlan, G::NTuple{2,Array{Float64,3}}, ar::Arena)
    if P.iE === nothing && maximum(P.mgrid) >= _CHIRP_MIN_CUBES
        Gc = allocc(ar, size(G[1])...)
        @inbounds for i in eachindex(Gc)
            Gc[i] = complex(G[1][i], G[2][i])
        end
        return _wide_coefs_chirp(P, Gc, ar)
    end
    return P.E2 === nothing ? _wide_coefs_triangular(P, G..., ar) : _wide_coefs_separable(P, G..., ar)
end

# Ct[r₃, r₂, r₁] = Re Σ_{m₁ ≥ 0} A[r₃, r₂, m₁] E₁[m₁, r₁] (weighted rows of E₁).
function _wide_coefs_final(P, Ar::Array{Float64,3}, Ai::Array{Float64,3}, ar::Arena)
    R3, R2, H1 = size(Ar)
    E1r, E1i = P.E1
    Ct = alloc(ar, R3, R2, size(E1r, 2))
    _zgemm_par!(Ct, nothing, Ar, Ai, E1r, E1i, R3 * R2, size(E1r, 2), H1)
    return Ct
end

# Orthorhombic: G permuted to (m₃, m₂, m₁); E₃ᵀ G over m₃ (one product), then
# E₂ per m₁ slice, then E₁.
function _wide_coefs_separable(P::WidePlan, Gr::Array{Float64,3}, Gi::Array{Float64,3}, ar::Arena)
    H1, K2, K3 = size(Gr)
    (E2r, E2i), (E3r, E3i) = P.E2, P.E3t
    R2, R3 = size(E2r, 2), size(E3r, 1)
    Gpr = alloc(ar, K3, K2, H1); Gpi = alloc(ar, K3, K2, H1)
    permutedims!(Gpr, Gr, (3, 2, 1)); permutedims!(Gpi, Gi, (3, 2, 1))
    Y1r = alloc(ar, R3, K2, H1); Y1i = alloc(ar, R3, K2, H1)
    _zgemm_par!(Y1r, Y1i, E3r, E3i, Gpr, Gpi, R3, K2 * H1, K3; bycols = true)
    Y2r = alloc(ar, R3, R2, H1); Y2i = alloc(ar, R3, R2, H1)
    parallel_foreach(collect(1:H1), fill(nothing, _nthreads())) do i1, _
        GC.@preserve Y1r Y1i Y2r Y2i E2r E2i begin
            o1 = 8 * R3 * K2 * (i1 - 1); o2 = 8 * R3 * R2 * (i1 - 1)
            _zgemm!(pointer(Y2r) + o2, pointer(Y2i) + o2, R3, pointer(Y1r) + o1, pointer(Y1i) + o1, R3,
                    pointer(E2r), pointer(E2i), K2, R3, R2, K2, false)
        end
    end
    return _wide_coefs_final(P, Y2r, Y2i, ar)
end

# Triclinic: per m₁ slice, m₃ (tables per mode, k₃ depends on all of m) then
# m₂ (tables per (m₁, m₂)), then one product over m₁.
function _wide_coefs_triangular(P::WidePlan, Gr::Array{Float64,3}, Gi::Array{Float64,3}, ar::Arena)
    H1, K2, K3 = size(Gr)
    Kh = P.Kh; B = P.B; pe = P.pe; rc = P.rc
    R2, R3 = pe * P.mgrid[2], pe * P.mgrid[3]
    cs = ntuple(d -> [P.xgrid0[d] + (s + 0.5) * rc for s in 0:(P.mgrid[d] - 1)], 3)
    A2r = alloc(ar, R3, R2, H1); A2i = alloc(ar, R3, R2, H1)
    wks = make_workspaces(() -> (A1r = alloc(ar, R3, K2), A1i = alloc(ar, R3, K2), E2r = alloc(ar, K2, R2), E2i = alloc(ar, K2, R2),
                                 jl = zeros(pe + 1), tr = zeros(pe), ti = zeros(pe)))
    parallel_foreach(collect(1:H1), wks) do i1, w
        m1 = i1 - 1
        A1r, A1i, tr, ti = w.A1r, w.A1i, w.tr, w.ti
        fill!(A1r, 0.0); fill!(A1i, 0.0)
        any2 = false
        for i2 in 1:K2
            m2 = i2 - Kh[2] - 1
            for i3 in 1:K3
                gv = complex(Gr[i1, i2, i3], Gi[i1, i2, i3])
                gv == 0 && continue
                any2 = true
                κ = B[3, 1] * m1 + B[3, 2] * m2 + B[3, 3] * (i3 - Kh[3] - 1)
                _legexp!(tr, ti, w.jl, pe, κ, rc / 2, 1)
                for s in 0:(P.mgrid[3] - 1)
                    er, ei = reim(gv * cis(κ * cs[3][s + 1]))
                    o = pe * s
                    @inbounds @simd for l in 1:pe
                        A1r[o + l, i2] += er * tr[l] - ei * ti[l]
                        A1i[o + l, i2] += er * ti[l] + ei * tr[l]
                    end
                end
            end
        end
        o2 = 1 + R3 * R2 * (i1 - 1)
        if !any2
            fill!(view(A2r, o2:(o2 + R3 * R2 - 1)), 0.0); fill!(view(A2i, o2:(o2 + R3 * R2 - 1)), 0.0)
            return
        end
        for i2 in 1:K2
            κ = B[2, 1] * m1 + B[2, 2] * (i2 - Kh[2] - 1)
            _legexp!(tr, ti, w.jl, pe, κ, rc / 2, 1)
            for s in 0:(P.mgrid[2] - 1)
                e = cis(κ * cs[2][s + 1])
                @inbounds for l in 1:pe
                    v = e * complex(tr[l], ti[l])
                    w.E2r[i2, l + pe * s] = real(v)
                    w.E2i[i2, l + pe * s] = imag(v)
                end
            end
        end
        # A2[:, :, m₁] = A1 E2
        GC.@preserve A2r A2i w begin
            _zgemm!(pointer(A2r, o2), pointer(A2i, o2), R3, pointer(A1r), pointer(A1i), R3,
                    pointer(w.E2r), pointer(w.E2i), K2, R3, R2, K2, false)
        end
    end
    return _wide_coefs_final(P, A2r, A2i, ar)
end
