# Oblique slab cells: modes -> Legendre coefficients of the wide potential on
# the level-0 cubes by a type-2 NUFFT evaluated only at Gauss nodes of the
# cubes, followed by a projection onto the Legendre basis of each cube.
#
# After the z step, for each z component r₃ (degree l₃ on z-cell s₃) the
# in-plane wide potential is the real 2D Fourier series
#
#   u_r₃(x) = Re Σ_{m₁ ≥ 0, m₂} wt(m₁) Out[m₁, m₂, r₃] e^{i m·ξ},   ξ = Bᵀx = 2π U₂⁻¹ x
#
# (x in the rotated frame, wt = 1 at m₁ = 0 and 2 otherwise, as the dense
# transform). It is evaluated at q × q Gauss–Legendre nodes of each needed
# cube (q > pe: oversampled) and projected onto degree < pe per axis by the
# Gauss quadrature. The dense transform instead applies the exact Legendre
# projection of each plane wave; the two agree to the aliasing error of the
# q-point projection and the NUFFT tolerance (checked in the tests).
#
# NUFFT (as FINUFFT): fine grid of σ = 2 times the band per axis, kernel
# ψ(z) = exp(β(√(1 - z²) - 1)) of width w points, deconvolution by its Fourier
# transform Ψ̂. The m₁ band 0 … Kh₁ is centred (m₁ = m' + sh), the shift is a
# phase e^{i sh ξ₁} per target. Interpolation contracts the w × w window in
# two stages: over the y window once per node row (ξ₂ depends on x₂ only),
# then over the x window per node.

const _NUFFT_CHUNK = 8               # cubes per work item

struct _SlabNUFFT
    w::Int                           # kernel width (points)
    β::Float64
    kc::Matrix{Float64}              # Chebyshev coefficients (degree × w) of the window weights
    n::NTuple{2,Int}                 # fine grid
    sh::Int                          # m₁ = m' + sh, m' centred
    dec::Matrix{Float64}             # (Kh₁+1, K₂): wt(m₁) / (Ψ̂₁(m') Ψ̂₂(m₂))
    i1::Vector{Int}                  # fine-grid index of each m₁ (1-based)
    i2::Vector{Int}                  # … of each m₂
    q::Int                           # nodes per axis per cube
    xq::Vector{Float64}              # Gauss–Legendre nodes on [-1, 1]
    PwT::Matrix{Float64}             # q × pe: (2l+1)/2 w_a P_l(x_a)
    cubes::Vector{NTuple{2,Int}}     # in-plane cubes needed (0-based)
    rows::Vector{Tuple{Int,Vector{Int}}}   # work items: (row j, consecutive cubes [i …] of it)
end

_es_kernel(z::Float64, β::Float64) = abs(z) < 1 ? exp(β * (sqrt(1 - z * z) - 1)) : 0.0

# Ψ̂(m/n) = ∫_{-w/2}^{w/2} ψ(2y/w) cos(2π m y / n) dy (Gauss–Legendre).
function _es_hat(ms, n::Int, w::Int, β::Float64)
    leg = ortho1d(LegendreBasis(), 4w + 40)
    y = leg.x .* (w / 2); wy = leg.w .* (w / 2)
    ψ = [_es_kernel(t, β) for t in leg.x]
    return [sum(wy[a] * ψ[a] * cos(2pi * m * y[a] / n) for a in eachindex(y)) for m in ms]
end

# Window of a target at fine-grid coordinate t: the points l₀ … l₀ + w - 1,
# l₀ = ⌈t - w/2⌉, with weights ψ((t - l₀ - e)/(w/2)), e = 0 … w-1, functions
# of v = t - l₀ - (w/2 - 1) ∈ (0, 1] only: tabulated as Chebyshev series in
# x = 2v - 1 (degree w + 6), evaluated by Clenshaw for all e at once.
function _es_table(w::Int, β::Float64)
    d = w + 6
    xs = [cos(pi * (k + 0.5) / (d + 1)) for k in 0:d]
    kc = zeros(d + 1, w)
    for e in 0:(w - 1)
        f = [_es_kernel(((x + 1) / 2 + w / 2 - 1 - e) / (w / 2), β) for x in xs]
        for j in 0:d
            kc[j + 1, e + 1] = (j == 0 ? 1 : 2) / (d + 1) * sum(f[k + 1] * cos(pi * j * (k + 0.5) / (d + 1)) for k in 0:d)
        end
    end
    return kc
end

@inline function _es_weights!(ψ::Vector{Float64}, t::Float64, P::_SlabNUFFT)
    w, kc = P.w, P.kc
    l0 = ceil(Int, t - w / 2)
    x = 2 * (t - l0 - (w / 2 - 1)) - 1
    d = size(kc, 1)
    @inbounds for e in 1:w
        b1 = 0.0; b2 = 0.0
        for j in d:-1:2
            b1, b2 = muladd(2x, b1, kc[j, e] - b2), b1
        end
        ψ[e] = muladd(x, b1, kc[1, e] - b2)
    end
    return l0
end

function _SlabNUFFT(Kh::NTuple{2,Int}, pe::Int, cubes::Vector{NTuple{2,Int}}, tol::Float64, q::Int)
    w = clamp(ceil(Int, log10(1 / tol)) + 1, 2, 16)
    β = 2.30 * w
    H1, K2 = Kh[1] + 1, 2Kh[2] + 1
    n = (nextprod([2, 3, 5], max(2 * H1, 2w)), nextprod([2, 3, 5], max(2 * K2, 2w)))
    sh = Kh[1] ÷ 2
    m1c = (0:Kh[1]) .- sh
    m2s = (-Kh[2]):Kh[2]
    h1 = _es_hat(m1c, n[1], w, β); h2 = _es_hat(m2s, n[2], w, β)
    dec = [(i1 == 1 ? 1.0 : 2.0) / (h1[i1] * h2[i2]) for i1 in 1:H1, i2 in 1:K2]
    i1 = [mod(m, n[1]) + 1 for m in m1c]
    i2 = [mod(m, n[2]) + 1 for m in m2s]
    leg = ortho1d(LegendreBasis(), q)
    PwT = zeros(q, pe); pv = zeros(pe)
    for a in 1:q
        legepols!(pv, leg.x[a], pe)
        for l in 1:pe
            PwT[a, l] = (2l - 1) / 2 * leg.w[a] * pv[l]
        end
    end
    byrow = Dict{Int,Vector{Int}}()
    for (i, j) in cubes
        push!(get!(Vector{Int}, byrow, j), i)
    end
    # work items: runs of at most _NUFFT_CHUNK consecutive cubes of a row
    rows = Tuple{Int,Vector{Int}}[]
    for (j, is) in sort(collect(byrow))
        sort!(is)
        k = 1
        while k <= length(is)
            e = k
            while e < length(is) && e - k + 1 < _NUFFT_CHUNK && is[e + 1] == is[e] + 1
                e += 1
            end
            push!(rows, (j, is[k:e]))
            k = e + 1
        end
    end
    return _SlabNUFFT(w, β, _es_table(w, β), n, sh, dec, i1, i2, q, collect(leg.x), PwT, cubes, rows)
end

# Ct[r₃, r₂, r₁] (reversed order, as the dense oblique transform) on the needed
# cubes. One work item per cube row j: for each node row x₂ of it, the y
# window of the fine grid is contracted once over the x range the row needs,
# then each node takes its x window; the q × q node values of a cube go to its
# Legendre coefficients by two products with the projection matrix.
function _slab_coefs_nufft(S, Out::Array{ComplexF64,3}, ar::Arena)
    P = S.nu
    H1, K2, R3 = size(Out)
    pe, rc, B = S.pe, S.rc, S.B
    mg1, mg2 = S.mgrid[1], S.mgrid[2]
    n1, n2 = P.n
    w, q = P.w, P.q
    G = allocc(ar, R3, n1, n2)
    parallel_foreach(collect(1:n2), fill(nothing, _nthreads())) do j2, _
        fill!(view(G, :, :, j2), 0)
    end
    parallel_foreach(collect(1:K2), fill(nothing, _nthreads())) do i2, _
        j2 = P.i2[i2]
        @inbounds for i1 in 1:H1
            d = P.dec[i1, i2]; j1 = P.i1[i1]
            @simd for r in 1:R3
                G[r, j1, j2] = d * Out[i1, i2, r]
            end
        end
    end
    _fft_dim!(reshape(G, R3, 1, n1, n2), 3, false)
    _fft_dim!(reshape(G, R3 * n1, 1, n2, 1), 3, false)
    Ct = alloc(ar, R3, pe * mg2, pe * mg1)      # only the needed cube blocks are written (and read)
    s1 = n1 / 2pi; s2 = n2 / 2pi
    maxrow = maximum(length(is) for (_, is) in P.rows; init = 0)
    nhx = ceil(Int, s1 * (abs(B[1, 1]) * mg1 + abs(B[2, 1])) * rc) + 2w + 4
    wks = make_workspaces(() -> (Hx = allocc(ar, R3, nhx), V = alloc(ar, R3, q), T = alloc(ar, R3, pe, q, maxrow),
                                 ψ1 = zeros(w), ψ2 = zeros(w), j2 = zeros(Int, w), acc = zeros(ComplexF64, R3)))
    parallel_foreach(P.rows, wks) do (cj, ilist), wk
        Hx, V, T, ψ1, ψ2, jv, acc = wk.Hx, wk.V, wk.T, wk.ψ1, wk.ψ2, wk.j2, wk.acc
        c2 = S.xgrid0[2] + (cj + 0.5) * rc
        x1lo = S.xgrid0[1] + ilist[1] * rc; x1hi = S.xgrid0[1] + (ilist[end] + 1) * rc
        for b in 1:q
            x2 = c2 + rc / 2 * P.xq[b]
            l2 = _es_weights!(ψ2, s2 * B[2, 2] * x2, P)
            for e in 1:w
                jv[e] = mod(l2 + e - 1, n2) + 1
            end
            # x window of the row: ξ₁ = B₁₁ x₁ + B₂₁ x₂
            ta = s1 * (B[1, 1] * x1lo + B[2, 1] * x2); tb = s1 * (B[1, 1] * x1hi + B[2, 1] * x2)
            llo = ceil(Int, min(ta, tb) - w / 2) - 1
            nl = floor(Int, max(ta, tb) + w / 2) + 1 - llo + 1
            @inbounds for jj in 1:nl
                j1 = mod(llo + jj - 1, n1) + 1
                for r in 1:R3
                    Hx[r, jj] = 0
                end
                for e in 1:w
                    p2 = ψ2[e]; j2 = jv[e]
                    @simd for r in 1:R3
                        Hx[r, jj] = muladd(p2, G[r, j1, j2], Hx[r, jj])
                    end
                end
            end
            for (k, ci) in enumerate(ilist)
                c1 = S.xgrid0[1] + (ci + 0.5) * rc
                for a in 1:q
                    ξ1 = B[1, 1] * (c1 + rc / 2 * P.xq[a]) + B[2, 1] * x2
                    l1 = _es_weights!(ψ1, s1 * ξ1, P)
                    o = l1 - llo
                    @inbounds begin
                        for r in 1:R3
                            acc[r] = 0
                        end
                        for e in 1:w
                            p1 = ψ1[e]
                            @simd for r in 1:R3
                                acc[r] = muladd(p1, Hx[r, o + e], acc[r])
                            end
                        end
                        pr, pim = reim(cis(P.sh * ξ1))
                        @simd for r in 1:R3
                            V[r, a] = pr * real(acc[r]) - pim * imag(acc[r])
                        end
                    end
                end
                # x projection: T[:, :, b, k] = V (R₃ × q) · PwT (q × pe)
                GC.@preserve V T P gemm!(pointer(T, 1 + R3 * pe * ((b - 1) + q * (k - 1))), R3, pointer(V), R3,
                                         pointer(P.PwT), q, R3, pe, q, 1.0, false)
            end
        end
        # y projection: Ct[:, pe cj + l₂, pe ci + l₁] = Σ_b T[:, l₁, b, k] PwT[b, l₂]
        for (k, ci) in enumerate(ilist), l1 in 1:pe
            GC.@preserve T Ct P gemm!(pointer(Ct, 1 + R3 * (pe * cj + pe * mg2 * (pe * ci + l1 - 1))), R3,
                                      pointer(T, 1 + R3 * ((l1 - 1) + pe * q * (k - 1))), R3 * pe,
                                      pointer(P.PwT), q, R3, pe, q, 1.0, false)
        end
    end
    return Ct
end

# Model costs (multiply-adds) of the two back-transforms, for the automatic
# choice; measured on genoa (48 threads) the NUFFT reaches about half the
# rate of the dense products, and pays off past ≈ 50 cubes per axis.
function _back_costs(Kh::NTuple{2,Int}, pe::Int, mg::NTuple{3,Int}, ncubes::Int, tol::Float64, q::Int)
    H1, K2 = Kh[1] + 1, 2Kh[2] + 1
    R1, R2, R3 = pe .* mg
    dense = float(H1) * R3 * R2 * (K2 + R1)
    w = clamp(ceil(Int, log10(1 / tol)) + 1, 2, 16)
    n1 = nextprod([2, 3, 5], max(2 * H1, 2w)); n2 = nextprod([2, 3, 5], max(2 * K2, 2w))
    nufft = float(R3) * ncubes * q * (2q * w + pe * (q + pe)) + 5.0 * R3 * n1 * n2 * log2(n1 * n2)
    return dense, nufft
end
