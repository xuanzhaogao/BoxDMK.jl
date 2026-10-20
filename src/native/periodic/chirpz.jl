# Modes -> Legendre coefficients of the wide term on the level-0 cubes when
# the cubes do not tile the cell (triclinic, or sides that are not multiples
# of rc), by chirp-z transforms.
#
# In the rotated frame k = B m is lower triangular: along axis d,
# k_d = κ₀ + B_dd m_d with κ₀ fixed by the earlier indices, and on the cube
# centre c_s = x₀ + (s + ½) rc
#
#   e^{ik_d c_s} = e^{iκ₀ c_s} e^{iB_dd m_d (x₀ + rc/2)} w^{m_d s},   w = e^{iB_dd rc}.
#
# The sum over modes of the last factor, X_s = Σ_j b_j w^{js}, is a chirp-z
# transform with w not a root of unity (the cubes do not tile the period).
# Bluestein: js = (j² + s² - (s - j)²)/2, so X_s = w^{s²/2} Σ_j (b_j w^{j²/2}) w^{-(s-j)²/2},
# a convolution evaluated exactly with FFTs of length L ≥ K + N - 1.

# Within the chirp-z path, axes with fewer cubes take the direct sum.
const _CHIRP_CELLS = 32
# Cubes on some axis from which the chirp-z path replaces the dense products.
# It needs several times fewer flops from a few tens of cubes on, but the
# dense products run as GEMMs at the FP64 peak while the batched short FFTs
# reach a fraction of it: measured on genoa, chirp-z is still 1.6× slower at
# 125 × 110 × 4 cubes and wins only from several hundred cubes per axis.
const _CHIRP_MIN_CUBES = 384

struct _Chirp
    K::Int                          # input length (modes)
    N::Int                          # output length (cubes)
    L::Int                          # FFT length (0: direct sum)
    pre::Vector{ComplexF64}         # w^{j²/2}, j < K
    post::Vector{ComplexF64}        # w^{s²/2} / L, s < N
    H::Vector{ComplexF64}           # FFT of the cyclic kernel w^{-j²/2}, -(K-1) ≤ j ≤ N-1
    D::Matrix{ComplexF64}           # direct sum (few cubes): w^{js}, (N × K)
end

function _Chirp(θ::Float64, K::Int, N::Int)
    N < _CHIRP_CELLS && return _Chirp(K, N, 0, ComplexF64[], ComplexF64[], ComplexF64[],
                                      [cis(θ * j * s) for s in 0:(N - 1), j in 0:(K - 1)])
    L = nextprod([2, 3, 5], K + N - 1)
    ph(j) = cis(θ * (j * j / 2))
    h = zeros(ComplexF64, L)
    for j in 0:(N - 1)
        h[j + 1] = conj(ph(j))
    end
    for j in 1:(K - 1)
        h[L - j + 1] = conj(ph(j))
    end
    H = _with_fftw_threads(() -> FFTW.fft(h), 1)
    return _Chirp(K, N, L, [ph(j) for j in 0:(K - 1)], [ph(s) / L for s in 0:(N - 1)], H, zeros(ComplexF64, 0, 0))
end

# Per-thread buffer (L × C) and its forward / inverse plans (single-threaded;
# planned outside the threaded loops, the FFTW planner is not thread-safe).
function _chirp_ws(cz::_Chirp, C::Int, ar::Arena)
    cz.L == 0 && return (buf = allocc(ar, max(cz.K, cz.N), C), tmp = zeros(ComplexF64, cz.N), pf = nothing, pb = nothing)
    buf = allocc(ar, cz.L, C)
    pf, pb = _with_fftw_threads(1) do
        FFTW.plan_fft!(buf, 1; flags = FFTW.ESTIMATE), FFTW.plan_bfft!(buf, 1; flags = FFTW.ESTIMATE)
    end
    return (buf = buf, tmp = ComplexF64[], pf = pf, pb = pb)
end

# X[s, c] = Σ_j b[j, c] w^{js} for the C columns of the workspace buffer, whose
# first K rows hold b on entry (overwritten); the result is left in rows 1:N.
function _chirpz!(ws, cz::_Chirp, C::Int)
    buf = ws.buf
    K, N, L = cz.K, cz.N, cz.L
    if L == 0                                    # direct sum
        D, t = cz.D, ws.tmp
        @inbounds for c in 1:C
            for s in 1:N
                acc = zero(ComplexF64)
                @simd for j in 1:K
                    acc = muladd(D[s, j], buf[j, c], acc)
                end
                t[s] = acc
            end
            for s in 1:N
                buf[s, c] = t[s]
            end
        end
        return buf
    end
    @inbounds for c in 1:C
        for j in 1:K
            buf[j, c] *= cz.pre[j]
        end
        for j in (K + 1):L
            buf[j, c] = 0
        end
    end
    ws.pf * buf
    @inbounds for c in 1:C, j in 1:L
        buf[j, c] *= cz.H[j]
    end
    ws.pb * buf
    @inbounds for c in 1:C, s in 1:N
        buf[s, c] *= cz.post[s]
    end
    return buf
end

# Ct[r₃, r₂, r₁] (reversed order, as `_wide_coefs_triangular`) from the modes
# G (Kh₁+1, K₂, K₃): per m₁ slice a chirp-z over m₃ (Bessel tables per mode),
# then over m₂ (tables per (m₁, m₂)), then over m₁ (real part, column blocks).
function _wide_coefs_chirp(P::WidePlan, G::Array{ComplexF64,3}, ar::Arena)
    H1, K2, K3 = size(G)
    Kh, B, pe, rc, mg, x0 = P.Kh, P.B, P.pe, P.rc, P.mgrid, P.xgrid0
    N1, N2, N3 = mg
    R2, R3 = pe * N2, pe * N3
    cz3 = _Chirp(B[3, 3] * rc, K3, N3)
    cz2 = _Chirp(B[2, 2] * rc, K2, N2)
    cz1 = _Chirp(B[1, 1] * rc, H1, N1)
    cs(d, s) = x0[d] + (s + 0.5) * rc
    # per-cube phases w^{-Kh s} (modes indexed from -Kh)
    sh3 = [cis(-B[3, 3] * rc * Kh[3] * s) for s in 0:(N3 - 1)]
    sh2 = [cis(-B[2, 2] * rc * Kh[2] * s) for s in 0:(N2 - 1)]
    A2 = allocc(ar, R3, R2, H1)
    wks = [(z3 = _chirp_ws(cz3, pe * K2, ar), z2 = _chirp_ws(cz2, R3 * pe, ar), A1 = allocc(ar, R3, K2),
            jl = zeros(pe + 1), tr = zeros(pe), ti = zeros(pe)) for _ in 1:_nthreads()]
    parallel_foreach(collect(1:H1), wks) do i1, w
        m1 = i1 - 1
        tr, ti, A1 = w.tr, w.ti, w.A1
        # z: b[j, (l, i₂)] = G t_l(κ₃) e^{iB₃₃ m₃ (x₀ + rc/2)}
        b3 = w.z3.buf
        @inbounds for i2 in 1:K2
            m2 = i2 - Kh[2] - 1
            for j in 1:K3
                gv = G[i1, i2, j]
                c0 = (i2 - 1) * pe
                if gv == 0
                    for l in 1:pe
                        b3[j, c0 + l] = 0
                    end
                    continue
                end
                m3 = j - Kh[3] - 1
                κ = B[3, 1] * m1 + B[3, 2] * m2 + B[3, 3] * m3
                _legexp!(tr, ti, w.jl, pe, κ, rc / 2, 1)
                e = gv * cis(B[3, 3] * m3 * (x0[3] + rc / 2))
                for l in 1:pe
                    b3[j, c0 + l] = e * complex(tr[l], ti[l])
                end
            end
        end
        _chirpz!(w.z3, cz3, pe * K2)
        @inbounds for i2 in 1:K2
            κ0 = B[3, 1] * m1 + B[3, 2] * (i2 - Kh[2] - 1)
            for s in 0:(N3 - 1)
                ph = cis(κ0 * cs(3, s)) * sh3[s + 1]
                for l in 1:pe
                    A1[l + pe * s, i2] = ph * b3[s + 1, (i2 - 1) * pe + l]
                end
            end
        end
        # y: b[i₂, (r₃, l₂)] = A1[r₃, i₂] t_l(κ₂) e^{iB₂₂ m₂ (x₀ + rc/2)}
        b2 = w.z2.buf
        @inbounds for i2 in 1:K2
            m2 = i2 - Kh[2] - 1
            κ = B[2, 1] * m1 + B[2, 2] * m2
            _legexp!(tr, ti, w.jl, pe, κ, rc / 2, 1)
            e = cis(B[2, 2] * m2 * (x0[2] + rc / 2))
            for l in 1:pe
                t = e * complex(tr[l], ti[l])
                for r3 in 1:R3
                    b2[i2, r3 + R3 * (l - 1)] = A1[r3, i2] * t
                end
            end
        end
        _chirpz!(w.z2, cz2, R3 * pe)
        κ0 = B[2, 1] * m1
        @inbounds for s in 0:(N2 - 1)
            ph = cis(κ0 * cs(2, s)) * sh2[s + 1]
            for l in 1:pe, r3 in 1:R3
                A2[r3, l + pe * s, i1] = ph * b2[s + 1, r3 + R3 * (l - 1)]
            end
        end
    end
    # x (real part): b[i₁, (col, l₁)] = A2[col, i₁] wt t_l(k₁) e^{iB₁₁ m₁ (x₀ + rc/2)}, col = (r₃, r₂)
    E1 = zeros(ComplexF64, H1, pe)
    let jl = zeros(pe + 1), tr = zeros(pe), ti = zeros(pe)
        for i1 in 1:H1
            m1 = i1 - 1
            _legexp!(tr, ti, jl, pe, B[1, 1] * m1, rc / 2, 1)
            e = (m1 == 0 ? 1.0 : 2.0) * cis(B[1, 1] * m1 * (x0[1] + rc / 2))
            for l in 1:pe
                E1[i1, l] = e * complex(tr[l], ti[l])
            end
        end
    end
    ncol = R3 * R2
    R1 = pe * N1
    cb = max(1, min(cld(ncol, 4 * _nthreads()), cld(1 << 16, pe)))     # columns per block
    Ct = alloc(ar, R3, R2, R1)
    wk1 = [_chirp_ws(cz1, cb * pe, ar) for _ in 1:_nthreads()]
    parallel_foreach(collect(1:cb:ncol), wk1) do c0, z
        nc = min(cb, ncol - c0 + 1)
        b1 = z.buf
        @inbounds for l in 1:pe, c in 1:nc, j in 1:H1
            b1[j, c + cb * (l - 1)] = A2[c0 + c - 1 + ncol * (j - 1)] * E1[j, l]
        end
        _chirpz!(z, cz1, cb * pe)
        @inbounds for s in 0:(N1 - 1), l in 1:pe, c in 1:nc
            Ct[c0 + c - 1 + ncol * (l - 1 + pe * s)] = real(b1[s + 1, c + cb * (l - 1)])
        end
    end
    return Ct
end
