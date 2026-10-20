# Exact per-axis transforms of the wide term through a uniform grid of cells.
# Along one axis the data index is l + p s (degree l < p, cell s < mt) and the
# mode m has the frequency r = m mod mt on the cell grid, since the phase of
# cell s is e^{∓2πi m (s + ½)/mt} = e^{∓iπm/mt} e^{∓2πi r s/mt}:
#
#   forward:  out[m]       = Σ_{l,s} T[l, m] e^{-2πi r_m s/mt} X[l + p s]
#   inverse:  out[l + p s] = Σ_m E[m, l] e^{+2πi r_m s/mt} Y[m]
#
# T and E hold the per-mode factors (Legendre-Bessel expansion of the cell
# and the phase of its centre), r the frequencies (0-based). The sum over
# cells is an FFT of length mt per degree, the modes fold onto its frequencies:
# exact, O(N log N) instead of O(modes × cells). Arrays are (A, axis, B), the
# transformed axis in the middle; every routine runs on all threads.

# FFTs along the cell axis (dimension 3 of (A, p, n, B) arrays; forward =
# e^{-2πi rs/n}, inverse unnormalised), run as single-threaded FFTW plans on
# pieces of the batch, in parallel on Julia threads. (FFTW's own threads,
# through FFTW.jl's spawnloop, crashed with a segfault in the test suite.)
function _with_fftw_threads(f, n::Int = 1)
    old = FFTW.get_num_threads()
    FFTW.set_num_threads(1)
    try
        return f()
    finally
        FFTW.set_num_threads(old)
    end
end

const _FFTW_FLAGS = FFTW.ESTIMATE | FFTW.UNALIGNED      # one plan for every piece

# Pieces (row range, batch range) of an (R, n, B) array for the threads.
function _fft_pieces(R::Int, B::Int)
    nt = _nthreads()
    if B >= 2nt
        cb = cld(B, 4nt)
        return [(1:R, b0:min(B, b0 + cb - 1)) for b0 in 1:cb:B]
    end
    rb = cld(R, max(1, cld(4nt, B)))
    return [(r0:min(R, r0 + rb - 1), b:b) for b in 1:B for r0 in 1:rb:R]
end

# Apply plan(view of in, view of out) piece by piece: `mk(vin, vout)` plans for
# one piece shape (serially), `run(P, vin, vout)` executes it.
function _fft_run!(mk, run, inp, out, R::Int, B::Int)
    pcs = _fft_pieces(R, B)
    plans = Dict{Tuple{Int,Int},Any}()
    for (rr, bb) in pcs
        key = (length(rr), length(bb))
        haskey(plans, key) || (plans[key] = _with_fftw_threads(() -> mk(view(inp, rr, :, bb), view(out, rr, :, bb))))
    end
    parallel_foreach(pcs, fill(nothing, _nthreads())) do (rr, bb), _
        run(plans[(length(rr), length(bb))], view(inp, rr, :, bb), view(out, rr, :, bb))
    end
    return out
end

function _fft_dim!(X4::AbstractArray{ComplexF64}, d::Int, forward::Bool)
    d == 3 || throw(ArgumentError("cell axis must be dimension 3"))
    A, p, n, B = size(X4)
    X = reshape(X4, A * p, n, B)
    _fft_run!((vi, _) -> forward ? FFTW.plan_fft!(vi, 2; flags = _FFTW_FLAGS) : FFTW.plan_bfft!(vi, 2; flags = _FFTW_FLAGS),
              (P, vi, _) -> P * vi, X, X, A * p, B)
    return X4
end

function _rfft_dim!(out4, X4::AbstractArray{Float64}, d::Int)
    d == 3 || throw(ArgumentError("cell axis must be dimension 3"))
    A, p, n, B = size(X4)
    _fft_run!((vi, vo) -> FFTW.plan_rfft(vi, 2; flags = _FFTW_FLAGS), (P, vi, vo) -> mul!(vo, P, vi),
              reshape(X4, A * p, n, B), reshape(out4, A * p, size(out4, 3), B), A * p, B)
    return out4
end

function _brfft_dim!(out4, V4::AbstractArray{ComplexF64}, n::Int, d::Int)
    d == 3 || throw(ArgumentError("cell axis must be dimension 3"))
    A, p, h, B = size(V4)
    _fft_run!((vi, vo) -> FFTW.plan_brfft(vi, n, 2; flags = _FFTW_FLAGS), (P, vi, vo) -> mul!(vo, P, vi),
              reshape(V4, A * p, h, B), reshape(out4, A * p, n, B), A * p, B)
    return out4
end

# Work items (first index of a block of `ablk` rows, column b) covering (A, ·, B).
_axis_items(A::Int, B::Int) = (ablk = A >= 2048 ? 512 : A; ([(a0, b) for b in 1:B for a0 in 1:ablk:A], ablk))

# Modes grouped by frequency (1-based index f + 1), so that each frequency's
# block of data is visited once and stays in cache.
function _freq_groups(r::Vector{Int}, n::Int)
    gr = [Int[] for _ in 1:n]
    for (m, f) in enumerate(r)
        push!(gr[f + 1], m)
    end
    return gr
end

# out[a, m, b] = Σ_l T[l, m] Xh[a, l, f, b] (or its conjugate) for the modes m of each frequency.
function _axis_fold!(out, Xh, A::Int, B::Int, p::Int, T::Matrix{ComplexF64}, groups, cj::Bool)
    items, ablk = _axis_items(A, B)
    parallel_foreach(items, fill(nothing, _nthreads())) do (a0, b), _
        a1 = min(A, a0 + ablk - 1)
        @inbounds for (fi, ms) in enumerate(groups), (m, c) in ms
            conj_ = cj && c
            if A == 1
                acc = zero(ComplexF64)
                if conj_
                    @simd for l in 1:p
                        acc = muladd(T[l, m], conj(Xh[1, l, fi, b]), acc)
                    end
                else
                    @simd for l in 1:p
                        acc = muladd(T[l, m], Xh[1, l, fi, b], acc)
                    end
                end
                out[1, m, b] = acc
                continue
            end
            for a in a0:a1
                out[a, m, b] = 0
            end
            for l in 1:p
                t = T[l, m]
                if conj_
                    @simd for a in a0:a1
                        out[a, m, b] = muladd(t, conj(Xh[a, l, fi, b]), out[a, m, b])
                    end
                else
                    @simd for a in a0:a1
                        out[a, m, b] = muladd(t, Xh[a, l, fi, b], out[a, m, b])
                    end
                end
            end
        end
    end
    return out
end

# Forward transform of real data X (A, p mt, B) -> (A, K, B): real FFT over
# the cells (half spectrum; frequency r > mt/2 is the conjugate of mt - r), fold.
function _axis_fwd(X::AbstractArray{Float64}, A::Int, B::Int, p::Int, mt::Int, T::Matrix{ComplexF64},
                   r::Vector{Int}, ar::Arena)
    h = mt ÷ 2 + 1
    Xh = allocc(ar, A, p, h, B)
    _rfft_dim!(Xh, reshape(X, A, p, mt, B), 3)
    groups = [Tuple{Int,Bool}[] for _ in 1:h]
    for (m, f) in enumerate(r)
        f < h ? push!(groups[f + 1], (m, false)) : push!(groups[mt - f + 1], (m, true))
    end
    return _axis_fold!(allocc(ar, A, length(r), B), Xh, A, B, p, T, groups, true)
end

# Forward transform of complex data X (A, p mt, B) -> (A, K, B); X is overwritten.
function _axis_fwd(X::AbstractArray{ComplexF64}, A::Int, B::Int, p::Int, mt::Int, T::Matrix{ComplexF64},
                   r::Vector{Int}, ar::Arena)
    Xh = reshape(X, A, p, mt, B)
    _fft_dim!(Xh, 3, true)
    groups = [[(m, false) for m in g] for g in _freq_groups(r, mt)]
    return _axis_fold!(allocc(ar, A, length(r), B), Xh, A, B, p, T, groups, false)
end

# W[a, l, f, b] (+)= Σ_{m of frequency f} E[m, l] Y[a, m, b] (half: E/2 and the
# conjugates on the mirrored frequencies), every frequency's block once.
function _axis_spread!(W, Y3, A::Int, B::Int, p::Int, Et::Matrix{ComplexF64}, groups)
    items, ablk = _axis_items(A, B)
    parallel_foreach(items, fill(nothing, _nthreads())) do (a0, b), _
        a1 = min(A, a0 + ablk - 1)
        @inbounds for (fi, ms) in enumerate(groups)
            for l in 1:p, a in a0:a1
                W[a, l, fi, b] = 0
            end
            for (m, c, w) in ms
                if A == 1
                    y = c ? conj(Y3[1, m, b]) : Y3[1, m, b]
                    @simd for l in 1:p
                        e = c ? conj(Et[l, m]) : Et[l, m]
                        W[1, l, fi, b] = muladd(w * e, y, W[1, l, fi, b])
                    end
                    continue
                end
                for l in 1:p
                    e = w * (c ? conj(Et[l, m]) : Et[l, m])
                    if c
                        @simd for a in a0:a1
                            W[a, l, fi, b] = muladd(e, conj(Y3[a, m, b]), W[a, l, fi, b])
                        end
                    else
                        @simd for a in a0:a1
                            W[a, l, fi, b] = muladd(e, Y3[a, m, b], W[a, l, fi, b])
                        end
                    end
                end
            end
        end
    end
    return W
end

# Inverse transform Y (A, K, B) -> complex (A, p mt, B): accumulate on the
# frequencies, inverse FFT over the cells.
function _axis_inv(Y::AbstractArray{ComplexF64}, A::Int, B::Int, p::Int, mt::Int, E::Matrix{ComplexF64},
                   r::Vector{Int}, ar::Arena)
    K = length(r)
    W = allocc(ar, A, p, mt, B)
    groups = [[(m, false, 1.0) for m in g] for g in _freq_groups(r, mt)]
    _axis_spread!(W, reshape(Y, A, K, B), A, B, p, Matrix(transpose(E)), groups)
    _fft_dim!(W, 3, false)
    return reshape(W, A, p * mt, B)
end

# Inverse transform with real output, Re Σ_m E e^{+2πi rs/mt} Y: the inverse
# real FFT of the Hermitian half (W_r + conj W_{mt-r})/2, accumulated directly.
function _axis_inv_real(Y::AbstractArray{ComplexF64}, A::Int, B::Int, p::Int, mt::Int, E::Matrix{ComplexF64},
                        r::Vector{Int}, ar::Arena)
    K = length(r)
    h = mt ÷ 2 + 1
    V = allocc(ar, A, p, h, B)
    groups = [Tuple{Int,Bool,Float64}[] for _ in 1:h]
    for (m, f) in enumerate(r)
        fb = mod(mt - f, mt)
        f < h && push!(groups[f + 1], (m, false, 0.5))
        fb < h && push!(groups[fb + 1], (m, true, 0.5))
    end
    _axis_spread!(V, reshape(Y, A, K, B), A, B, p, Matrix(transpose(E)), groups)
    out = alloc(ar, A, p * mt, B)
    _brfft_dim!(reshape(out, A, p, mt, B), V, mt, 3)
    return out
end

# Forward and inverse tables of one axis: T[l, m] = (2l+1)(-i)^l j_l(πm/mt) e^{-iπm/mt}
# (unit cells of the u-measure) and E[m, l] = wt(m) (2l+1) i^l j_l(k rc/2) e^{ik(x₀ + rc/2)}, k = 2πm/(mt rc).
_axis_T(ms, p::Int, mt::Int) =
    Matrix(transpose(complex.(_legtable(ms, p, 1, m -> pi * m / mt, 1.0, -1, (m, s) -> -pi * m / mt)...)))
_axis_E(ms, p::Int, mt::Int, rc::Float64, x0::Float64, wt = m -> 1.0) =
    complex.(_legtable(ms, p, 1, m -> 2pi * m / (mt * rc), rc / 2, 1, (m, s) -> 2pi * m / (mt * rc) * (x0 + rc / 2), wt)...)
_axis_r(ms, mt::Int) = [mod(m, mt) for m in ms]
