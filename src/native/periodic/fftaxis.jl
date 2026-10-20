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

# FFTW on all threads (forward = e^{-2πi rs/n}, inverse unnormalised).
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

# Work items (first index of a block of `ablk` rows, column b) covering (A, ·, B).
_axis_items(A::Int, B::Int) = (ablk = A >= 2048 ? 512 : A; ([(a0, b) for b in 1:B for a0 in 1:ablk:A], ablk))

# Forward transform of real data X (A, p mt, B) -> (A, K, B): real FFT over
# the cells (half spectrum; frequency r > mt/2 is the conjugate of mt - r), fold.
function _axis_fwd(X::AbstractArray{Float64}, A::Int, B::Int, p::Int, mt::Int, T::Matrix{ComplexF64},
                   r::Vector{Int}, ar::Arena)
    h = mt ÷ 2 + 1
    Xh = allocc(ar, A, p, h, B)
    _rfft_dim!(Xh, reshape(X, A, p, mt, B), 3)
    K = length(r)
    out = allocc(ar, A, K, B)
    items, ablk = _axis_items(A, B)
    parallel_foreach(items, fill(nothing, _nthreads())) do (a0, b), _
        a1 = min(A, a0 + ablk - 1)
        @inbounds for m in 1:K
            f = r[m]; cj = f >= h; fi = cj ? mt - f + 1 : f + 1
            for a in a0:a1
                out[a, m, b] = 0
            end
            for l in 1:p
                t = T[l, m]
                if cj
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

# Forward transform of complex data X (A, p mt, B) -> (A, K, B); X is overwritten.
function _axis_fwd(X::AbstractArray{ComplexF64}, A::Int, B::Int, p::Int, mt::Int, T::Matrix{ComplexF64},
                   r::Vector{Int}, ar::Arena)
    Xh = reshape(X, A, p, mt, B)
    _fft_dim!(Xh, 3, true)
    K = length(r)
    out = allocc(ar, A, K, B)
    items, ablk = _axis_items(A, B)
    parallel_foreach(items, fill(nothing, _nthreads())) do (a0, b), _
        a1 = min(A, a0 + ablk - 1)
        @inbounds for m in 1:K
            fi = r[m] + 1
            for a in a0:a1
                out[a, m, b] = 0
            end
            for l in 1:p
                t = T[l, m]
                @simd for a in a0:a1
                    out[a, m, b] = muladd(t, Xh[a, l, fi, b], out[a, m, b])
                end
            end
        end
    end
    return out
end

# Inverse transform Y (A, K, B) -> complex (A, p mt, B): accumulate on the
# frequencies, inverse FFT over the cells.
function _axis_inv(Y::AbstractArray{ComplexF64}, A::Int, B::Int, p::Int, mt::Int, E::Matrix{ComplexF64},
                   r::Vector{Int}, ar::Arena)
    K = length(r)
    W = allocc(ar, A, p, mt, B)
    items, ablk = _axis_items(A, B)
    Y3 = reshape(Y, A, K, B)
    parallel_foreach(items, fill(nothing, _nthreads())) do (a0, b), _
        a1 = min(A, a0 + ablk - 1)
        @inbounds begin
            for f in 1:mt, l in 1:p, a in a0:a1
                W[a, l, f, b] = 0
            end
            for m in 1:K
                fi = r[m] + 1
                for l in 1:p
                    e = E[m, l]
                    @simd for a in a0:a1
                        W[a, l, fi, b] = muladd(e, Y3[a, m, b], W[a, l, fi, b])
                    end
                end
            end
        end
    end
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
    items, ablk = _axis_items(A, B)
    Y3 = reshape(Y, A, K, B)
    parallel_foreach(items, fill(nothing, _nthreads())) do (a0, b), _
        a1 = min(A, a0 + ablk - 1)
        @inbounds begin
            for f in 1:h, l in 1:p, a in a0:a1
                V[a, l, f, b] = 0
            end
            for m in 1:K
                f = r[m]; fb = mod(mt - f, mt)
                for l in 1:p
                    e = E[m, l] / 2
                    if f < h
                        @simd for a in a0:a1
                            V[a, l, f + 1, b] = muladd(e, Y3[a, m, b], V[a, l, f + 1, b])
                        end
                    end
                    if fb < h
                        ec = conj(e)
                        @simd for a in a0:a1
                            V[a, l, fb + 1, b] = muladd(ec, conj(Y3[a, m, b]), V[a, l, fb + 1, b])
                        end
                    end
                end
            end
        end
    end
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
