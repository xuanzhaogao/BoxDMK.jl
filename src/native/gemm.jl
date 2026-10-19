# Register-blocked SIMD GEMM for the small dense products of the solver.
#
#   C[M×N] = α A[M×K] B[K×N]            (add = false)
#   C[M×N] = C + α A[M×K] B[K×N]        (add = true)
#
# All matrices column major, addressed by pointer and leading dimension.
# Rows are processed in blocks of 16 (two 8-wide vectors, masked at the
# tail) and columns in blocks of NR, with all accumulators in registers.

const V8 = Vec{8,Float64}
const _GEMM_NR = 6

# SIMD width (elements per 64-byte vector) of the element type.
_vw(::Type{Float64}) = 8
_vw(::Type{Float32}) = 16
const _GemmT = Union{Float64,Float32}

@generated function _gemm_block!(pC::Ptr{T}, ldc::Int, pA::Ptr{T}, lda::Int,
                                 pB::Ptr{T}, ldb::Int, K::Int, α::T, add::Bool,
                                 m1::Vec{W,Bool}, m2::Vec{W,Bool},
                                 ::Val{MV}, ::Val{NR}, ::Val{MASKED}) where {T,W,MV,NR,MASKED}
    V = Vec{W,T}
    S = sizeof(T)
    loadA(r) = MASKED ? :(vload($V, pA + $S * ($(W * (r - 1)) + k * lda), $(r == 1 ? :m1 : :m2))) :
                        :(vload($V, pA + $S * ($(W * (r - 1)) + k * lda)))
    body = Expr(:block)
    for c in 1:NR, r in 1:MV
        push!(body.args, :($(Symbol(:acc_, r, :_, c)) = zero($V)))
    end
    kloop = Expr(:block)
    for r in 1:MV
        push!(kloop.args, :($(Symbol(:a_, r)) = $(loadA(r))))
    end
    for c in 1:NR
        push!(kloop.args, :($(Symbol(:b_, c)) = $V(unsafe_load(pB + $S * (k + $(c - 1) * ldb)))))
        for r in 1:MV
            acc = Symbol(:acc_, r, :_, c)
            push!(kloop.args, :($acc = muladd($(Symbol(:a_, r)), $(Symbol(:b_, c)), $acc)))
        end
    end
    push!(body.args, :(for k in 0:(K - 1)
        $kloop
    end))
    av = :($V(α))
    for c in 1:NR, r in 1:MV
        acc = Symbol(:acc_, r, :_, c)
        ptr = :(pC + $S * ($(W * (r - 1)) + $(c - 1) * ldc))
        mask = r == 1 ? :m1 : :m2
        if MASKED
            push!(body.args, quote
                if add
                    vstore(muladd($av, $acc, vload($V, $ptr, $mask)), $ptr, $mask)
                else
                    vstore($av * $acc, $ptr, $mask)
                end
            end)
        else
            push!(body.args, quote
                if add
                    vstore(muladd($av, $acc, vload($V, $ptr)), $ptr)
                else
                    vstore($av * $acc, $ptr)
                end
            end)
        end
    end
    push!(body.args, :(return nothing))
    return body
end

@inline _rowmask(nrows::Int) = _rowmask(Val(8), nrows)
@inline _rowmask(::Val{W}, nrows::Int) where {W} = Vec{W,Bool}(ntuple(i -> i <= nrows, Val(W)))

@inline function _gemm_cols!(pC::Ptr{T}, ldc, pA::Ptr{T}, lda, pB::Ptr{T}, ldb, N, K, α::T, add, m1, m2,
                             mv::Val, masked::Val) where {T}
    S = sizeof(T)
    j0 = 0
    if mv === Val(1)        # one vector of rows: twice the columns, to hide the FMA latency
        while j0 + 2 * _GEMM_NR <= N
            _gemm_block!(pC + S * j0 * ldc, ldc, pA, lda, pB + S * j0 * ldb, ldb, K, α, add, m1, m2, mv,
                         Val(2 * _GEMM_NR), masked)
            j0 += 2 * _GEMM_NR
        end
    end
    while j0 + _GEMM_NR <= N
        _gemm_block!(pC + S * j0 * ldc, ldc, pA, lda, pB + S * j0 * ldb, ldb, K, α, add, m1, m2, mv, Val(_GEMM_NR), masked)
        j0 += _GEMM_NR
    end
    rem = N - j0
    pc = pC + S * j0 * ldc
    pb = pB + S * j0 * ldb
    if rem == 1
        _gemm_block!(pc, ldc, pA, lda, pb, ldb, K, α, add, m1, m2, mv, Val(1), masked)
    elseif rem == 2
        _gemm_block!(pc, ldc, pA, lda, pb, ldb, K, α, add, m1, m2, mv, Val(2), masked)
    elseif rem == 3
        _gemm_block!(pc, ldc, pA, lda, pb, ldb, K, α, add, m1, m2, mv, Val(3), masked)
    elseif rem == 4
        _gemm_block!(pc, ldc, pA, lda, pb, ldb, K, α, add, m1, m2, mv, Val(4), masked)
    elseif rem == 5
        _gemm_block!(pc, ldc, pA, lda, pb, ldb, K, α, add, m1, m2, mv, Val(5), masked)
    end
    return nothing
end

"""
    gemm!(pC, ldc, pA, lda, pB, ldb, M, N, K, α, add)

`C = α A B` or `C += α A B` for column-major matrices given by pointers
(Float64 or Float32).
"""
function gemm!(pC::Ptr{T}, ldc::Int, pA::Ptr{T}, lda::Int, pB::Ptr{T}, ldb::Int,
               M::Int, N::Int, K::Int, α::Real, add::Bool) where {T<:_GemmT}
    (M == 0 || N == 0) && return nothing
    if K == 0
        add || for j in 0:(N - 1), i in 1:M
            unsafe_store!(pC, zero(T), i + j * ldc)
        end
        return nothing
    end
    W = _vw(T)
    S = sizeof(T)
    a = T(α)
    full = _rowmask(Val(W), W)
    i0 = 0
    while i0 < M
        rows = min(2W, M - i0)
        pc = pC + S * i0
        pa = pA + S * i0
        if rows == 2W
            _gemm_cols!(pc, ldc, pa, lda, pB, ldb, N, K, a, add, full, full, Val(2), Val(false))
        elseif rows > W
            _gemm_cols!(pc, ldc, pa, lda, pB, ldb, N, K, a, add, full, _rowmask(Val(W), rows - W), Val(2), Val(true))
        elseif rows == W
            _gemm_cols!(pc, ldc, pa, lda, pB, ldb, N, K, a, add, full, full, Val(1), Val(false))
        else
            _gemm_cols!(pc, ldc, pa, lda, pB, ldb, N, K, a, add, _rowmask(Val(W), rows), full, Val(1), Val(true))
        end
        i0 += 2W
    end
    return nothing
end

# Matrix convenience wrapper (used in tests).
function gemm!(C::StridedMatrix{Float64}, A::StridedMatrix{Float64}, B::StridedMatrix{Float64},
               α::Float64 = 1.0, add::Bool = false)
    M, K = size(A)
    N = size(B, 2)
    GC.@preserve C A B gemm!(pointer(C), stride(C, 2), pointer(A), stride(A, 2), pointer(B), stride(B, 2),
                             M, N, K, α, add)
    return C
end

"""
    gemm_batch!(pC, ldc, sC, pA, lda, sA, pB, ldb, M, N, K, nbatch, α, add)

`nbatch` independent products C_k (+)= α A_k B with C_k = pC + k*sC and
A_k = pA + k*sA (strides in elements); B is shared.
"""
function gemm_batch!(pC::Ptr{T}, ldc::Int, sC::Int, pA::Ptr{T}, lda::Int, sA::Int,
                     pB::Ptr{T}, ldb::Int, M::Int, N::Int, K::Int, nbatch::Int, α::Real, add::Bool) where {T<:_GemmT}
    S = sizeof(T)
    for k in 0:(nbatch - 1)
        gemm!(pC + S * k * sC, ldc, pA + S * k * sA, lda, pB, ldb, M, N, K, α, add)
    end
    return nothing
end
