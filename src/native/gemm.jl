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

@generated function _gemm_block!(pC::Ptr{Float64}, ldc::Int, pA::Ptr{Float64}, lda::Int,
                                 pB::Ptr{Float64}, ldb::Int, K::Int, α::Float64, add::Bool,
                                 m1::Vec{8,Bool}, m2::Vec{8,Bool},
                                 ::Val{MV}, ::Val{NR}, ::Val{MASKED}) where {MV,NR,MASKED}
    loadA(r) = MASKED ? :(vload(V8, pA + 8 * ($(8 * (r - 1)) + k * lda), $(r == 1 ? :m1 : :m2))) :
                        :(vload(V8, pA + 8 * ($(8 * (r - 1)) + k * lda)))
    body = Expr(:block)
    for c in 1:NR, r in 1:MV
        push!(body.args, :($(Symbol(:acc_, r, :_, c)) = zero(V8)))
    end
    kloop = Expr(:block)
    for r in 1:MV
        push!(kloop.args, :($(Symbol(:a_, r)) = $(loadA(r))))
    end
    for c in 1:NR
        push!(kloop.args, :($(Symbol(:b_, c)) = V8(unsafe_load(pB + 8 * (k + $(c - 1) * ldb)))))
        for r in 1:MV
            acc = Symbol(:acc_, r, :_, c)
            push!(kloop.args, :($acc = muladd($(Symbol(:a_, r)), $(Symbol(:b_, c)), $acc)))
        end
    end
    push!(body.args, :(for k in 0:(K - 1)
        $kloop
    end))
    av = :(V8(α))
    for c in 1:NR, r in 1:MV
        acc = Symbol(:acc_, r, :_, c)
        ptr = :(pC + 8 * ($(8 * (r - 1)) + $(c - 1) * ldc))
        mask = r == 1 ? :m1 : :m2
        if MASKED
            push!(body.args, quote
                if add
                    vstore(muladd($av, $acc, vload(V8, $ptr, $mask)), $ptr, $mask)
                else
                    vstore($av * $acc, $ptr, $mask)
                end
            end)
        else
            push!(body.args, quote
                if add
                    vstore(muladd($av, $acc, vload(V8, $ptr)), $ptr)
                else
                    vstore($av * $acc, $ptr)
                end
            end)
        end
    end
    push!(body.args, :(return nothing))
    return body
end

@inline function _rowmask(nrows::Int)
    return Vec{8,Bool}(ntuple(i -> i <= nrows, 8))
end

@inline function _gemm_cols!(pC, ldc, pA, lda, pB, ldb, N, K, α, add, m1, m2, mv::Val, masked::Val)
    j0 = 0
    while j0 + _GEMM_NR <= N
        _gemm_block!(pC + 8 * j0 * ldc, ldc, pA, lda, pB + 8 * j0 * ldb, ldb, K, α, add, m1, m2, mv, Val(_GEMM_NR), masked)
        j0 += _GEMM_NR
    end
    rem = N - j0
    pc = pC + 8 * j0 * ldc
    pb = pB + 8 * j0 * ldb
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

`C = α A B` or `C += α A B` for column-major matrices given by pointers.
"""
function gemm!(pC::Ptr{Float64}, ldc::Int, pA::Ptr{Float64}, lda::Int, pB::Ptr{Float64}, ldb::Int,
               M::Int, N::Int, K::Int, α::Float64, add::Bool)
    (M == 0 || N == 0) && return nothing
    if K == 0
        add || for j in 0:(N - 1), i in 1:M
            unsafe_store!(pC, 0.0, i + j * ldc)
        end
        return nothing
    end
    full = _rowmask(8)
    i0 = 0
    while i0 < M
        rows = min(16, M - i0)
        pc = pC + 8 * i0
        pa = pA + 8 * i0
        if rows == 16
            _gemm_cols!(pc, ldc, pa, lda, pB, ldb, N, K, α, add, full, full, Val(2), Val(false))
        elseif rows > 8
            _gemm_cols!(pc, ldc, pa, lda, pB, ldb, N, K, α, add, full, _rowmask(rows - 8), Val(2), Val(true))
        elseif rows == 8
            _gemm_cols!(pc, ldc, pa, lda, pB, ldb, N, K, α, add, full, full, Val(1), Val(false))
        else
            _gemm_cols!(pc, ldc, pa, lda, pB, ldb, N, K, α, add, _rowmask(rows), full, Val(1), Val(true))
        end
        i0 += 16
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
function gemm_batch!(pC::Ptr{Float64}, ldc::Int, sC::Int, pA::Ptr{Float64}, lda::Int, sA::Int,
                     pB::Ptr{Float64}, ldb::Int, M::Int, N::Int, K::Int, nbatch::Int, α::Float64, add::Bool)
    for k in 0:(nbatch - 1)
        gemm!(pC + 8 * k * sC, ldc, pA + 8 * k * sA, lda, pB, ldb, M, N, K, α, add)
    end
    return nothing
end
