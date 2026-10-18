# Dense 3D tensor-product kernels. Every inner loop runs along the first
# (contiguous) array index so that it vectorizes.

# out[a,b,c] (+)= α Σ A[a,i] Bt[j,b] Ct[k,c] inp[i,j,k]
# inp: n1*n2*n3 entries, A: (m1,n1), Bt: (n2,m2), Ct: (n3,m3) (the y and z
# matrices are passed transposed). w1 needs m1*n2*n3 entries, w2 m1*m2*n3.
function tp3!(out::AbstractArray{Float64}, inp::AbstractArray{Float64},
              A::AbstractMatrix{Float64}, Bt::AbstractMatrix{Float64}, Ct::AbstractMatrix{Float64},
              w1::Vector{Float64}, w2::Vector{Float64}, α::Float64, add::Bool)
    m1, n1 = size(A)
    n2, m2 = size(Bt)
    n3, m3 = size(Ct)
    GC.@preserve out inp A Bt Ct w1 w2 begin
        pw1 = pointer(w1); pw2 = pointer(w2)
        pA = pointer(A); pB = pointer(Bt); pC = pointer(Ct)
        gemm!(pw1, m1, pA, stride(A, 2), pointer(inp), n1, m1, n2 * n3, n1, 1.0, false)
        for k in 0:(n3 - 1)
            gemm!(pw2 + 8 * k * m1 * m2, m1, pw1 + 8 * k * m1 * n2, m1, pB, stride(Bt, 2), m1, m2, n2, 1.0, false)
        end
        gemm!(pointer(out), m1 * m2, pw2, m1 * m2, pC, stride(Ct, 2), m1 * m2, m3, n3, α, add)
    end
    return out
end

# out (+)= α (D applied along dimension `dim` of the n×n×n tensor inp).
function apply_dim!(out::AbstractArray{Float64}, inp::AbstractArray{Float64}, D::Matrix{Float64},
                    dim::Int, n::Int, α::Float64, add::Bool)
    if !add
        @inbounds for i in 1:(n^3)
            out[i] = 0.0
        end
    end
    if dim == 1
        @inbounds for jk in 0:(n * n - 1)
            b = jk * n
            for i in 1:n
                s = α * inp[b + i]
                @simd for a in 1:n
                    out[b + a] = muladd(D[a, i], s, out[b + a])
                end
            end
        end
    elseif dim == 2
        @inbounds for k in 0:(n - 1), bb in 1:n, j in 1:n
            s = α * D[bb, j]
            ob = (k * n + bb - 1) * n
            ib = (k * n + j - 1) * n
            @simd for a in 1:n
                out[ob + a] = muladd(s, inp[ib + a], out[ob + a])
            end
        end
    else
        nn = n * n
        @inbounds for c in 1:n, k in 1:n
            s = α * D[c, k]
            ob = (c - 1) * nn
            ib = (k - 1) * nn
            @simd for ab in 1:nn
                out[ob + ab] = muladd(s, inp[ib + ab], out[ob + ab])
            end
        end
    end
    return out
end

# Run f(item, ws) over `items` on all threads with dynamic scheduling;
# workspaces[t] is private to task t.
function parallel_foreach(f::F, items, workspaces::Vector) where {F}
    n = length(items)
    n == 0 && return nothing
    nt = min(length(workspaces), n)
    if nt == 1
        ws = workspaces[1]
        for i in 1:n
            f(items[i], ws)
        end
        return nothing
    end
    counter = Threads.Atomic{Int}(1)
    @sync for t in 1:nt
        Threads.@spawn _worker(f, items, workspaces[t], counter, n)
    end
    return nothing
end

function _worker(f::F, items, ws, counter, n) where {F}
    while true
        i = Threads.atomic_add!(counter, 1)
        i > n && break
        f(items[i], ws)
    end
    return nothing
end
