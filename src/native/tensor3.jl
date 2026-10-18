# Dense 3D tensor-product kernels. Every inner loop runs along the first
# (contiguous) array index so that it vectorizes.

# out[a,b,c] (+)= α Σ A[a,i] B[b,j] C[c,k] inp[i,j,k]
# inp: n1*n2*n3 entries, A: (m1,n1), B: (m2,n2), C: (m3,n3).
# w1 needs m1*n2*n3 entries, w2 needs m1*m2*n3 entries.
function tp3!(out::AbstractArray{Float64}, inp::AbstractArray{Float64},
              A::AbstractMatrix{Float64}, B::AbstractMatrix{Float64}, C::AbstractMatrix{Float64},
              w1::Vector{Float64}, w2::Vector{Float64}, α::Float64, add::Bool)
    m1, n1 = size(A)
    m2, n2 = size(B)
    m3, n3 = size(C)
    # x: w1[a, j, k] = Σ_i A[a,i] inp[i, j, k]
    @inbounds for jk in 0:(n2 * n3 - 1)
        ob = jk * m1
        ib = jk * n1
        for a in 1:m1
            w1[ob + a] = 0.0
        end
        for i in 1:n1
            s = inp[ib + i]
            @simd for a in 1:m1
                w1[ob + a] = muladd(A[a, i], s, w1[ob + a])
            end
        end
    end
    # y: w2[a, b, k] = Σ_j B[b,j] w1[a, j, k]
    @inbounds for k in 0:(n3 - 1)
        for b in 1:m2
            ob = (k * m2 + b - 1) * m1
            for a in 1:m1
                w2[ob + a] = 0.0
            end
            for j in 1:n2
                s = B[b, j]
                ib = (k * n2 + j - 1) * m1
                @simd for a in 1:m1
                    w2[ob + a] = muladd(s, w1[ib + a], w2[ob + a])
                end
            end
        end
    end
    # z: out[ab, c] (+)= α Σ_k C[c,k] w2[ab, k]
    mab = m1 * m2
    @inbounds for c in 1:m3
        ob = (c - 1) * mab
        if !add
            for ab in 1:mab
                out[ob + ab] = 0.0
            end
        end
        for k in 1:n3
            s = α * C[c, k]
            ib = (k - 1) * mab
            @simd for ab in 1:mab
                out[ob + ab] = muladd(s, w2[ib + ab], out[ob + ab])
            end
        end
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
