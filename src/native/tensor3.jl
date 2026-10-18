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

# out (+)= α (D applied along dimension `dim` of the n×n×n tensor inp);
# DT = transpose(D).
function apply_dim!(out::AbstractArray{Float64}, inp::AbstractArray{Float64}, D::Matrix{Float64},
                    DT::Matrix{Float64}, dim::Int, n::Int, α::Float64, add::Bool)
    nn = n * n
    GC.@preserve out inp D DT begin
        po = pointer(out); pi = pointer(inp)
        if dim == 1
            gemm!(po, n, pointer(D), n, pi, n, n, nn, n, α, add)
        elseif dim == 2
            gemm_batch!(po, n, nn, pi, n, nn, pointer(DT), n, n, n, n, n, α, add)
        else
            gemm!(po, nn, pi, nn, pointer(DT), n, nn, n, n, α, add)
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
        old = get_zero_subnormals()
        set_zero_subnormals(true)
        try
            for i in 1:n
                f(items[i], ws)
            end
        finally
            set_zero_subnormals(old)
        end
        return nothing
    end
    counter = Threads.Atomic{Int}(1)
    @sync for t in 1:nt
        Threads.@spawn _worker(f, items, workspaces[t], counter, n)
    end
    return nothing
end

# Subnormals are flushed to zero while working: products of the rapidly
# decaying tables and densities otherwise underflow into slow subnormal
# arithmetic, at no accuracy benefit.
function _worker(f::F, items, ws, counter, n) where {F}
    old = get_zero_subnormals()
    set_zero_subnormals(true)
    try
        while true
            i = Threads.atomic_add!(counter, 1)
            i > n && break
            f(items[i], ws)
        end
    finally
        set_zero_subnormals(old)
    end
    return nothing
end

# Children -> parent anterpolation, sharing stages among the 8 children:
# out (+)= α Σ_children (A[sx] ⊗ B[sy] ⊗ C[sz]) q_child with the z stage applied
# per child, the y stage per (sx, sy) and the x stage per sx. Children are of
# order mi, the parent of order mo: A[s] (mo × mi), Bt[s] = B[s]' and
# Ct[s] = C[s]' (mi × mo). qs(slot) returns a child's data or nothing,
# slot = 1 + sx' + 2sy' + 4sz'. Z, Y need mi*mi*mo and mi*mo*mo entries.
function children2parent!(out::AbstractArray{Float64}, qs::F, A::NTuple{2,Matrix{Float64}},
                          Bt::NTuple{2,Matrix{Float64}}, Ct::NTuple{2,Matrix{Float64}}, mi::Int, mo::Int,
                          Z::Vector{Float64}, Y::Vector{Float64}, add::Bool, α::Float64 = 1.0) where {F}
    mii = mi * mi
    started = add
    GC.@preserve out Z Y A Bt Ct begin
        pZ = pointer(Z); pY = pointer(Y)
        for sx in 1:2
            haveY = false
            for sy in 1:2
                haveZ = false
                for sz in 1:2
                    q = qs(1 + (sx - 1) + 2 * (sy - 1) + 4 * (sz - 1))
                    q === nothing && continue
                    GC.@preserve q gemm!(pZ, mii, pointer(q), mii, pointer(Ct[sz]), mi, mii, mo, mi, 1.0, haveZ)
                    haveZ = true
                end
                haveZ || continue
                gemm_batch!(pY, mi, mi * mo, pZ, mi, mii, pointer(Bt[sy]), mi, mi, mo, mi, mo, 1.0, haveY)
                haveY = true
            end
            haveY || continue
            gemm!(pointer(out), mo, pointer(A[sx]), mo, pY, mi, mo, mo * mo, mi, α, started)
            started = true
        end
        started || fill!(out, 0.0)
    end
    return out
end

# Parent -> children interpolation, sharing stages: x per sx, y per (sx, sy),
# z per child. Parent of order mi, children of order mo; A[s] (mo × mi),
# Bt[s], Ct[s] (mi × mo). outs(slot) returns a child's output (added to) or
# nothing. X, Y need mo*mi*mi and mo*mo*mi entries.
function parent2children!(outs::F, phi::AbstractArray{Float64}, A::NTuple{2,Matrix{Float64}},
                          Bt::NTuple{2,Matrix{Float64}}, Ct::NTuple{2,Matrix{Float64}}, mi::Int, mo::Int,
                          X::Vector{Float64}, Y::Vector{Float64}) where {F}
    moo = mo * mo
    GC.@preserve phi X Y A Bt Ct begin
        pX = pointer(X); pY = pointer(Y)
        for sx in 1:2
            computed = false
            for sy in 1:2
                haveY = false
                for sz in 1:2
                    o = outs(1 + (sx - 1) + 2 * (sy - 1) + 4 * (sz - 1))
                    o === nothing && continue
                    if !computed
                        gemm!(pX, mo, pointer(A[sx]), mo, pointer(phi), mi, mo, mi * mi, mi, 1.0, false)
                        computed = true
                    end
                    if !haveY
                        gemm_batch!(pY, mo, moo, pX, mo, mo * mi, pointer(Bt[sy]), mi, mo, mo, mi, mi, 1.0, false)
                        haveY = true
                    end
                    GC.@preserve o gemm!(pointer(o), moo, pY, moo, pointer(Ct[sz]), mi, moo, mo, mi, 1.0, true)
                end
            end
        end
    end
    return nothing
end

# Per-task workspaces, each allocated (and first touched) on its own thread.
function make_workspaces(f::F) where {F}
    tasks = [Threads.@spawn f() for _ in 1:Threads.nthreads()]
    return [fetch(t) for t in tasks]
end

# Fill `a` with v using all threads (parallel first touch of fresh memory).
function pfill!(a::Array{Float64}, v::Float64 = 0.0)
    n = length(a)
    nt = Threads.nthreads()
    (nt == 1 || n < 1 << 16) && return fill!(a, v)
    nchunk = 4 * nt
    ranges = [(1 + div((i - 1) * n, nchunk)):div(i * n, nchunk) for i in 1:nchunk]
    parallel_foreach(ranges, fill(nothing, nt)) do r, _
        fill!(view(a, r), v)
    end
    return a
end

pzeros(dims::Int...) = pfill!(Array{Float64}(undef, dims...))

# Large solver buffers are taken from malloc and released when the solve ends,
# so they neither trigger nor burden garbage collection.
struct Arena
    ptrs::Vector{Ptr{Cvoid}}
    lock::ReentrantLock
end
Arena() = Arena(Ptr{Cvoid}[], ReentrantLock())

function alloc(ar::Arena, dims::Int...)
    n = prod(dims)
    n == 0 && return Array{Float64}(undef, dims...)
    p = Libc.malloc(8 * n)
    p == C_NULL && throw(OutOfMemoryError())
    lock(() -> push!(ar.ptrs, p), ar.lock)
    return unsafe_wrap(Array, Ptr{Float64}(p), dims)
end

function release!(ar::Arena)
    lock(ar.lock) do
        foreach(Libc.free, ar.ptrs)
        empty!(ar.ptrs)
    end
    return nothing
end
