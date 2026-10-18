# Near-field interactions by 1D tables (Fortran `bdmk_local.f`, 3D).
#
# A 1D table T (n×n) maps source values (first index j) to targets (second
# index k); `ind[1:2, k]` bounds the nonzero j for target k and
# `ind[1:2, n+1]` bounds the nonzero targets.

# Sparse contraction along dimension 1: out[k, i2, i3] = Σ_j T[j,k] inp[j, i2, i3].
@inline function _sp1!(out, inp, T, ind, K, r2, r3, w, acc::Bool)
    @inbounds for i3 in r3, i2 in r2, k in K
        s = 0.0
        @simd for j in ind[1, k]:ind[2, k]
            s = muladd(T[j, k], inp[j, i2, i3], s)
        end
        if acc
            out[k, i2, i3] = muladd(s, w, out[k, i2, i3])
        else
            out[k, i2, i3] = s
        end
    end
    return nothing
end

# Sparse contraction along dimension 2: out[i1, k, i3] = Σ_j T[j,k] inp[i1, j, i3].
@inline function _sp2!(out, inp, T, ind, r1, K, r3, w, acc::Bool)
    @inbounds for i3 in r3, k in K
        if !acc
            for i1 in r1
                out[i1, k, i3] = 0.0
            end
        end
        for j in ind[1, k]:ind[2, k]
            t = acc ? w * T[j, k] : T[j, k]
            @simd for i1 in r1
                out[i1, k, i3] = muladd(t, inp[i1, j, i3], out[i1, k, i3])
            end
        end
    end
    return nothing
end

# Sparse contraction along dimension 3: out[i1, i2, k] = Σ_j T[j,k] inp[i1, i2, j].
@inline function _sp3!(out, inp, T, ind, r1, r2, K, w, acc::Bool)
    @inbounds for k in K
        if !acc
            for i2 in r2, i1 in r1
                out[i1, i2, k] = 0.0
            end
        end
        for j in ind[1, k]:ind[2, k]
            t = acc ? w * T[j, k] : T[j, k]
            for i2 in r2
                @simd for i1 in r1
                    out[i1, i2, k] = muladd(t, inp[i1, i2, j], out[i1, i2, k])
                end
            end
        end
    end
    return nothing
end

@inline function _sp!(d, out, inp, T, ind, r, w, acc)
    if d == 1
        _sp1!(out, inp, T, ind, r[1], r[2], r[3], w, acc)
    elseif d == 2
        _sp2!(out, inp, T, ind, r[1], r[2], r[3], w, acc)
    else
        _sp3!(out, inp, T, ind, r[1], r[2], r[3], w, acc)
    end
end

@inline _krange(ind, n) = ind[1, n + 1]:ind[2, n + 1]

struct LocalWork
    ff::Array{Float64,3}
    ff2::Array{Float64,3}
    gx::Vector{Array{Float64,3}}      # stage-x buffers for derivative orders 0:2
    gxy::Vector{Array{Float64,3}}     # stage-y buffers for the (dx, dy) pairs
    w1::Vector{Float64}
    w2::Vector{Float64}
end

LocalWork(n::Int) = LocalWork(zeros(n, n, n), zeros(n, n, n),
                              [zeros(n, n, n) for _ in 1:3], [zeros(n, n, n) for _ in 1:6],
                              zeros(n^3), zeros(n^3))

# pot += w * (Tx ⊗ Ty ⊗ Tz) f   (Fortran `tens_prod_to_potloc_3d`).
# tabs = (Tx, Ty, Tz) as (j, k) tables, tabsT their transposes, inds the patterns.
function local_pot!(pot::AbstractArray{Float64,3}, f::AbstractArray{Float64,3}, w::Float64,
                    tabs, tabsT, inds, n::Int, lw::LocalWork)
    Kx = _krange(inds[1], n); Ky = _krange(inds[2], n); Kz = _krange(inds[3], n)
    nx = length(Kx); ny = length(Ky); nz = length(Kz)
    (nx == 0 || ny == 0 || nz == 0) && return nothing
    if nx + ny + nz >= n
        tp3!(pot, f, tabsT[1], tabsT[2], tabsT[3], lw.w1, lw.w2, w, true)
        return nothing
    end
    order = if nx <= ny && ny <= nz
        (1, 2, 3)
    elseif nx <= nz && nz < ny
        (1, 3, 2)
    elseif ny < nx && nx <= nz
        (2, 1, 3)
    elseif ny <= nz && nz < nx
        (2, 3, 1)
    elseif nz < nx && nx <= ny
        (3, 1, 2)
    else
        (3, 2, 1)
    end
    full = 1:n
    Ks = (Kx, Ky, Kz)
    r = [full, full, full]
    d1, d2, d3 = order
    r[d1] = Ks[d1]
    _sp!(d1, lw.ff, f, tabs[d1], inds[d1], r, 1.0, false)
    r[d2] = Ks[d2]
    _sp!(d2, lw.ff2, lw.ff, tabs[d2], inds[d2], r, 1.0, false)
    r[d3] = Ks[d3]
    _sp!(d3, pot, lw.ff2, tabs[d3], inds[d3], r, w, true)
    return nothing
end

# pot/grad/hess += w * derivative tensor products (Fortran
# `tens_prod_to_pgloc_3d` / `tens_prod_to_pghloc_3d`).
# tabs[d][dir] is the table of derivative order d-1 along direction dir.
# outs[m] are the n×n×n outputs for the combos _PGH_COMBOS[1:ncomb].
function local_pgh!(outs, f::AbstractArray{Float64,3}, w::Float64, tabs, inds, n::Int,
                    ifpgh::Int, lw::LocalWork)
    Kx = _krange(inds[1], n); Ky = _krange(inds[2], n); Kz = _krange(inds[3], n)
    (isempty(Kx) || isempty(Ky) || isempty(Kz)) && return nothing
    full = 1:n
    nd1 = ifpgh == 2 ? 2 : 3
    for d in 1:nd1
        _sp1!(lw.gx[d], f, tabs[d][1], inds[1], Kx, full, full, 1.0, false)
    end
    nxy = ifpgh == 2 ? 3 : 6
    for m in 1:nxy
        dx, dy = _PGH_XY[m]
        _sp2!(lw.gxy[m], lw.gx[dx + 1], tabs[dy + 1][2], inds[2], Kx, Ky, full, 1.0, false)
    end
    for m in 1:_ncombos(ifpgh)
        dx, dy, dz = _PGH_COMBOS[m]
        ixy = findfirst(==((dx, dy)), _PGH_XY)
        _sp3!(outs[m], lw.gxy[ixy], tabs[dz + 1][3], inds[3], Kx, Ky, Kz, w, true)
    end
    return nothing
end
