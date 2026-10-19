# Near-field interactions by 1D tables (Fortran `bdmk_local.f`, 3D).
#
# A 1D table T (n×n) maps source values (first index j) to targets (second
# index k); `ind[1:2, k]` bounds the nonzero j for target k and
# `ind[1:2, n+1]` bounds the nonzero targets. Fortran evaluates a pair
# densely (all table entries) when the pattern lengths sum to at least n and
# sparsely (pattern entries only) otherwise; both are reproduced here.

struct LocalWork
    ff::Array{Float64,3}
    ff2::Array{Float64,3}
    gx::Vector{Array{Float64,3}}      # stage-x buffers for derivative orders 0:2
    gxy::Vector{Array{Float64,3}}     # stage-y buffers for the (dx, dy) pairs
    w1::Vector{Float64}
    w2::Vector{Float64}
end

# Buffers are only read where they were written, so they start uninitialized.
LocalWork(n::Int, ar::Arena) = LocalWork(alloc(ar, n, n, n), alloc(ar, n, n, n),
                                         [alloc(ar, n, n, n) for _ in 1:3], [alloc(ar, n, n, n) for _ in 1:6],
                                         alloc(ar, n^3), alloc(ar, n^3))

# Sparse-path near field with GEMMs on pattern-zeroed tables, restricted to
# the nonzero blocks (identical to Fortran's sparse loops up to rounding):
# outs[m] += w * (T_dx ⊗ T_dy ⊗ T_dz) f for the combos m = 1:ncomb.
# T = the LocTab1D, (sx, sy, sz) the slots; lw.gx/gxy are scratch.
function local_sparse!(outs, f::AbstractArray{Float64,3}, w::Float64, T::LocTab1D,
                       sx::Int, sy::Int, sz::Int, n::Int, ncomb::Int, lw::LocalWork)
    kr = T.krange
    kxl, kxh, jxl, jxh = kr[1, sx], kr[2, sx], kr[3, sx], kr[4, sx]
    kyl, kyh, jyl, jyh = kr[1, sy], kr[2, sy], kr[3, sy], kr[4, sy]
    kzl, kzh, jzl, jzh = kr[1, sz], kr[2, sz], kr[3, sz], kr[4, sz]
    nkx = kxh - kxl + 1; nky = kyh - kyl + 1; nkz = kzh - kzl + 1
    (nkx <= 0 || nky <= 0 || nkz <= 0) && return nothing
    njx = jxh - jxl + 1; njy = jyh - jyl + 1; njz = jzh - jzl + 1
    nn = n * n
    nd1 = ncomb == 1 ? 1 : (ncomb == 4 ? 2 : 3)
    nxy = ncomb == 1 ? 1 : (ncomb == 4 ? 3 : 6)
    sl(slot) = (slot - 1) * nn
    GC.@preserve outs f T lw begin
        pf = pointer(f)
        # x: gx[d][Kx, j2, j3] = Σ_j1 TZ_d[j1, k1] f[j1, j2, j3], j2 ∈ Jy, j3 ∈ Jz
        for d in 1:nd1
            pg = pointer(lw.gx[d])
            pA = pointer(T.tabZT[d]) + 8 * (sl(sx) + (kxl - 1) + (jxl - 1) * n)
            for j3 in jzl:jzh
                o = (jyl - 1) * n + (j3 - 1) * nn
                gemm!(pg + 8 * (o + kxl - 1), n, pA, n, pf + 8 * (o + jxl - 1), n, nkx, njy, njx, 1.0, false)
            end
        end
        # y: gxy[m][Kx, Ky, j3] = Σ_j2 gx[dx][Kx, j2, j3] TZ_dy[j2, k2]
        for m in 1:nxy
            dx, dy = _PGH_XY[m]
            pg = pointer(lw.gx[dx + 1]); pq = pointer(lw.gxy[m])
            pB = pointer(T.tabZ[dy + 1]) + 8 * (sl(sy) + (jyl - 1) + (kyl - 1) * n)
            for j3 in jzl:jzh
                gemm!(pq + 8 * ((kxl - 1) + (kyl - 1) * n + (j3 - 1) * nn), n,
                      pg + 8 * ((kxl - 1) + (jyl - 1) * n + (j3 - 1) * nn), n, pB, n, nkx, nky, njy, 1.0, false)
            end
        end
        # z: out[Kx, k2, Kz] += w Σ_j3 gxy[Kx, k2, j3] TZ_dz[j3, k3]
        for m in 1:ncomb
            dz = _PGH_COMBOS[m][3]
            pq = pointer(lw.gxy[_PGH_XYIDX[m]])
            po = outptr(outs, m)
            pB = pointer(T.tabZ[dz + 1]) + 8 * (sl(sz) + (jzl - 1) + (kzl - 1) * n)
            for k2 in kyl:kyh
                gemm!(po + 8 * ((kxl - 1) + (k2 - 1) * n + (kzl - 1) * nn), nn,
                      pq + 8 * ((kxl - 1) + (k2 - 1) * n + (jzl - 1) * nn), nn, pB, n, nkx, nkz, njz, w, true)
            end
        end
    end
    return nothing
end
