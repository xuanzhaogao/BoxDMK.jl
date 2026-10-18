# Plane-wave operators. A plane-wave expansion is stored as separate real and
# imaginary arrays indexed (u, v, c) -> u + npw*(v-1) + npw^2*(c-1): u and v
# are the x and y frequencies (1:npw) and c the z frequency restricted to the
# first half (1:npw÷2) by conjugate symmetry.
#
# The 1D tables satisfy T[npw+1-u, :] = conj(T[u, :]) because the nodes are
# symmetric, so every complex contraction is done on half the frequencies
# with real GEMMs.

struct PWLevel
    npw::Int
    nh::Int
    nexp::Int
    ts::Vector{Float64}
    ws::Vector{Float64}
    # proxy coefficients -> outgoing PW (table conj(P0))
    B1::Matrix{Float64}             # (p, npw) = [Re T_h' Im T_h']   (z and y stages)
    Ax::Matrix{Float64}             # (npw, p) = [Re T_h; Im T_h]    (x stage)
    # incoming PW -> proxy coefficients, derivative order d = 0, 1, 2
    A1::NTuple{3,Matrix{Float64}}   # (p, npw) = [Re P_h' Im P_h']   (x stage)
    Bv::NTuple{3,Matrix{Float64}}   # (npw, p) = [Re P_h; Im P_h]    (y stage)
    Bc::NTuple{3,Matrix{Float64}}   # (npw, p) = [2Re P_h; -2Im P_h] (z stage)
    kernelft::Vector{Float64}       # (nexp)
    shift_re::Matrix{Float64}       # (npw, 3): exp(i t_j xmin k), k = -1:1
    shift_im::Matrix{Float64}
end

function PWLevel(eps::Float64, ilev::Int, porder::Int, bsize::Float64, bs0::Float64,
                 deltas::AbstractVector{Float64}, wdeltas::AbstractVector{Float64}, prox::Ortho1D)
    npw = pw_terms(eps, ilev)
    ts, ws = pw_nodes(eps, ilev, npw, bs0)
    nh = npw ÷ 2
    nexp = npw * npw * nh
    dsq = bsize / 2
    # P0[j, i] = Σ_k exp(i t_j x_k dsq) U[i, k]
    tmp = [cis(ts[j] * prox.x[k] * dsq) for j in 1:npw, k in 1:porder]
    P0 = tmp * transpose(prox.U)
    Ps = (P0, [im * ts[j] * P0[j, i] for j in 1:npw, i in 1:porder],
          [-ts[j]^2 * P0[j, i] for j in 1:npw, i in 1:porder])
    Th = conj.(P0[1:nh, :])
    B1 = hcat(transpose(real.(Th)), transpose(imag.(Th)))
    Ax = vcat(real.(Th), imag.(Th))
    A1 = ntuple(d -> Matrix(hcat(transpose(real.(Ps[d][1:nh, :])), transpose(imag.(Ps[d][1:nh, :])))), 3)
    Bv = ntuple(d -> vcat(real.(Ps[d][1:nh, :]), imag.(Ps[d][1:nh, :])), 3)
    Bc = ntuple(d -> vcat(2 .* real.(Ps[d][1:nh, :]), -2 .* imag.(Ps[d][1:nh, :])), 3)
    kft = zeros(nexp)
    for k in eachindex(deltas)
        delta = deltas[k]
        ww = [ws[i] * exp(-ts[i]^2 * delta / 4) * sqrt(delta) for i in 1:npw]
        idx = 0
        for c in 1:nh, b in 1:npw, a in 1:npw
            idx += 1
            kft[idx] += wdeltas[k] * ww[a] * ww[b] * ww[c]
        end
    end
    sh = [cis(ts[j] * bsize * k) for j in 1:npw, k in -1:1]
    return PWLevel(npw, nh, nexp, ts, ws, Matrix(B1), Matrix(Ax), A1, Bv, Bc, kft, real.(sh), imag.(sh))
end

struct PWWork
    F1::Vector{Float64}             # (p², npw)
    U::Vector{Float64}              # (p, npw)
    V::Vector{Float64}
    F2r::Vector{Float64}            # (p, npw*nh)
    F2i::Vector{Float64}
    R1::Vector{Float64}             # (npw, npw*nh)
    R2::Vector{Float64}
    G1r::Vector{Vector{Float64}}    # (p, npw*nh) per derivative order
    G1i::Vector{Vector{Float64}}
    G2::Vector{Vector{Float64}}     # (p², npw) per (dx, dy) pair
    locr::Vector{Float64}           # (nexp)
    loci::Vector{Float64}
end

function PWWork(npw::Int, porder::Int)
    nh = npw ÷ 2
    p = porder
    u(m) = Vector{Float64}(undef, m)     # every buffer is written before it is read
    PWWork(u(p * p * npw), u(p * npw), u(p * npw),
           u(p * npw * nh), u(p * npw * nh), u(npw * npw * nh), u(npw * npw * nh),
           [u(p * npw * nh) for _ in 1:3], [u(p * npw * nh) for _ in 1:3],
           [u(p * p * npw) for _ in 1:6], u(npw * npw * nh), u(npw * npw * nh))
end

# Outgoing expansion of real proxy coefficients q (porder^3), optionally
# multiplied by the kernel's Fourier transform (kmul).
#
# The expansion entry (u, col) is stored at pwr[off0 + (col-1)*cs + u].
function proxy2pw!(pwr::AbstractVector{Float64}, pwi::AbstractVector{Float64}, q::AbstractVector{Float64},
                   L::PWLevel, p::Int, w::PWWork, off0::Int = 0, cs::Int = L.npw, kmul::Bool = false)
    npw, nh = L.npw, L.nh
    pp = p * p
    ncol = npw * nh
    GC.@preserve pwr pwi q L w begin
        pF1 = pointer(w.F1); pU = pointer(w.U); pV = pointer(w.V)
        pB1 = pointer(L.B1)
        # z: F1[xy, 1:nh] = Re, F1[xy, nh+1:npw] = Im of Σ_z q[xy, z] T[c, z]
        gemm!(pF1, pp, pointer(q), pp, pB1, p, pp, npw, p, 1.0, false)
        # y (per c): [U Ui] = F1r_c * B1, [V Vi] = F1i_c * B1
        F2r = w.F2r; F2i = w.F2i; U = w.U; V = w.V
        for c in 1:nh
            gemm!(pU, p, pF1 + 8 * (c - 1) * pp, p, pB1, p, p, npw, p, 1.0, false)
            gemm!(pV, p, pF1 + 8 * (nh + c - 1) * pp, p, pB1, p, p, npw, p, 1.0, false)
            base = (c - 1) * npw * p
            @inbounds for v in 1:nh
                ov = base + (v - 1) * p
                ov2 = base + (npw - v) * p
                iu = (v - 1) * p
                ii = (nh + v - 1) * p
                @simd for x in 1:p
                    ur = U[iu + x]; ui = U[ii + x]; vr = V[iu + x]; vi = V[ii + x]
                    F2r[ov + x] = ur - vi
                    F2i[ov + x] = ui + vr
                    F2r[ov2 + x] = ur + vi
                    F2i[ov2 + x] = vr - ui
                end
            end
        end
        # x: R = Ax * F2, then assemble both halves in u
        gemm!(pointer(w.R1), npw, pointer(L.Ax), npw, pointer(F2r), p, npw, ncol, p, 1.0, false)
        gemm!(pointer(w.R2), npw, pointer(L.Ax), npw, pointer(F2i), p, npw, ncol, p, 1.0, false)
        R1 = w.R1; R2 = w.R2
        kft = L.kernelft
        @inbounds for col in 0:(ncol - 1)
            o = col * npw
            od = off0 + col * cs
            @simd for u in 1:nh
                a = R1[o + u]; b = R2[o + nh + u]; c2 = R2[o + u]; d = R1[o + nh + u]
                k1 = kmul ? kft[o + u] : 1.0
                k2 = kmul ? kft[o + npw + 1 - u] : 1.0
                pwr[od + u] = k1 * (a - b)
                pwi[od + u] = k1 * (c2 + d)
                pwr[od + npw + 1 - u] = k2 * (a + b)
                pwi[od + npw + 1 - u] = k2 * (c2 - d)
            end
        end
    end
    return nothing
end

# loc += mp * shift(dx, dy, dz), offsets in -1:1.
function shiftpw_add!(locr, loci, mpr, mpi, L::PWLevel, dx::Int, dy::Int, dz::Int)
    npw, nh = L.npw, L.nh
    Sr, Si = L.shift_re, L.shift_im
    kx, ky, kz = dx + 2, dy + 2, dz + 2
    @inbounds for c in 1:nh
        zr = Sr[c, kz]; zi = Si[c, kz]
        for b in 1:npw
            yr = Sr[b, ky]; yi = Si[b, ky]
            yzr = yr * zr - yi * zi
            yzi = yr * zi + yi * zr
            ob = ((c - 1) * npw + b - 1) * npw
            @simd for a in 1:npw
                wr = Sr[a, kx] * yzr - Si[a, kx] * yzi
                wi = Sr[a, kx] * yzi + Si[a, kx] * yzr
                mr = mpr[ob + a]; mi = mpi[ob + a]
                locr[ob + a] = muladd(mr, wr, muladd(-mi, wi, locr[ob + a]))
                loci[ob + a] = muladd(mr, wi, muladd(mi, wr, loci[ob + a]))
            end
        end
    end
    return nothing
end

# Derivative-order triples for the proxy outputs: pot, then grad, then hess
# (xx, yy, zz, xy, xz, yz).
const _PGH_COMBOS = ((0, 0, 0),
                     (1, 0, 0), (0, 1, 0), (0, 0, 1),
                     (2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1))
const _PGH_XY = ((0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (1, 1))
const _PGH_XYIDX = (1, 2, 3, 1, 4, 5, 1, 6, 2, 3)   # combo -> (dx, dy) pair index

_ncombos(ifpgh::Int) = ifpgh == 1 ? 1 : (ifpgh == 2 ? 4 : 10)

# outs[m] (porder^3) += 2 Re( Σ P_dx[u,x] P_dy[v,y] P_dz[c,z] pw[u,v,c] ) for combo m.
function pw2proxy!(outs, pwr::AbstractVector{Float64}, pwi::AbstractVector{Float64},
                   L::PWLevel, p::Int, ifpgh::Int, w::PWWork, off0::Int = 0, cs::Int = L.npw)
    npw, nh = L.npw, L.nh
    pp = p * p
    ncol = npw * nh
    nd1 = ifpgh == 1 ? 1 : (ifpgh == 2 ? 2 : 3)
    nxy = ifpgh == 1 ? 1 : (ifpgh == 2 ? 3 : 6)
    GC.@preserve outs pwr pwi L w begin
        # x: stacked B_r = [S_r; -D_i], B_i = [S_i; D_r] with S/D = pw[u] ± pw[u'] (u ≤ nh)
        Br = w.R1; Bi = w.R2
        @inbounds for col in 0:(ncol - 1)
            o = col * npw
            od = off0 + col * cs
            @simd for u in 1:nh
                ar = pwr[od + u]; ai = pwi[od + u]
                br = pwr[od + npw + 1 - u]; bi = pwi[od + npw + 1 - u]
                Br[o + u] = ar + br
                Br[o + nh + u] = bi - ai
                Bi[o + u] = ai + bi
                Bi[o + nh + u] = ar - br
            end
        end
        for d in 1:nd1
            gemm!(pointer(w.G1r[d]), p, pointer(L.A1[d]), p, pointer(Br), npw, p, ncol, npw, 1.0, false)
            gemm!(pointer(w.G1i[d]), p, pointer(L.A1[d]), p, pointer(Bi), npw, p, ncol, npw, 1.0, false)
        end
        # y (per c): G2[:, c] = [S2_r -D2_i] * Bv, G2[:, nh+c] = [S2_i D2_r] * Bv
        Xr = w.U; Xi = w.V
        for m in 1:nxy
            dx, dy = _PGH_XY[m]
            G1r = w.G1r[dx + 1]; G1i = w.G1i[dx + 1]
            pBv = pointer(L.Bv[dy + 1])
            pG2 = pointer(w.G2[m])
            for c in 1:nh
                base = (c - 1) * npw * p
                @inbounds for v in 1:nh
                    o1 = base + (v - 1) * p
                    o2 = base + (npw - v) * p
                    ia = (v - 1) * p
                    ib = (nh + v - 1) * p
                    @simd for x in 1:p
                        ar = G1r[o1 + x]; ai = G1i[o1 + x]; br = G1r[o2 + x]; bi = G1i[o2 + x]
                        Xr[ia + x] = ar + br
                        Xr[ib + x] = bi - ai
                        Xi[ia + x] = ai + bi
                        Xi[ib + x] = ar - br
                    end
                end
                gemm!(pG2 + 8 * (c - 1) * pp, p, pointer(Xr), p, pBv, npw, p, p, npw, 1.0, false)
                gemm!(pG2 + 8 * (nh + c - 1) * pp, p, pointer(Xi), p, pBv, npw, p, p, npw, 1.0, false)
            end
        end
        # z: out[xy, z] += G2 * Bc
        for m in 1:_ncombos(ifpgh)
            dz = _PGH_COMBOS[m][3]
            out = outs[m]
            gemm!(pointer(out), pp, pointer(w.G2[_PGH_XYIDX[m]]), pp, pointer(L.Bc[dz + 1]), npw, pp, p, npw, 1.0, true)
        end
    end
    return nothing
end

# Sources of the plane-wave translation at one level, CSR by target:
# srck = source column, srcd = 1 + (dx+1) + 3(dy+1) + 9(dz+1) for the
# offset target - source (self included with offset 0).
struct M2LList
    ptr::Vector{Int}
    k::Vector{Int}
    d::Vector{Int}
end

# loc[:, ind, t, col] = Σ_sources shift(t - s) ⊙ mp[:, ind, s, col] for the
# frequency columns `cols` (a column is the npw x-frequencies at fixed (v, c));
# expansions are stored as (npw, nd, nboxes, ncols) so that a column of all
# boxes is contiguous. Shift factors are grouped by dx:
# loc = Σ_dx wx_dx ⊙ (Σ_{s: dx} (wy wz)_s mp_s).
function m2l_columns!(locr, loci, mpr, mpi, lst::M2LList, L::PWLevel, nd::Int, ntarg::Int,
                      cols::UnitRange{Int}, acc::Vector{Float64})
    npw = L.npw
    Sr, Si = L.shift_re, L.shift_im
    @inbounds for col in cols
        v = (col - 1) % npw + 1
        cz = (col - 1) ÷ npw + 1
        for t in 1:ntarg, ind in 1:nd
            fill!(acc, 0.0)
            for q in lst.ptr[t]:(lst.ptr[t + 1] - 1)
                ks = lst.k[q]
                dd = lst.d[q] - 1
                kx = dd % 3; ky = (dd ÷ 3) % 3 + 1; kz = dd ÷ 9 + 1
                yr = Sr[v, ky]; yi = Si[v, ky]; zr = Sr[cz, kz]; zi = Si[cz, kz]
                sr = yr * zr - yi * zi
                si = yr * zi + yi * zr
                ab = 2 * npw * kx
                @simd for u in 1:npw
                    mr = mpr[u, ind, ks, col]; mi = mpi[u, ind, ks, col]
                    acc[ab + u] = muladd(sr, mr, muladd(-si, mi, acc[ab + u]))
                    acc[ab + npw + u] = muladd(sr, mi, muladd(si, mr, acc[ab + npw + u]))
                end
            end
            @simd for u in 1:npw
                lr = 0.0; li = 0.0
                for kx in 0:2
                    ar = acc[2 * npw * kx + u]; ai = acc[2 * npw * kx + npw + u]
                    wr = Sr[u, kx + 1]; wi = Si[u, kx + 1]
                    lr = muladd(wr, ar, muladd(-wi, ai, lr))
                    li = muladd(wr, ai, muladd(wi, ar, li))
                end
                locr[u, ind, t, col] = lr
                loci[u, ind, t, col] = li
            end
        end
    end
    return nothing
end
