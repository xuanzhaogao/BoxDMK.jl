# Plane-wave operators. A plane-wave expansion is stored as separate real and
# imaginary arrays indexed (u, v, c) -> u + npw*(v-1) + npw^2*(c-1): u and v
# are the x and y frequencies (1:npw) and c the z frequency restricted to the
# first half (1:npw÷2) by conjugate symmetry.
#
# The 1D tables satisfy T[npw+1-u, :] = conj(T[u, :]) because the nodes are
# symmetric, so every complex contraction is done on half the frequencies
# with real GEMMs.

# Conversion tables between plane waves and a tensor-product basis of order m
# (proxy coefficients, m = porder, or leaf grid values, m = norder).
struct PWTables
    m::Int
    # basis -> outgoing PW (table T = conj(P0))
    B1::Matrix{Float64}             # (m, npw) = [Re T_h' Im T_h']   (z and y stages)
    Ax::Matrix{Float64}             # (npw, m) = [Re T_h; Im T_h]    (x stage)
    # incoming PW -> basis, derivative order d = 0, 1, 2
    A1::NTuple{3,Matrix{Float64}}   # (m, npw) = [Re P_h' Im P_h']   (x stage)
    Bv::NTuple{3,Matrix{Float64}}   # (npw, m) = [Re P_h; Im P_h]    (y stage)
    Bc::NTuple{3,Matrix{Float64}}   # (npw, m) = [2Re P_h; -2Im P_h] (z stage)
end

# Tables for the proxy basis composed with `fin` (proxy coefficients from an
# input basis, m×porder ... here porder × m) on the formation side and `fout`
# (output basis from proxy coefficients, m × porder) on the evaluation side.
function PWTables(P0::Matrix{ComplexF64}, ts::Vector{Float64}, fin::Matrix{Float64}, fout::Matrix{Float64})
    npw, porder = size(P0)
    nh = npw ÷ 2
    Ps = (P0, [im * ts[j] * P0[j, i] for j in 1:npw, i in 1:porder],
          [-ts[j]^2 * P0[j, i] for j in 1:npw, i in 1:porder])
    Th = conj.(P0[1:nh, :]) * fin
    B1 = Matrix(hcat(transpose(real.(Th)), transpose(imag.(Th))))
    Ax = vcat(real.(Th), imag.(Th))
    Ph = ntuple(d -> Ps[d][1:nh, :] * transpose(fout), 3)
    A1 = ntuple(d -> Matrix(hcat(transpose(real.(Ph[d])), transpose(imag.(Ph[d])))), 3)
    Bv = ntuple(d -> vcat(real.(Ph[d]), imag.(Ph[d])), 3)
    Bc = ntuple(d -> vcat(2 .* real.(Ph[d]), -2 .* imag.(Ph[d])), 3)
    return PWTables(size(fin, 2), B1, Ax, A1, Bv, Bc)
end

struct PWLevel
    npw::Int
    nh::Int
    nexp::Int
    ts::Vector{Float64}
    ws::Vector{Float64}
    prox::PWTables                  # proxy coefficients <-> PW
    leaf::Union{Nothing,PWTables}   # leaf grid values <-> PW (fused with the proxy maps)
    kernelft::Vector{Float64}       # (nexp)
    shift_re::Matrix{Float64}       # (npw, 3): exp(i t_j xmin k), k = -1:1
    shift_im::Matrix{Float64}
end

# `leafmaps = (den2pc * sc³ᐟ³, poteval)` adds the fused leaf tables.
function PWLevel(eps::Float64, ilev::Int, porder::Int, bsize::Float64, bs0::Float64,
                 deltas::AbstractVector{Float64}, wdeltas::AbstractVector{Float64}, prox::Ortho1D,
                 leafmaps = nothing)
    npw = pw_terms(eps, ilev)
    ts, ws = pw_nodes(eps, ilev, npw, bs0)
    nh = npw ÷ 2
    nexp = npw * npw * nh
    dsq = bsize / 2
    # P0[j, i] = Σ_k exp(i t_j x_k dsq) U[i, k]
    tmp = [cis(ts[j] * prox.x[k] * dsq) for j in 1:npw, k in 1:porder]
    P0 = tmp * transpose(prox.U)
    eye = Matrix{Float64}(I, porder, porder)
    ptab = PWTables(P0, ts, eye, eye)
    ltab = leafmaps === nothing ? nothing : PWTables(P0, ts, leafmaps[1], leafmaps[2])
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
    return PWLevel(npw, nh, nexp, ts, ws, ptab, ltab, kft, real.(sh), imag.(sh))
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
function proxy2pw!(pwr::AbstractVector{Float64}, pwi::AbstractVector{Float64}, q::AbstractArray{Float64},
                   L::PWLevel, tb::PWTables, w::PWWork, off0::Int = 0, cs::Int = L.npw, kmul::Bool = false)
    npw, nh = L.npw, L.nh
    p = tb.m
    pp = p * p
    ncol = npw * nh
    GC.@preserve pwr pwi q tb w begin
        pF1 = pointer(w.F1); pU = pointer(w.U); pV = pointer(w.V)
        pB1 = pointer(tb.B1)
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
        gemm!(pointer(w.R1), npw, pointer(tb.Ax), npw, pointer(F2r), p, npw, ncol, p, 1.0, false)
        gemm!(pointer(w.R2), npw, pointer(tb.Ax), npw, pointer(F2i), p, npw, ncol, p, 1.0, false)
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
                   L::PWLevel, tb::PWTables, ifpgh::Int, w::PWWork, off0::Int = 0, cs::Int = L.npw)
    npw, nh = L.npw, L.nh
    p = tb.m
    pp = p * p
    ncol = npw * nh
    nd1 = ifpgh == 1 ? 1 : (ifpgh == 2 ? 2 : 3)
    nxy = ifpgh == 1 ? 1 : (ifpgh == 2 ? 3 : 6)
    GC.@preserve outs pwr pwi tb w begin
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
            gemm!(pointer(w.G1r[d]), p, pointer(tb.A1[d]), p, pointer(Br), npw, p, ncol, npw, 1.0, false)
            gemm!(pointer(w.G1i[d]), p, pointer(tb.A1[d]), p, pointer(Bi), npw, p, ncol, npw, 1.0, false)
        end
        # y (per c): G2[:, c] = [S2_r -D2_i] * Bv, G2[:, nh+c] = [S2_i D2_r] * Bv
        Xr = w.U; Xi = w.V
        for m in 1:nxy
            dx, dy = _PGH_XY[m]
            G1r = w.G1r[dx + 1]; G1i = w.G1i[dx + 1]
            pBv = pointer(tb.Bv[dy + 1])
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
            gemm!(pointer(out), pp, pointer(w.G2[_PGH_XYIDX[m]]), pp, pointer(tb.Bc[dz + 1]), npw, pp, p, npw, 1.0, true)
        end
    end
    return nothing
end

# Sources of the plane-wave translation at one level, CSR by target and
# sorted by x-offset: k = source column, (kx, ky, kz) = offset index in 1:3
# of target - source (self included), nkx[:, t] = sources per kx.
struct M2LList
    ptr::Vector{Int}
    k::Vector{Int}
    kyz::Vector{Int}            # ky + 3(kz - 1)
    nkx::Matrix{Int}
end

function M2LList(srcs::Vector{Vector{NTuple{4,Int}}})
    ptr = ones(Int, length(srcs) + 1)
    k = Int[]; kyz = Int[]
    nkx = zeros(Int, 3, length(srcs))
    for (t, list) in enumerate(srcs)
        for (ks, dx, dy, dz) in sort(list; by = x -> x[2])
            push!(k, ks); push!(kyz, (dy + 2) + 3 * (dz + 1))
            nkx[dx + 2, t] += 1
        end
        ptr[t + 1] = length(k) + 1
    end
    return M2LList(ptr, k, kyz, nkx)
end

# loc[:, ind, t, col] = Σ_sources shift(t - s) ⊙ mp[:, ind, s, col] for the
# frequency columns `cols` (a column is the npw x-frequencies at fixed (v, c));
# expansions are stored as (npw, nd, nboxes, ncols) so that a column of all
# boxes is contiguous. Shift factors are grouped by dx:
# loc = Σ_dx wx_dx ⊙ (Σ_{s: dx} (wy wz)_s mp_s), evaluated in blocks of 8
# frequencies held in registers. `scr` holds per-source scalars.
function m2l_columns!(locr::Array{Float64,4}, loci::Array{Float64,4}, mpr::Array{Float64,4}, mpi::Array{Float64,4},
                      lst::M2LList, L::PWLevel, nd::Int, ntarg::Int, cols::UnitRange{Int}, scr::Vector{Float64})
    npw = L.npw
    Sr, Si = L.shift_re, L.shift_im
    cstr = npw * nd * size(mpr, 3)            # column stride
    bstr = npw * nd                           # box stride
    GC.@preserve locr loci mpr mpi scr Sr Si begin
        pmr = pointer(mpr); pmi = pointer(mpi); plr = pointer(locr); pli = pointer(loci)
        pS = pointer(scr); pSr = pointer(Sr); pSi = pointer(Si)
        for col in cols
            v = (col - 1) % npw + 1
            cz = (col - 1) ÷ npw + 1
            cbase = (col - 1) * cstr
            # the 9 (y, z) shift products of this column
            wyz = ntuple(9) do j
                ky = (j - 1) % 3 + 1; kz = (j - 1) ÷ 3 + 1
                @inbounds yr = Sr[v, ky]; @inbounds yi = Si[v, ky]
                @inbounds zr = Sr[cz, kz]; @inbounds zi = Si[cz, kz]
                (yr * zr - yi * zi, yr * zi + yi * zr)
            end
            for t in 1:ntarg
                q0 = lst.ptr[t]
                nq = lst.ptr[t + 1] - q0
                @inbounds for i in 0:(nq - 1)
                    q = q0 + i
                    sr, si = wyz[lst.kyz[q]]
                    scr[2i + 1] = sr
                    scr[2i + 2] = si
                end
                n1 = lst.nkx[1, t]; n2 = lst.nkx[2, t]
                for ind in 1:nd
                    ib = cbase + npw * (ind - 1)
                    u0 = 0
                    while u0 < npw
                        # 16 frequencies per pass (two vectors, masked at the end);
                        # the sr and si products use separate accumulators so that
                        # no FMA waits on another.
                        m1 = _rowmask(npw - u0)
                        m2 = _rowmask(npw - u0 - 8)
                        lr1 = zero(V8); li1 = zero(V8); lr2 = zero(V8); li2 = zero(V8)
                        i = 0
                        for kxg in 0:2
                            iend = kxg == 0 ? n1 : (kxg == 1 ? n1 + n2 : nq)
                            a1 = zero(V8); b1 = zero(V8); c1 = zero(V8); d1 = zero(V8)
                            a2 = zero(V8); b2 = zero(V8); c2 = zero(V8); d2 = zero(V8)
                            while i < iend
                                sr = V8(unsafe_load(pS, 2i + 1))
                                si = V8(unsafe_load(pS, 2i + 2))
                                o = ib + bstr * (@inbounds(lst.k[q0 + i]) - 1) + u0
                                mr1 = vload(V8, pmr + 8o, m1); mi1 = vload(V8, pmi + 8o, m1)
                                mr2 = vload(V8, pmr + 8(o + 8), m2); mi2 = vload(V8, pmi + 8(o + 8), m2)
                                a1 = muladd(sr, mr1, a1); b1 = muladd(si, mi1, b1)
                                c1 = muladd(sr, mi1, c1); d1 = muladd(si, mr1, d1)
                                a2 = muladd(sr, mr2, a2); b2 = muladd(si, mi2, b2)
                                c2 = muladd(sr, mi2, c2); d2 = muladd(si, mr2, d2)
                                i += 1
                            end
                            ar1 = a1 - b1; ai1 = c1 + d1; ar2 = a2 - b2; ai2 = c2 + d2
                            pw = 8 * (u0 + kxg * npw)
                            wr1 = vload(V8, pSr + pw, m1); wi1 = vload(V8, pSi + pw, m1)
                            wr2 = vload(V8, pSr + pw + 64, m2); wi2 = vload(V8, pSi + pw + 64, m2)
                            lr1 = muladd(wr1, ar1, muladd(-wi1, ai1, lr1))
                            li1 = muladd(wr1, ai1, muladd(wi1, ar1, li1))
                            lr2 = muladd(wr2, ar2, muladd(-wi2, ai2, lr2))
                            li2 = muladd(wr2, ai2, muladd(wi2, ar2, li2))
                        end
                        o = ib + bstr * (t - 1) + u0
                        vstore(lr1, plr + 8o, m1); vstore(li1, pli + 8o, m1)
                        vstore(lr2, plr + 8(o + 8), m2); vstore(li2, pli + 8(o + 8), m2)
                        u0 += 16
                    end
                end
            end
        end
    end
    return nothing
end
