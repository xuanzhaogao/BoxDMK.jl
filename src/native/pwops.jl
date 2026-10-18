# Plane-wave operators. A plane-wave expansion is stored as separate real and
# imaginary arrays indexed (a, b, c) -> a + npw*(b-1) + npw^2*(c-1), with a/b
# the x/y frequencies (1:npw) and c the z frequency restricted to the first
# half (1:npw÷2) using conjugate symmetry.

struct PWLevel
    npw::Int
    nh::Int
    nexp::Int
    ts::Vector{Float64}
    ws::Vector{Float64}
    # proxy coefficients -> outgoing PW: T[a, i] (npw × porder)
    c2pw_re::Matrix{Float64}
    c2pw_im::Matrix{Float64}
    # incoming PW -> proxy coefficients for derivative order d = 0, 1, 2:
    # P[d+1][a, x] (npw × porder) and its transpose PT[d+1][x, a]
    pw2c_re::NTuple{3,Matrix{Float64}}
    pw2c_im::NTuple{3,Matrix{Float64}}
    pw2cT_re::NTuple{3,Matrix{Float64}}
    pw2cT_im::NTuple{3,Matrix{Float64}}
    kernelft::Vector{Float64}           # (nexp)
    shift_re::Matrix{Float64}           # (npw, 3): exp(i t_j xmin k), k = -1:1
    shift_im::Matrix{Float64}
end

function PWLevel(eps::Float64, ilev::Int, porder::Int, bsize::Float64, bs0::Float64,
                 deltas::AbstractVector{Float64}, wdeltas::AbstractVector{Float64}, prox::Ortho1D)
    npw = pw_terms(eps, ilev)
    ts, ws = pw_nodes(eps, ilev, npw, bs0)
    nh = npw ÷ 2
    nexp = npw * npw * nh
    dsq = bsize / 2
    # P[j, i] = Σ_k exp(i t_j x_k dsq) U[i, k]
    tmp = [cis(ts[j] * prox.x[k] * dsq) for j in 1:npw, k in 1:porder]
    P0 = tmp * transpose(prox.U)
    P1 = [im * ts[j] * P0[j, i] for j in 1:npw, i in 1:porder]
    P2 = [-ts[j]^2 * P0[j, i] for j in 1:npw, i in 1:porder]
    Ps = (P0, P1, P2)
    C = conj.(P0)
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
    return PWLevel(npw, nh, nexp, ts, ws, real.(C), imag.(C),
                   ntuple(d -> real.(Ps[d]), 3), ntuple(d -> imag.(Ps[d]), 3),
                   ntuple(d -> Matrix(transpose(real.(Ps[d]))), 3),
                   ntuple(d -> Matrix(transpose(imag.(Ps[d]))), 3),
                   kft, real.(sh), imag.(sh))
end

struct PWWork
    f1r::Vector{Float64}; f1i::Vector{Float64}
    f2r::Vector{Float64}; f2i::Vector{Float64}
    g1r::Vector{Vector{Float64}}; g1i::Vector{Vector{Float64}}
    g2r::Vector{Vector{Float64}}; g2i::Vector{Vector{Float64}}
    locr::Vector{Float64}; loci::Vector{Float64}
end

function PWWork(npw::Int, porder::Int)
    nh = npw ÷ 2
    p = porder
    n1 = max(p * p * nh, p * npw * nh)
    n2 = max(p * npw * nh, p * p * nh)
    PWWork(zeros(n1), zeros(n1), zeros(n2), zeros(n2),
           [zeros(p * npw * nh) for _ in 1:3], [zeros(p * npw * nh) for _ in 1:3],
           [zeros(p * p * nh) for _ in 1:6], [zeros(p * p * nh) for _ in 1:6],
           zeros(npw * npw * nh), zeros(npw * npw * nh))
end

# Outgoing expansion of real proxy coefficients q (porder^3).
function proxy2pw!(pwr::AbstractVector{Float64}, pwi::AbstractVector{Float64}, q::AbstractVector{Float64},
                   L::PWLevel, porder::Int, w::PWWork)
    npw, nh, p = L.npw, L.nh, porder
    Tr, Ti = L.c2pw_re, L.c2pw_im
    f1r, f1i, f2r, f2i = w.f1r, w.f1i, w.f2r, w.f2i
    pp = p * p
    # z (half range): f1[ij, c] = Σ_k T[c, k] q[ij, k]
    @inbounds for c in 1:nh
        ob = (c - 1) * pp
        for ij in 1:pp
            f1r[ob + ij] = 0.0
            f1i[ob + ij] = 0.0
        end
        for k in 1:p
            tr = Tr[c, k]; ti = Ti[c, k]
            ib = (k - 1) * pp
            @simd for ij in 1:pp
                qv = q[ib + ij]
                f1r[ob + ij] = muladd(tr, qv, f1r[ob + ij])
                f1i[ob + ij] = muladd(ti, qv, f1i[ob + ij])
            end
        end
    end
    # y: f2[i, b, c] = Σ_j T[b, j] f1[i, j, c]
    @inbounds for c in 1:nh, b in 1:npw
        ob = ((c - 1) * npw + b - 1) * p
        for i in 1:p
            f2r[ob + i] = 0.0
            f2i[ob + i] = 0.0
        end
        for j in 1:p
            tr = Tr[b, j]; ti = Ti[b, j]
            ib = ((c - 1) * p + j - 1) * p
            @simd for i in 1:p
                xr = f1r[ib + i]; xi = f1i[ib + i]
                f2r[ob + i] = muladd(tr, xr, muladd(-ti, xi, f2r[ob + i]))
                f2i[ob + i] = muladd(tr, xi, muladd(ti, xr, f2i[ob + i]))
            end
        end
    end
    # x: pw[a, bc] = Σ_i T[a, i] f2[i, bc]
    @inbounds for bc in 0:(npw * nh - 1)
        ob = bc * npw
        ib = bc * p
        for a in 1:npw
            pwr[ob + a] = 0.0
            pwi[ob + a] = 0.0
        end
        for i in 1:p
            xr = f2r[ib + i]; xi = f2i[ib + i]
            @simd for a in 1:npw
                tr = Tr[a, i]; ti = Ti[a, i]
                pwr[ob + a] = muladd(tr, xr, muladd(-ti, xi, pwr[ob + a]))
                pwi[ob + a] = muladd(tr, xi, muladd(ti, xr, pwi[ob + a]))
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

_ncombos(ifpgh::Int) = ifpgh == 1 ? 1 : (ifpgh == 2 ? 4 : 10)

# outs[m] (porder^3) += 2 Re( Σ P_dx[a,x] P_dy[b,y] P_dz[c,z] pw[a,b,c] ) for combo m.
function pw2proxy!(outs, pwr::AbstractVector{Float64}, pwi::AbstractVector{Float64},
                   L::PWLevel, porder::Int, ifpgh::Int, w::PWWork)
    npw, nh, p = L.npw, L.nh, porder
    ncomb = _ncombos(ifpgh)
    nd1 = ifpgh == 1 ? 1 : (ifpgh == 2 ? 2 : 3)
    # x: g1[d][x, bc] = Σ_a P_d[a, x] pw[a, bc]
    for d in 1:nd1
        PTr = L.pw2cT_re[d]; PTi = L.pw2cT_im[d]
        g1r = w.g1r[d]; g1i = w.g1i[d]
        @inbounds for bc in 0:(npw * nh - 1)
            ob = bc * p
            ib = bc * npw
            for x in 1:p
                g1r[ob + x] = 0.0
                g1i[ob + x] = 0.0
            end
            for a in 1:npw
                xr = pwr[ib + a]; xi = pwi[ib + a]
                @simd for x in 1:p
                    tr = PTr[x, a]; ti = PTi[x, a]
                    g1r[ob + x] = muladd(tr, xr, muladd(-ti, xi, g1r[ob + x]))
                    g1i[ob + x] = muladd(tr, xi, muladd(ti, xr, g1i[ob + x]))
                end
            end
        end
    end
    # y: g2[(dx,dy)][x, y, c] = Σ_b P_dy[b, y] g1[dx][x, b, c]
    nxy = ifpgh == 1 ? 1 : (ifpgh == 2 ? 3 : 6)
    for m in 1:nxy
        dx, dy = _PGH_XY[m]
        Pr = L.pw2c_re[dy + 1]; Pi = L.pw2c_im[dy + 1]
        g1r = w.g1r[dx + 1]; g1i = w.g1i[dx + 1]
        g2r = w.g2r[m]; g2i = w.g2i[m]
        @inbounds for c in 1:nh, y in 1:p
            ob = ((c - 1) * p + y - 1) * p
            for x in 1:p
                g2r[ob + x] = 0.0
                g2i[ob + x] = 0.0
            end
            for b in 1:npw
                tr = Pr[b, y]; ti = Pi[b, y]
                ib = ((c - 1) * npw + b - 1) * p
                @simd for x in 1:p
                    xr = g1r[ib + x]; xi = g1i[ib + x]
                    g2r[ob + x] = muladd(tr, xr, muladd(-ti, xi, g2r[ob + x]))
                    g2i[ob + x] = muladd(tr, xi, muladd(ti, xr, g2i[ob + x]))
                end
            end
        end
    end
    # z: out[xy, z] += 2 Σ_c Re(P_dz[c, z] g2[xy, c])
    pp = p * p
    for m in 1:ncomb
        dx, dy, dz = _PGH_COMBOS[m]
        ixy = findfirst(==((dx, dy)), _PGH_XY)
        g2r = w.g2r[ixy]; g2i = w.g2i[ixy]
        Pr = L.pw2c_re[dz + 1]; Pi = L.pw2c_im[dz + 1]
        out = outs[m]
        @inbounds for z in 1:p
            ob = (z - 1) * pp
            for c in 1:nh
                tr = 2 * Pr[c, z]; ti = -2 * Pi[c, z]
                ib = (c - 1) * pp
                @simd for xy in 1:pp
                    out[ob + xy] = muladd(tr, g2r[ib + xy], muladd(ti, g2i[ib + xy], out[ob + xy]))
                end
            end
        end
    end
    return nothing
end
