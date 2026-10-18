# 1D near-field Gaussian interaction tables (Fortran `bdmk_local_tables.f`).
#
# tab[:, :, slot] maps source-box values (first index) to target values
# (second index) for the 1D offset `slot - 7` in units of a quarter of the
# source box (same-level offsets 0, ±4; coarse/fine offsets ±1, ±2, ±3, ±6).

const NLOCTAB = 6
const NLOCTAB2 = 2 * NLOCTAB + 1

struct LocTab1D
    tab::Array{Float64,3}       # (n, n, 13)
    tabT::Array{Float64,3}      # tab transposed per slot (dense path)
    tabx::Array{Float64,3}
    tabxx::Array{Float64,3}
    ind::Array{Int,3}           # (2, n+1, 13) sparse pattern (Fortran ind_loc)
    # Sparse path: tables (derivative order 0:2) with the entries outside the
    # pattern set to zero, their transposes, and per slot the target range K
    # and the source envelope J (krange[:, slot] = (klo, khi, jlo, jhi)).
    tabZ::NTuple{3,Array{Float64,3}}
    tabZT::NTuple{3,Array{Float64,3}}
    krange::Matrix{Int}
end

const _SQRTPIH = 0.886226925452757940959713778283913

function _legetab_recur!(fint, fx, fxx, fp, fpp, m, targ, lambda, boxdim)
    lambda2 = lambda * lambda
    a = (-1 - targ) / lambda
    b = (1 - targ) / lambda
    erfa = _erf(a); erfb = _erf(b)
    expa = exp(-a * a); expb = exp(-b * b)
    d1 = lambda * (expb - expa)
    d2 = lambda * (expb + expa)
    fint[1] = _SQRTPIH * (erfb - erfa)
    m >= 2 && (fint[2] = -d1 / 2 + targ * fint[1])
    if m >= 3
        fint[3] = 0.75 * (-d2 - targ * d1 + (lambda2 - 2.0 / 3 + 2 * targ^2) * fint[1])
    end
    if m >= 4
        fint[4] = 5.0 * (-d1 + 2 * targ * (4.0 * fint[3] / 3 - fint[1] / 3) - (0.4 - 4 * lambda2) * fint[2]) / 8
    end
    for k in 5:m
        n = k - 3
        v = targ * (fint[k - 1] - fint[k - 3]) +
            (2n + 1) * (lambda2 / 2 + 1.0 / ((2n + 3) * (2n - 1))) * fint[k - 2] +
            (n - 1.0) * fint[k - 4] / (2n - 1)
        fint[k] = (2n + 3.0) * v / (n + 2)
    end
    fp[1] = 0.0
    m >= 2 && (fp[2] = fint[1] * lambda)
    for k in 3:m
        fp[k] = fp[k - 2] + (2k - 3) * fint[k - 1] * lambda
    end
    fpp[1] = 0.0
    m >= 2 && (fpp[2] = 0.0)
    for k in 3:m
        fpp[k] = fpp[k - 2] + (2k - 3) * fp[k - 1]
    end
    isign = 1
    for k in 1:m
        px1 = (k - 1) * k / 2.0
        fxx[k] = fpp[k] - px1 * (expb + isign * expa) - 2 * (b * expb - isign * a * expa) / lambda
        isign = -isign
    end
    ww = boxdim * lambda / 2
    for k in 1:m
        fint[k] *= ww
        fxx[k] = fxx[k] * 2 / boxdim
    end
    for k in 2:2:m
        fx[k] = fp[k] - d2 / lambda
    end
    for k in 1:2:m
        fx[k] = fp[k] - d1 / lambda
    end
    return nothing
end

function _chebtab_recur!(fint, fx, fxx, fp, fpp, m, targ, lambda, boxdim)
    lambda2 = lambda * lambda
    a = (-1 - targ) / lambda
    b = (1 - targ) / lambda
    erfa = _erf(a); erfb = _erf(b)
    expa = exp(-a * a); expb = exp(-b * b)
    d1 = lambda * (expb - expa)
    d2 = lambda * (expb + expa)
    fint[1] = _SQRTPIH * (erfb - erfa)
    m >= 2 && (fint[2] = -d1 / 2 + targ * fint[1])
    m >= 3 && (fint[3] = -d2 - targ * d1 + (lambda2 - 1 + 2 * targ^2) * fint[1])
    m >= 4 && (fint[4] = -d1 + 2 * targ * fint[3] - (1 - 4 * lambda2) * fint[2])
    isign = 1
    for k in 5:m
        dc = isign == 1 ? d2 : d1
        n = k - 1
        dd = (n - 1.0) / (n - 3)
        fint[k] = 2 * targ * fint[k - 1] + 2 * ((n - 1) * lambda2 + 1.0 / (n - 3)) * fint[k - 2] +
                  dd * (fint[k - 4] - 2 * targ * fint[k - 3]) - (1 - dd) * dc
        isign = -isign
    end
    fp[1] = 0.0
    m >= 2 && (fp[2] = fint[1] * lambda)
    m >= 3 && (fp[3] = 4 * fint[2] * lambda)
    for k in 4:m
        fp[k] = (fp[k - 2] / (k - 3) + 2 * fint[k - 1] * lambda) * (k - 1)
    end
    fpp[1] = 0.0
    m >= 2 && (fpp[2] = 0.0)
    m >= 3 && (fpp[3] = 4 * fint[1] * lambda)
    for k in 4:m
        fpp[k] = (fpp[k - 2] / (k - 3) + 2 * fp[k - 1]) * (k - 1)
    end
    isign = 1
    for k in 1:m
        tp1 = (k - 1)^2
        fxx[k] = fpp[k] - tp1 * (expb + isign * expa) - 2 * (b * expb - isign * a * expa) / lambda
        isign = -isign
    end
    ww = boxdim * lambda / 2
    sc = 2 / boxdim
    for k in 1:m
        fint[k] *= ww
        fxx[k] *= sc
    end
    for k in 2:2:m
        fx[k] = fp[k] - d2 / lambda
    end
    for k in 1:2:m
        fx[k] = fp[k] - d1 / lambda
    end
    return nothing
end

_tab_recur!(::LegendreBasis, args...) = _legetab_recur!(args...)
_tab_recur!(::ChebyshevBasis, args...) = _chebtab_recur!(args...)

# tab[m, j] = ∫_{source box} p_m(x) exp(-(targ_j - x)²/δ) dx (plus x/xx derivatives).
function _mk_loctab!(tab, tabx, tabxx, basis, n, delta, tnodes, tboxdim, tboxcen, scale, xq, wq, polyv)
    nquad = length(xq)
    targs = [tnodes[i] / scale + tboxcen for i in 1:n]
    bs = tboxdim * scale
    sigma = 4 * delta / bs^2
    lambda = sqrt(sigma)
    dmax = sqrt(log(1 / 1e-16) * sigma)
    fint = zeros(n); fx = zeros(n); fxx = zeros(n); fp = zeros(n); fpp = zeros(n)
    if lambda <= 0.125
        for j in 1:n
            targ = targs[j]
            dis = targ < -1 ? -1 - targ : (targ > 1 ? targ - 1 : 0.0)
            if dis > dmax
                tab[:, j] .= 0; tabx[:, j] .= 0; tabxx[:, j] .= 0
            else
                _tab_recur!(basis, fint, fx, fxx, fp, fpp, n, targ, lambda, bs)
                tab[:, j] .= fint; tabx[:, j] .= fx; tabxx[:, j] .= fxx
            end
        end
    else
        wexp = zeros(nquad, n)
        for j in 1:n, i in 1:nquad
            dx = targs[j] - xq[i]
            wexp[i, j] = abs(dx) < dmax ? exp(-dx * dx / sigma) * wq[i] : 0.0
        end
        bsh = bs / 2
        for j in 1:n
            targ = targs[j]
            dis = targ < -1 ? -1 - targ : (targ > 1 ? targ - 1 : 0.0)
            if dis > dmax
                tab[:, j] .= 0; tabx[:, j] .= 0; tabxx[:, j] .= 0
            else
                for m in 1:n
                    rsum = 0.0; rsumx = 0.0; rsumxx = 0.0
                    for i in 1:nquad
                        dx = targ - xq[i]
                        dd = -2 * dx / sigma * 2 / bs
                        pw = polyv[i, m] * wexp[i, j]
                        rsum += pw
                        rsumx += pw * dd
                        rsumxx += pw * (dd * dd - 2 / delta)
                    end
                    tab[m, j] = rsum * bsh
                    tabx[m, j] = rsumx * bsh
                    tabxx[m, j] = rsumxx * bsh
                end
            end
        end
    end
    return nothing
end

# Fortran `compute_sparse_pattern`: per column, first/last row above
# eps*max|a|/n; column n+1 holds the first/last nonzero column.
function _sparse_pattern!(indc, eps, n, a)
    for i in 1:(n + 1)
        indc[1, i] = 0
        indc[2, i] = -1
    end
    dmax = maximum(abs, a)
    dmax <= 2e-16 && return indc
    thr = eps * dmax / n
    for i in 1:n
        for j in 1:n
            if abs(a[j, i]) > thr
                indc[1, i] = j
                break
            end
        end
        for j in n:-1:1
            if abs(a[j, i]) > thr
                indc[2, i] = j
                break
            end
        end
    end
    for i in 1:n
        if indc[1, i] > 0
            indc[1, n + 1] = i
            break
        end
    end
    for i in n:-1:1
        if indc[2, i] > 0
            indc[2, n + 1] = i
            break
        end
    end
    return indc
end

function _merged_pattern!(ind, slot, eps, n, tab, tabx, tabxx, scratch)
    i0, ix, ixx = scratch
    _sparse_pattern!(i0, eps, n, @view tab[:, :, slot])
    _sparse_pattern!(ix, eps, n, @view tabx[:, :, slot])
    _sparse_pattern!(ixx, eps, n, @view tabxx[:, :, slot])
    for j in 1:(n + 1)
        i2 = max(i0[2, j], ix[2, j], ixx[2, j])
        ind[2, j, slot] = i2
        i1 = i0[1, j]
        ix[1, j] != 0 && i1 > ix[1, j] && (i1 = ix[1, j])
        ixx[1, j] != 0 && i1 > ixx[1, j] && (i1 = ixx[1, j])
        (i2 > 0 && i1 == 0) && (i1 = 1)
        ind[1, j, slot] = i1
    end
end

struct LocTabQuad
    xq::Vector{Float64}
    wq::Vector{Float64}
    polyv::Matrix{Float64}      # (nquad, n)
    tnodes::Vector{Float64}
    U::Matrix{Float64}
end

function LocTabQuad(basis::AbstractBasis, n::Int; nquad::Int = 50)
    xq, wq = nodes_and_weights(LegendreBasis(), nquad)
    polyv = zeros(nquad, n)
    p = zeros(n)
    for i in 1:nquad
        polyvals!(p, basis, xq[i], n)
        polyv[i, :] .= p
    end
    t = ortho1d(basis, n)
    return LocTabQuad(xq, wq, polyv, t.x, t.U)
end

"""All 1D local tables for Gaussian `delta` with target box size `boxdim` (Fortran `mk_loctab_all`)."""
function mk_loctab_all(eps::Float64, basis::AbstractBasis, n::Int, delta::Float64, boxdim::Float64, q::LocTabQuad)
    tab = zeros(n, n, NLOCTAB2); tabx = zeros(n, n, NLOCTAB2); tabxx = zeros(n, n, NLOCTAB2)
    ind = zeros(Int, 2, n + 1, NLOCTAB2)
    t = zeros(n, n); tx = zeros(n, n); txx = zeros(n, n)
    scratch = (zeros(Int, 2, n + 1), zeros(Int, 2, n + 1), zeros(Int, 2, n + 1))
    m0 = 1
    for i0 in m0:-1:-1, j0 in 1:(2^m0)
        k = i0 > 0 ? -2^(m0 - i0) - 2^(m0 - i0 + 1) * (j0 - 1) :
            i0 == 0 ? -2^(m0 - i0 + 1) * (j0 - 1) : -2^m0 - 2^(m0 + 1) * (j0 - 1)
        k < -NLOCTAB && break
        scale = 2.0^i0
        cen = k / 2.0^m0
        _mk_loctab!(t, tx, txx, basis, n, delta, q.tnodes, boxdim, cen, scale, q.xq, q.wq, q.polyv)
        sk = k + NLOCTAB + 1
        mul!(@view(tab[:, :, sk]), transpose(q.U), t)
        mul!(@view(tabx[:, :, sk]), transpose(q.U), tx)
        mul!(@view(tabxx[:, :, sk]), transpose(q.U), txx)
        # Mirror to the right-hand offset; done in place like the Fortran so the
        # k = 0 slot is symmetrized identically.
        sm = -k + NLOCTAB + 1
        for j in 1:n, m in 1:n
            tab[m, j, sm] = tab[n - m + 1, n - j + 1, sk]
            tabx[m, j, sm] = -tabx[n - m + 1, n - j + 1, sk]
            tabxx[m, j, sm] = tabxx[n - m + 1, n - j + 1, sk]
        end
        _merged_pattern!(ind, sm, eps, n, tab, tabx, tabxx, scratch)
        _merged_pattern!(ind, sk, eps, n, tab, tabx, tabxx, scratch)
    end
    tabT = permutedims(tab, (2, 1, 3))
    krange = zeros(Int, 4, NLOCTAB2)
    tabZ = (zeros(n, n, NLOCTAB2), zeros(n, n, NLOCTAB2), zeros(n, n, NLOCTAB2))
    for slot in 1:NLOCTAB2
        klo, khi = ind[1, n + 1, slot], ind[2, n + 1, slot]
        jlo, jhi = n + 1, 0
        for k in max(klo, 1):khi
            lo, hi = ind[1, k, slot], ind[2, k, slot]
            (lo >= 1 && hi >= lo) || continue
            jlo = min(jlo, lo); jhi = max(jhi, hi)
            for (Z, A) in zip(tabZ, (tab, tabx, tabxx)), j in lo:hi
                Z[j, k, slot] = A[j, k, slot]
            end
        end
        krange[:, slot] .= (klo, khi, jlo, jhi)
    end
    tabZT = map(Z -> permutedims(Z, (2, 1, 3)), tabZ)
    return LocTab1D(tab, tabT, tabx, tabxx, ind, tabZ, tabZT, krange)
end
