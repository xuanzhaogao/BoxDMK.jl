# Periodic box-DMK for the Laplace kernel 1/r in 3D (port of the C++
# `bdmk_laplace3d_pot_periodic`, orthorhombic cells). The kernel is the
# u-series sum of Gaussians, split at the box size rc of the level-0 grid:
# narrow Gaussians (cutoff ≤ rc) run through the free-space box machinery on
# the image-halo forest, wide ones through the reciprocal-space sum of
# wide.jl, injected into the level-0 proxy potentials before the downward
# pass. The density is assumed charge-neutral (the k = 0 mode is dropped).

"""
    useries_sog(ndigits, bsize) -> (r0, weights, deltas)

u-series sum of Gaussians `1/r ≈ Σ w exp(-r²/δ)` (Liang–Xu–Zhou) with relative
error `10^-ndigits/2` on `[2⁻¹⁶, 10^ndigits] * bsize`.
"""
function useries_sog(nd::Int, bsize::Float64)
    nd > 0 || throw(ArgumentError("ndigits must be positive"))
    epsv = 10.0^(-nd)
    logb = pi^2 / (2 * (nd * log(10) + log(4 * sqrt(2))))
    b = exp(logb)
    cw = sqrt(2 / pi) * logb
    relerr(r, lo, hi) = abs(1 - r * sum(cw / b^l * exp(-r^2 / (2 * b^(2l))) for l in lo:hi))
    mmin = floor(Int, -33 * log(2) / (2 * logb))
    a = 2 * nd * log(10)
    mmax = floor(Int, (a + 0.48) * (a + 3.466) / pi^2)
    rlo = 2.0^-16
    rhi = 10.0^nd
    for _ in 1:128
        relerr(rlo, mmin, mmax) > epsv / 2 || break
        mmin -= 1
    end
    for _ in 1:128
        relerr(rhi, mmin, mmax) > epsv / 2 || break
        mmax += 1
    end
    ts = [2 * b^(2l) * bsize^2 for l in mmin:mmax]
    ws = [cw / b^l / bsize for l in mmin:mmax]
    return rlo * bsize, ws, ts
end

"""
    PeriodicResult

Output of [`bdmk_periodic`](@ref): the forest `ft`, its level-0 `grid`, the
density samples `fvals` and potential `pot` (both `(n, n, n, nboxes)` on each
box's Legendre grid), and `istarget` (false on image-halo boxes, whose
potential is not computed). Evaluate anywhere with
[`evaluate_periodic`](@ref).
"""
struct PeriodicResult
    ft::FlatTree
    grid::PeriodicGrid
    fvals::Array{Float64,4}
    pot::Array{Float64,4}
    istarget::BitVector
    norder::Int
end

"""
    bdmk_periodic(f, L; eps=1e-6, norder=16, rc_factor=0.25, nhalo=1, timings=nothing)

Periodic Coulomb potential `u(x) = ∫_cell Σ_n f(y) / |x - y - n∘L| dy` (k = 0
mode dropped) of the charge-neutral, `L`-periodic density
`f(x::AbstractVector) -> Real` on the orthorhombic cell `[0, L₁) × [0, L₂) × [0, L₃)`
(`L` a number for a cube). The level-0 boxes have side
`rc ≈ rc_factor * minimum(L)`, rounded so that every side is a multiple of it.
`timings` (length ≥ 6) receives wall times (s): [1] forest, [2] setup,
[3] local expansions + upward pass, [4] wide part, [5] plane waves,
[6] near field.
"""
function bdmk_periodic(f, L; eps::Real = 1e-6, norder::Int = 16, rc_factor::Real = 0.25, nhalo::Int = 1,
                       timings::Union{Nothing,Vector{Float64}} = nothing)
    old_ftz = get_zero_subnormals()
    set_zero_subnormals(true)
    try
        return _bdmk_periodic(f, L isa Real ? (Float64(L), Float64(L), Float64(L)) : Float64.(Tuple(L)),
                              Float64(eps), norder, Float64(rc_factor), nhalo, timings)
    finally
        set_zero_subnormals(old_ftz)
    end
end

function _bdmk_periodic(f, L::NTuple{3,Float64}, eps::Float64, n::Int, rc_factor::Float64, nhalo::Int, timings)
    tm = timings === nothing ? zeros(6) : timings
    fill!(tm, 0.0)
    nhalo >= 1 || throw(ArgumentError("nhalo must be ≥ 1"))
    dmin = minimum(L)
    rc = dmin / ceil(dmin / (rc_factor * dmin) * (1 - 1e-12))
    0 < rc < 0.95 * dmin || throw(ArgumentError("need 0 < rc < 0.95 min(L)"))
    g = PeriodicGrid(L, rc, nhalo)

    t0 = time_ns()
    ft, fvals, istarget = build_periodic_forest(f, g; norder = n, eps = eps, min_levels = 1)
    tm[1] = (time_ns() - t0) / 1e9

    t0 = time_ns()
    r0, ws, ts = useries_sog(round(Int, -log10(eps)), rc)
    dmin > 2r0 || throw(ArgumentError("cell too small for the SOG"))
    iswide = [sqrt(t * log(1 / eps)) > rc for t in ts]
    narrow = sort(findall(.!iswide); by = i -> -ts[i])     # δ descending
    dn, wn = ts[narrow], ws[narrow]
    tsw, wsw = ts[iswide], ws[iswide]
    taylor = taylor_coefficients(LaplaceKernel(), r0, wn, dn)
    s = DMKSetup(ft, LegendreBasis(), n, 1, eps, 1, r0, wn, dn, taylor; asym_safety = 1e-4, istarget = istarget)
    wplan = isempty(tsw) ? nothing : WidePlan(g, eps, s.porder, tsw, wsw)
    tm[2] = (time_ns() - t0) / 1e9
    try
        pot = _periodic_steps(s, g, fvals, wplan, tm)
        return PeriodicResult(ft, g, fvals, pot, istarget, n)
    finally
        release!(s.arena)
    end
end

function _periodic_steps(s::DMKSetup, g::PeriodicGrid, fvals::Array{Float64,4}, wplan, tm)
    ft, n, p = s.ft, s.n, s.porder
    nb = ft.nboxes
    fv = reshape(fvals, n, n, n, 1, nb)
    pot = pzeros(n, n, n, 1, nb)
    gdum = alloc(s.arena, n, n, n, 1, 3, 0)
    hdum = alloc(s.arena, n, n, n, 1, 6, 0)

    t0 = time_ns()
    step2_local_expansions!(pot, gdum, hdum, s, fv)
    nodeidx, nnode = node_index(s)
    pc = alloc(s.arena, p^3, 1, nnode)
    tpw = make_workspaces(() -> TPWork(max(p, n)^3, s.arena))
    step3_upward!(pc, nodeidx, s, fv, tpw)
    pp = pfill!(alloc(s.arena, p^3, 1, 1, nnode))
    tm[3] = (time_ns() - t0) / 1e9

    t0 = time_ns()
    wplan === nothing || _wide_inject!(pot, pp, pc, nodeidx, s, g, fv, wplan, tpw)
    tm[4] = (time_ns() - t0) / 1e9

    t0 = time_ns()
    pwtm = zeros(10)
    steps456_planewave!(pot, gdum, hdum, pp, pc, nodeidx, s, fv, tpw, pwtm)
    tm[5] = (time_ns() - t0) / 1e9

    t0 = time_ns()
    step7_local!(pot, gdum, hdum, s, fv)
    step9_proxy2pot!(pot, gdum, hdum, pp, nodeidx, s, tpw)
    tm[6] = (time_ns() - t0) / 1e9
    return reshape(pot, n, n, n, nb)
end

# Central level-0 box (0-based central grid indices) of each level-0 box, or nothing.
function _central_index(g::PeriodicGrid, ibox::Int)
    m1, m2 = g.mle[1], g.mle[2]
    i = (ibox - 1) % m1 - g.nhalo
    j = ((ibox - 1) ÷ m1) % m2 - g.nhalo
    k = (ibox - 1) ÷ (m1 * m2) - g.nhalo
    (0 <= i < g.mgrid[1] && 0 <= j < g.mgrid[2] && 0 <= k < g.mgrid[3]) || return nothing
    return (i, j, k)
end

# Moments of the central boxes -> u_wide Legendre coefficients, added to the
# proxy potentials of nonleaf boxes (spread by the downward pass) and
# evaluated directly on leaf boxes.
function _wide_inject!(pot, pp, pc, nodeidx, s::DMKSetup, g::PeriodicGrid, fv, P::WidePlan, tpw)
    ft, p = s.ft, s.porder
    boxes = [(ibox, c) for ibox in levelboxes(ft, 0) for c in (_central_index(g, ibox),) if c !== nothing]
    M = zeros(p * g.mgrid[1], p * g.mgrid[2], p * g.mgrid[3])
    sc3 = (g.rc / 2)^3
    parallel_foreach(boxes, tpw) do (ibox, (i, j, k)), w
        q = if ft.nchild[ibox] == 0
            out = zeros(p, p, p)
            tp3!(out, view(fv, :, :, :, 1, ibox), s.den2pc, s.den2pcT, s.den2pcT, w.w1, w.w2, sc3, false)
            out
        else
            reshape(view(pc, :, 1, nodeidx[ibox]), p, p, p)
        end
        M[p * i + 1:p * (i + 1), p * j + 1:p * (j + 1), p * k + 1:p * (k + 1)] .= q
    end
    C = wide_coefficients(P, M)
    parallel_foreach(boxes, tpw) do (ibox, (i, j, k)), w
        cb = C[p * i + 1:p * (i + 1), p * j + 1:p * (j + 1), p * k + 1:p * (k + 1)]
        if ft.nchild[ibox] == 0
            tp3!(view(pot, :, :, :, 1, ibox), cb, s.poteval, s.potevalT, s.potevalT, w.w1, w.w2, 1.0, true)
        else
            vec(view(pp, :, 1, 1, nodeidx[ibox])) .+= vec(cb)
        end
    end
    return nothing
end

"""
    evaluate_periodic(res::PeriodicResult, x::AbstractVector) -> Float64

Potential at any point `x` (wrapped into the cell), by interpolation on the
leaf containing it.
"""
function evaluate_periodic(res::PeriodicResult, x::AbstractVector)
    g, ft, n = res.grid, res.ft, res.norder
    y = ntuple(d -> mod(x[d], g.L[d]), 3)
    i, j, k = ntuple(d -> min(floor(Int, y[d] / g.rc), g.mgrid[d] - 1) + g.nhalo, 3)
    ibox = 1 + i + g.mle[1] * (j + g.mle[2] * k)
    while ft.nchild[ibox] > 0
        slot = 1
        y[1] > ft.centers[1, ibox] && (slot += 1)
        y[2] > ft.centers[2, ibox] && (slot += 2)
        y[3] > ft.centers[3, ibox] && (slot += 4)
        ibox = ft.child[slot, ibox]
    end
    leg = ortho1d(LegendreBasis(), n)
    sc = 2.0 / ft.boxsize[ft.level[ibox] + 1]
    pv = zeros(n)
    l = ntuple(3) do d
        _interp_weights!(zeros(n), pv, LegendreBasis(), leg.U, (y[d] - ft.centers[d, ibox]) * sc, n)
    end
    v = view(res.pot, :, :, :, ibox)
    acc = 0.0
    for c in 1:n, b in 1:n, a in 1:n
        acc += l[1][a] * l[2][b] * l[3][c] * v[a, b, c]
    end
    return acc
end
