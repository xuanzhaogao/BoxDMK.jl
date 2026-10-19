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
    bdmk_periodic(f, cell; eps=1e-6, norder=16, rc_factor=0.25, nhalo=1, f_batch=nothing, timings=nothing)

Periodic Coulomb potential

    u(x) = ∫_cell Σ_n f(y) / |x - y - A n| dy        (k = 0 mode dropped)

of the cell-periodic density `f(x::AbstractVector) -> Real`. `cell` is the
3×3 matrix `A` of lattice vectors (columns), a 3-tuple `L` for the
orthorhombic cell `[0, L₁) × [0, L₂) × [0, L₃)`, or a number for a cube. A
density with nonzero mean is taken with a uniform neutralizing background
(the k = 0 mode of the whole potential is dropped).
`f_batch!(values, points)` (values `(1, npts)`, points `(3, npts)`) may be given
instead of `f`. The level-0 boxes have side `rc = d_min / ⌈1/rc_factor⌉`,
`d_min` the smallest distance between lattice planes. With `skip_negligible`
boxes whose density is too small to change the potential by `eps` relative
are left out of the real-space part (see `_source_boxes`). With `interpolate`
(cells whose level-0 cubes do not tile the cell) only the boxes meeting the
cell sample `f`, the others interpolate them at periodic images; `false` gives
the forest of the C++ builder. `timings` (length ≥ 6)
receives wall times (s): [1] forest, [2] setup and wide source, [3] local
expansions + upward pass, [4] wide part, [5] plane waves, [6] near field, and
if length ≥ 10 the plane-wave split [7] formation, [8] translation,
[9] evaluation, [10] downward pass.
"""
function bdmk_periodic(f, cell; eps::Real = 1e-6, norder::Int = 16, rc_factor::Real = 0.25, nhalo::Int = 1,
                       f_batch = nothing, skip_negligible::Bool = true, interpolate::Bool = true,
                       timings::Union{Nothing,Vector{Float64}} = nothing)
    f_batch === nothing || (f = _BatchDensity(f_batch))
    old_ftz = get_zero_subnormals()
    set_zero_subnormals(true)
    try
        f === nothing && throw(ArgumentError("provide a density f or a batch density f_batch"))
        return _bdmk_periodic(f, _cell_matrix(cell), Float64(eps), norder, Float64(rc_factor), nhalo, skip_negligible,
                              interpolate, timings)
    finally
        set_zero_subnormals(old_ftz)
    end
end

_cell_matrix(L::Real) = Matrix(Float64(L) * I, 3, 3)
_cell_matrix(L::Tuple) = Matrix(Diagonal(collect(Float64.(L))))
_cell_matrix(A::AbstractMatrix) = Matrix{Float64}(A)

function _bdmk_periodic(f, A::Matrix{Float64}, eps::Float64, n::Int, rc_factor::Float64, nhalo::Int,
                        skip_negligible::Bool, interpolate::Bool, timings)
    tm = timings === nothing ? zeros(10) : timings
    fill!(tm, 0.0)
    nhalo >= 1 || throw(ArgumentError("nhalo must be ≥ 1"))
    0 < rc_factor < 0.95 || throw(ArgumentError("need 0 < rc_factor < 0.95"))
    size(A) == (3, 3) && abs(det(A)) > 0 || throw(ArgumentError("degenerate cell"))
    _, U = _lattice_qr(A)
    Uinv = inv(UpperTriangular(U))
    dmin = minimum(1 / norm(Uinv[i, :]) for i in 1:3)
    rc = dmin / ceil(1 / rc_factor * (1 - 1e-12))
    g = PeriodicGrid(A, rc, nhalo)

    t0 = time_ns()
    ft, fvals, istarget = build_periodic_forest(f, g; norder = n, eps = eps, min_levels = 1, interpolate)
    tm[1] = (time_ns() - t0) / 1e9

    t0 = time_ns()
    r0, ws, ts = useries_sog(round(Int, -log10(eps)), rc)
    dmin > 2r0 || throw(ArgumentError("cell too small for the SOG"))
    iswide = [sqrt(t * log(1 / eps)) > rc for t in ts]
    any(iswide) || throw(ArgumentError("no wide Gaussians; increase rc_factor"))
    narrow = sort(findall(.!iswide); by = i -> -ts[i])     # δ descending
    dn, wn = ts[narrow], ws[narrow]
    tsw, wsw = ts[iswide], ws[iswide]
    taylor = taylor_coefficients(LaplaceKernel(), r0, wn, dn)
    issource = skip_negligible ? _source_boxes(ft, fvals, eps, sum(wn .* (pi .* dn) .^ 1.5)) : trues(ft.nboxes)
    s = DMKSetup(ft, LegendreBasis(), n, 1, eps, 1, r0, wn, dn, taylor; asym_safety = 1e-4, istarget, issource)
    wplan = WidePlan(g, eps, wide_orders(eps, s.porder, n, rc)..., tsw, wsw)
    # Source moments of one period: from the level-0 cubes when they tile the
    # cell (after the upward pass), else from a tree in fractional coordinates
    # refined like the forest (threshold of its level-0 test).
    Msrc = nothing
    if !g.tiled
        w3 = _tensor_weights(n)
        rint0 = sqrt(sum(b -> _box_l2sq(view(fvals, :, :, :, b), w3, rc^2 / 8), levelboxes(ft, 0)))
        Msrc = fractional_moments(f, g, ft, n, wplan.pf, eps * rc^-1.5 * rint0)
    end
    tm[2] = (time_ns() - t0) / 1e9
    try
        pot, fmean = _periodic_steps(s, g, fvals, wplan, Msrc, tm)
        # neutralizing background: remove the k = 0 mode of the real-space parts
        # (narrow Gaussians and local correction); the wide part has none
        dc = fmean * (sum(wn .* (pi .* dn) .^ 1.5) + taylor[1])
        for ibox in 1:ft.nboxes
            (ft.nchild[ibox] == 0 && istarget[ibox]) || continue
            view(pot, :, :, :, ibox) .-= dc
        end
        return PeriodicResult(ft, g, fvals, pot, istarget, n)
    finally
        release!(s.arena)
    end
end

# Boxes whose density can matter in the narrow (real-space) part. A leaf with
# max|f| ≤ τ changes the potential anywhere by at most τ ∫ K_narrow = τ S_n
# (K_narrow > 0), in all at most that much since the boxes are disjoint; τ is
# eps U / S_n with U = max over leaves of max|f| min(S_n, 2.4 h²), the
# potential scale of the strongest box (2.4 h² ≥ ∫_cube 1/r). A nonleaf box is
# a source if a leaf below it is. The wide part still uses every box.
function _source_boxes(ft::FlatTree, fvals, eps::Float64, Sn::Float64)
    nb = ft.nboxes
    bmax = zeros(nb)
    parallel_foreach(collect(1:nb), fill(nothing, _nthreads())) do b, _
        ft.nchild[b] == 0 && (bmax[b] = maximum(abs, view(fvals, :, :, :, b)))
    end
    U = maximum(b -> ft.nchild[b] == 0 ? bmax[b] * min(Sn, 2.4 * ft.boxsize[ft.level[b] + 1]^2) : 0.0, 1:nb)
    τ = eps * U / Sn
    src = falses(nb)
    for b in nb:-1:1                       # level order: children after parents
        src[b] = ft.nchild[b] == 0 ? bmax[b] > τ : any(c -> c > 0 && src[c], view(ft.child, :, b))
    end
    return src
end

_tensor_weights(n::Int) = (w = ortho1d(LegendreBasis(), n).w; vec([w[i] * w[j] * w[k] for i in 1:n, j in 1:n, k in 1:n]))

# Mean of f over the cell from the root-cell moments (u-measure).
_cell_mean(P::WidePlan, M::Array{Float64,3}) =
    sum(M[1 + P.pf * s1, 1 + P.pf * s2, 1 + P.pf * s3] for s1 in 0:(P.mt[1] - 1), s2 in 0:(P.mt[2] - 1), s3 in 0:(P.mt[3] - 1)) / prod(P.mt)

function _periodic_steps(s::DMKSetup, g::PeriodicGrid, fvals::Array{Float64,4}, wplan::WidePlan, Msrc, tm)
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
    fmean = _wide_inject!(pot, pp, pc, nodeidx, s, g, fv, wplan, Msrc)
    tm[4] = (time_ns() - t0) / 1e9

    t0 = time_ns()
    pwtm = zeros(10)
    steps456_planewave!(pot, gdum, hdum, pp, pc, nodeidx, s, fv, tpw, pwtm)
    tm[5] = (time_ns() - t0) / 1e9
    length(tm) >= 10 && (tm[7:10] .= pwtm[[5, 6, 8, 9]])

    t0 = time_ns()
    step7_local!(pot, gdum, hdum, s, fv)
    step9_proxy2pot!(pot, gdum, hdum, pp, nodeidx, s, tpw)
    tm[6] = (time_ns() - t0) / 1e9
    return reshape(pot, n, n, n, nb), fmean
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

# Root-cell moments -> modes -> u_wide Legendre coefficients on the central
# cubes, added to the proxy potentials of nonleaf boxes (spread by the
# downward pass) and evaluated directly on leaf boxes. For a tiled cell the
# moments are those of the central cubes (proxy charges of the upward pass).
# Returns the mean of f over the cell.
function _wide_inject!(pot, pp, pc, nodeidx, s::DMKSetup, g::PeriodicGrid, fv, P::WidePlan, Msrc)
    ft, p = s.ft, s.porder
    pf, pe = P.pf, P.pe
    boxes = [(ibox, c) for ibox in levelboxes(ft, 0) for c in (_central_index(g, ibox),) if c !== nothing]
    wks = make_workspaces(() -> (w1 = alloc(s.arena, max(p, s.n)^3), w2 = alloc(s.arena, max(p, s.n)^3),
                                  w3 = alloc(s.arena, p^3)))
    M = Msrc
    if M === nothing
        M = alloc(s.arena, pf * g.mgrid[1], pf * g.mgrid[2], pf * g.mgrid[3])
        rc3 = g.rc^3
        parallel_foreach(boxes, wks) do (ibox, (i, j, k)), w
            q, sc = if ft.nchild[ibox] == 0
                out = reshape(view(w.w3, 1:p^3), p, p, p)
                tp3!(out, view(fv, :, :, :, 1, ibox), s.den2pc, s.den2pcT, s.den2pcT, w.w1, w.w2, 1 / 8, false)
                out, 1.0
            else
                reshape(view(pc, :, 1, nodeidx[ibox]), p, p, p), 1 / rc3
            end
            M[pf * i + 1:pf * (i + 1), pf * j + 1:pf * (j + 1), pf * k + 1:pf * (k + 1)] .= sc .* view(q, 1:pf, 1:pf, 1:pf)
        end
    end
    G = wide_modes(P, M, s.arena)
    C = wide_coefficients(P, G, s.arena)
    parallel_foreach([b for b in boxes if s.istarget[b[1]]], wks) do (ibox, (i, j, k)), w
        cb = reshape(view(w.w3, 1:p^3), p, p, p)
        fill!(cb, 0.0)
        cb[1:pe, 1:pe, 1:pe] .= view(C, pe * i + 1:pe * (i + 1), pe * j + 1:pe * (j + 1), pe * k + 1:pe * (k + 1))
        if ft.nchild[ibox] == 0
            tp3!(view(pot, :, :, :, 1, ibox), cb, s.poteval, s.potevalT, s.potevalT, w.w1, w.w2, 1.0, true)
        else
            vec(view(pp, :, 1, 1, nodeidx[ibox])) .+= vec(cb)
        end
    end
    return _cell_mean(P, M)
end

"""
    evaluate_periodic(res::PeriodicResult, x::AbstractVector) -> Float64

Potential at any point `x` (wrapped into the cell), by interpolation on the
leaf containing it.
"""
function evaluate_periodic(res::PeriodicResult, x::AbstractVector)
    g, ft, n = res.grid, res.ft, res.norder
    t = g.Uinv * (transpose(g.Q) * x)            # fractional coordinates
    t .-= floor.(t)
    y = g.U * t                                    # in-cell point, rotated frame
    ibox = _locate_leaf(ft, g, y)
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
