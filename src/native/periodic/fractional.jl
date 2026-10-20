# Source side of the wide term when the level-0 cubes do not tile one period
# (triclinic cells, sides not multiples of rc): an adaptive tree over exactly
# one period in the coordinates u = mt ∘ t (t fractional), whose root cells
# are unit cubes mapped to parallelepipeds with edges |a_d| / mt[d] ≤ rc. The
# density is sampled on its own Legendre grids; cells refine by the tail test
# of the main forest and, so that features the main forest resolves cannot
# hide between sample points, while they are more than twice the size of the
# finest main-forest leaf around them. An upward pass yields the Legendre
# moments of each root cell (u-measure) up to degree pf - 1.

# Leaf of the forest `ft` containing the rotated-frame point x (inside the
# central region).
function _locate_leaf(ft::FlatTree, g::PeriodicGrid, x)
    i, j, k = ntuple(d -> _gridcell(g, x[d], d), 3)
    ibox = 1 + i + g.mle[1] * (j + g.mle[2] * k)
    while ft.nchild[ibox] > 0
        slot = 1
        x[1] > ft.centers[1, ibox] && (slot += 1)
        x[2] > ft.centers[2, ibox] && (slot += 2)
        x[3] > ft.centers[3, ibox] && (slot += 4)
        ibox = ft.child[slot, ibox]
    end
    return ibox
end

# The density of the main forest (Legendre polynomials of its central
# leaves) as the source of the fractional tree, when f is sampled once.
struct _ForestSource
    ft::FlatTree
    g::PeriodicGrid
    coef::Vector{Vector{Float64}}   # Legendre coefficients of the central leaves
end

function _ForestSource(ft::FlatTree, g::PeriodicGrid, fvals::Array{Float64,4}, n::Int, ar::Arena)
    leaves = [b for b in 1:ft.nboxes if ft.nchild[b] == 0 && !_inhalo(g, view(ft.centers, :, b))]
    cf = _leaf_coefs(ar, n, [view(fvals, :, :, :, b) for b in leaves])
    coef = Vector{Vector{Float64}}(undef, ft.nboxes)
    for (i, b) in enumerate(leaves)
        coef[b] = cf[i]
    end
    return _ForestSource(ft, g, coef)
end

# Values of the fractional boxes `bs` from the forest. The nodes of a box are
# y = Ur u on its tensor grid u, with Ur upper triangular: y₃ depends on the
# z index k only, y₂ on (j, k). On each forest leaf the polynomial is thus
# contracted along z per k, along y per (j, k) and along x per node (≈ 6n⁴
# flops for a box in one leaf instead of 2n⁶ point by point).
function _sample!(F::_Forest, fs::_ForestSource, bs, x::Vector{Float64})
    n = F.n; np = n^3
    length(F.fvals) < _nbox(F) && resize!(F.fvals, _nbox(F))
    for b in bs
        F.fvals[b] = alloc(F.ar, np)
    end
    Ur = Matrix(transpose(fs.g.Q)) * F.R                  # u -> rotated frame (upper triangular)
    ws = [(gid = zeros(Int, np), grp = Int[], y1 = zeros(np), y2 = zeros(n, n), y3 = zeros(n),
           P = zeros(n, n), D = alloc(F.ar, n * n, n), E = zeros(n, n), ks = zeros(Int, n), js = zeros(Int, n),
           pv = zeros(n)) for _ in 1:_nthreads()]
    parallel_foreach(collect(bs), ws) do b, w
        c = F.centers[b]
        h = F.boxsize[F.level[b] + 1] / 2
        _forest_values!(F.fvals[b], c, h, x, Ur, fs, n, w)
    end
    return F
end

function _forest_values!(v, c, h::Float64, x::Vector{Float64}, Ur::Matrix{Float64}, fs::_ForestSource, n::Int, w)
    ft, g = fs.ft, fs.g
    y1, y2, y3, gid, grp = w.y1, w.y2, w.y3, w.gid, w.grp
    @inbounds for k in 1:n
        u3 = c[3] + h * x[k]
        y3[k] = Ur[3, 3] * u3
        for j in 1:n
            u2 = c[2] + h * x[j]
            y2[j, k] = Ur[2, 2] * u2 + Ur[2, 3] * u3
        end
    end
    # leaf of each node (the previous node's leaf when it strictly holds it)
    empty!(grp)
    Lc = 0; lo = (0.0, 0.0, 0.0); hi = (0.0, 0.0, 0.0); gc = 0
    q = 0
    @inbounds for k in 1:n, j in 1:n, i in 1:n
        q += 1
        y = (Ur[1, 1] * (c[1] + h * x[i]) + Ur[1, 2] * (c[2] + h * x[j]) + Ur[1, 3] * (c[3] + h * x[k]), y2[j, k], y3[k])
        y1[q] = y[1]
        if !(Lc != 0 && lo[1] < y[1] < hi[1] && lo[2] < y[2] < hi[2] && lo[3] < y[3] < hi[3])
            Lc = _locate_leaf(ft, g, y)
            hl = ft.boxsize[ft.level[Lc] + 1] / 2
            lo = (ft.centers[1, Lc] - hl, ft.centers[2, Lc] - hl, ft.centers[3, Lc] - hl)
            hi = (ft.centers[1, Lc] + hl, ft.centers[2, Lc] + hl, ft.centers[3, Lc] + hl)
            gc = 0
            for e in eachindex(grp)
                grp[e] == Lc && (gc = e; break)
            end
            gc == 0 && (push!(grp, Lc); gc = length(grp))
        end
        gid[q] = gc
    end
    P, D, E, ks, js, pv = w.P, w.D, w.E, w.ks, w.js, w.pv
    nn = n * n
    for (gi, L) in enumerate(grp)
        hl = ft.boxsize[ft.level[L] + 1] / 2
        cl = (ft.centers[1, L], ft.centers[2, L], ft.centers[3, L])
        C = fs.coef[L]
        # z: D[:, r] = C (n² × n) P₃(ξ₃(k_r)) for the k holding nodes of this leaf
        nk = 0
        @inbounds for k in 1:n
            any(q -> gid[q] == gi, (1 + nn * (k - 1)):(nn * k)) || continue
            ks[nk += 1] = k
            legepols!(pv, (y3[k] - cl[3]) / hl, n)
            for a in 1:n
                P[a, nk] = pv[a]
            end
        end
        GC.@preserve C D P gemm!(pointer(D), nn, pointer(C), nn, pointer(P), n, nn, nk, n, 1.0, false)
        for r in 1:nk
            k = ks[r]
            # y: E[:, s] = D_k (n × n) P₂(ξ₂(j_s, k))
            nj = 0
            @inbounds for j in 1:n
                o = n * (j - 1) + nn * (k - 1)
                any(i -> gid[o + i] == gi, 1:n) || continue
                js[nj += 1] = j
                legepols!(pv, (y2[j, k] - cl[2]) / hl, n)
                for a in 1:n
                    P[a, nj] = pv[a]
                end
            end
            GC.@preserve D E P gemm!(pointer(E), n, pointer(D) + 8 * nn * (r - 1), n, pointer(P), n, n, nj, n, 1.0, false)
            # x: per node
            @inbounds for s in 1:nj
                j = js[s]
                o = n * (j - 1) + nn * (k - 1)
                for i in 1:n
                    gid[o + i] == gi || continue
                    legepols!(pv, (y1[o + i] - cl[1]) / hl, n)
                    t = 0.0
                    @simd for a in 1:n
                        t = muladd(pv[a], E[a, s], t)
                    end
                    v[o + i] = t
                end
            end
        end
    end
    return v
end

"""
    fractional_moments(f, g, ft, norder, pf, thr) -> M

Moments `M (pf mt₁, pf mt₂, pf mt₃)` (index `l + pf s` per axis, u-measure) of
the root cells of one period, `mt = source_cells(g)`; `ft` is the main forest
and `thr` its level-0 refinement threshold.
"""
function fractional_moments(f, g::PeriodicGrid, ft::FlatTree, n::Int, pf::Int, thr::Float64)
    mt = source_cells(g)
    R = g.Q * g.U * Diagonal(1 ./ collect(mt))         # u -> original frame
    Ur = g.U * Diagonal(1 ./ collect(mt))               # u -> rotated frame
    F = _Forest(n, 1.0, Arena(), R)
    try
        return _fractional_moments!(F, f, g, ft, Ur, mt, pf, thr)
    finally
        release!(F.ar)
    end
end

# Root cells of one period refined by the tail test (threshold thr) and while
# larger than `ratio` times the finest main-forest leaf around them.
function _fractional_tree!(F::_Forest, f, g::PeriodicGrid, ft::FlatTree, Ur::Matrix{Float64}, mt, thr::Float64,
                           ratio::Float64 = 2.0)
    n = F.n
    leg = ortho1d(LegendreBasis(), n)
    x = leg.x
    tail = BitArray([i^2 + j^2 + k^2 >= n^2 for i in 1:n, j in 1:n, k in 1:n])
    for k in 0:(mt[3] - 1), j in 0:(mt[2] - 1), i in 0:(mt[1] - 1)
        _addbox!(F, (i + 0.5, j + 0.5, k + 0.5), 0, 0)
    end
    nroot = _nbox(F)
    _sample!(F, f, 1:nroot, x)
    edge = maximum(norm(view(Ur, :, d)) for d in 1:3)  # physical size of a unit u-cube (max edge)
    level_boxes = collect(1:nroot)
    wss = [(zeros(n^3), zeros(n^3), zeros(n^3), zeros(3)) for _ in 1:_nthreads()]
    Ut = Matrix(transpose(leg.U))
    while true
        flag = zeros(Bool, length(level_boxes))
        lb = level_boxes                     # not captured as a reassigned (boxed) variable
        parallel_foreach(collect(eachindex(lb)), wss) do i, (c3, w1, w2, y)
            b = lb[i]
            h = F.boxsize[F.level[b] + 1]
            if _tail_error(_fbox(F, b), leg.U, Ut, tail, c3, w1, w2) > thr
                flag[i] = true
                return
            end
            # finest main-forest leaf at the 27 points of the cell (slightly inside)
            c = F.centers[b]
            hb = edge * h
            for oz in -1:1, oy in -1:1, ox in -1:1
                u1 = c[1] + 0.49 * h * ox; u2 = c[2] + 0.49 * h * oy; u3 = c[3] + 0.49 * h * oz
                for d in 1:3
                    y[d] = Ur[d, 1] * u1 + Ur[d, 2] * u2 + Ur[d, 3] * u3
                end
                leaf = _locate_leaf(ft, g, y)
                if ft.boxsize[ft.level[leaf] + 1] < hb / ratio
                    flag[i] = true
                    return
                end
            end
        end
        any(flag) || break
        level_boxes = _refine!(F, f, level_boxes[flag], x)
    end
    return F
end

function _fractional_moments!(F::_Forest, f, g::PeriodicGrid, ft::FlatTree, Ur::Matrix{Float64}, mt, pf::Int, thr::Float64)
    _fractional_tree!(F, f, g, ft, Ur, mt, thr)
    n = F.n
    leg = ortho1d(LegendreBasis(), n)
    nroot = prod(mt)

    # Upward pass. Leaf moments from values: ∫ ℓ_i P_l = 2/(2l+1) U[l, i]
    # (l < n, zero above), times the cell volume / 8.
    D = zeros(pf, n)
    for l in 0:(min(pf, n) - 1), i in 1:n
        D[l + 1, i] = 2 / (2l + 1) * leg.U[l + 1, i]
    end
    DT = Matrix(transpose(D))
    _, c2p = coefs_translation_matrices(LegendreBasis(), pf)
    c2pT = map(m -> Matrix(transpose(m)), c2p)
    c2pf = map(m -> m * D, c2p)
    c2pfT = map(m -> Matrix(transpose(m)), c2pf)
    nb = _nbox(F)
    lv = _levelorder(F)
    mom = Vector{Vector{Float64}}(undef, nb)
    m = max(n, pf)
    tpw = [(alloc(F.ar, m^3), alloc(F.ar, m^3)) for _ in 1:_nthreads()]
    for L in (length(lv) - 1):-1:0
        boxes = [b for b in lv[L + 1] if !_isleaf(F, b) || F.parent[b] == 0]
        parallel_foreach(boxes, tpw) do b, (Z, Y)
            out = alloc(F.ar, pf^3)
            if _isleaf(F, b)
                tp3!(out, _fbox(F, b), D, DT, DT, Z, Y, (F.boxsize[L + 1] / 2)^3, false)
            else
                ch = F.child[b]
                children2parent!(out, s -> _isleaf(F, ch[s]) ? nothing : mom[ch[s]], c2p, c2pT, c2pT, pf, pf, Z, Y, false)
                children2parent!(out, s -> _isleaf(F, ch[s]) ? _fbox(F, ch[s]) : nothing, c2pf, c2pfT, c2pfT, n, pf, Z, Y,
                                 true, (F.boxsize[L + 2] / 2)^3)
            end
            mom[b] = out
        end
    end
    M = zeros(pf * mt[1], pf * mt[2], pf * mt[3])
    for b in 1:nroot
        i = b - 1
        s1 = i % mt[1]; s2 = (i ÷ mt[1]) % mt[2]; s3 = i ÷ (mt[1] * mt[2])
        M[pf * s1 + 1:pf * (s1 + 1), pf * s2 + 1:pf * (s2 + 1), pf * s3 + 1:pf * (s3 + 1)] .= reshape(mom[b], pf, pf, pf)
    end
    return M
end

"""
    PeriodQuadrature

Quadrature over exactly one period of the cell: Gauss–Legendre nodes `x`
(3 × M, caller's frame) and weights `w` on the leaves of a tree in fractional
coordinates (parallelepiped cells tiling the period), refined like the wide
source tree. Made by [`period_quadrature`](@ref); field values on the plan's
forest are interpolated to its nodes by [`period_values!`](@ref).
"""
struct PeriodQuadrature
    x::Matrix{Float64}
    w::Vector{Float64}
    centers::Vector{NTuple{3,Float64}}   # leaves, u coordinates
    halfsize::Vector{Float64}            # leaves, u half sides
    Ur::Matrix{Float64}                  # u -> rotated frame
    n::Int
end

"""
    period_quadrature(plan; ratio=1.0) -> PeriodQuadrature

One-period quadrature for the plan's current density (`plan.fvals`): the
fractional tree is refined by the tail test of the forest and while a cell is
larger than `ratio` times the finest forest leaf around it, so that products
of fields resolved by the forest are integrated to its accuracy.
"""
function period_quadrature(plan; ratio::Real = 1.0)
    g, ft, n = plan.g, plan.ft, plan.n
    mt = source_cells(g)
    Ur = g.U * Diagonal(1 ./ collect(mt))
    F = _Forest(n, 1.0, Arena(), g.Q * Ur)
    ar = Arena()
    try
        w3 = _tensor_weights(n)
        rint0 = sqrt(sum(b -> _box_l2sq(view(plan.fvals, :, :, :, b), w3, g.rc^2 / 8), levelboxes(ft, 0)))
        _fractional_tree!(F, _ForestSource(ft, g, plan.fvals, n, ar), g, ft, Ur, mt, plan.eps * g.rc^-1.5 * rint0,
                          Float64(ratio))
        leaves = [b for b in 1:_nbox(F) if _isleaf(F, b)]
        leg = ortho1d(LegendreBasis(), n)
        x, wl = leg.x, leg.w
        np = n^3
        X = zeros(3, np * length(leaves)); W = zeros(np * length(leaves))
        R = g.Q * Ur
        J = abs(det(Ur))
        cs = [F.centers[b] for b in leaves]
        hs = [F.boxsize[F.level[b] + 1] / 2 for b in leaves]
        parallel_foreach(collect(eachindex(leaves)), fill(nothing, _nthreads())) do i, _
            c, h = cs[i], hs[i]
            q = np * (i - 1)
            @inbounds for k in 1:n, j in 1:n, l in 1:n
                q += 1
                u = (c[1] + h * x[l], c[2] + h * x[j], c[3] + h * x[k])
                for d in 1:3
                    X[d, q] = R[d, 1] * u[1] + R[d, 2] * u[2] + R[d, 3] * u[3]
                end
                X[3, q] += g.zorg
                W[q] = J * h^3 * wl[l] * wl[j] * wl[k]
            end
        end
        return PeriodQuadrature(X, W, cs, hs, Ur, n)
    finally
        release!(ar)
        release!(F.ar)
    end
end

"""
    period_values!(out, Q::PeriodQuadrature, res::PeriodicResult; field=res.pot) -> out

Values at the nodes of `Q` of a field on the forest of `res` (the potential by
default; `res.fvals` for the density), from the polynomials of its central leaves.
"""
function period_values!(out::AbstractVector{Float64}, Q::PeriodQuadrature, res; field = res.pot)
    ft, g, n = res.ft, res.grid, res.norder
    n == Q.n || throw(ArgumentError("quadrature and forest orders differ"))
    length(out) == size(Q.x, 2) || throw(ArgumentError("need one output per node"))
    np = n^3
    x = ortho1d(LegendreBasis(), n).x
    ar = Arena()
    try
        fs = _ForestSource(ft, g, field, n, ar)
        ws = [(gid = zeros(Int, np), grp = Int[], y1 = zeros(np), y2 = zeros(n, n), y3 = zeros(n),
               P = zeros(n, n), D = alloc(ar, n * n, n), E = zeros(n, n), ks = zeros(Int, n), js = zeros(Int, n),
               pv = zeros(n), v = zeros(np)) for _ in 1:_nthreads()]
        parallel_foreach(collect(eachindex(Q.centers)), ws) do i, w
            _forest_values!(w.v, Q.centers[i], Q.halfsize[i], x, Q.Ur, fs, n, w)
            copyto!(out, 1 + np * (i - 1), w.v, 1, np)
        end
    finally
        release!(ar)
    end
    return out
end
