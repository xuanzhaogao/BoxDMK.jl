# Rect-grid forest of the periodic solver (C++ `bdmk_build_voltree_rectgrid`):
# a grid of cubic boxes of side rc covering the cell plus a one-box image halo
# on every face. Each grid box roots an adaptive octree (Legendre tail test of
# the free-space builder), and the whole forest is then 2:1 balanced. Halo
# boxes sample the periodic density outside the cell, which is how the images
# enter the (free-space) narrow cascade.

"""
    PeriodicGrid

Lattice and level-0 geometry of the periodic forest. The cell
`{A t : t ∈ [0,1)³}` (lattice vectors = columns of `A`) is handled in the
rotated frame `x_rot = Qᵀ x` with `A = Q U`, `U` upper triangular with a
positive diagonal (C++ `BdmkLattice`). The level-0 grid has `mgrid[d]`
cubes of side `rc` covering the bounding box of the cell from `xgrid0`,
plus `nhalo` image layers: `mle = mgrid .+ 2nhalo` boxes per axis from
`xelo = xgrid0 .- nhalo * rc`.
"""
struct PeriodicGrid
    A::Matrix{Float64}
    Q::Matrix{Float64}
    U::Matrix{Float64}
    Uinv::Matrix{Float64}
    V::Float64
    dmin::Float64               # smallest distance between lattice planes
    rc::Float64
    mgrid::NTuple{3,Int}
    xgrid0::NTuple{3,Float64}
    nhalo::Int
    mle::NTuple{3,Int}
    xelo::NTuple{3,Float64}
    diagonal::Bool              # orthorhombic (U diagonal)
    tiled::Bool                 # orthorhombic with sides multiple of rc: the central cubes tile the cell
end

# A = Q U with U upper triangular, diag(U) > 0.
function _lattice_qr(A::AbstractMatrix)
    F = qr(Matrix{Float64}(A))
    D = Diagonal(sign.(diag(F.R)))
    Q = Matrix(F.Q) * D
    U = D * F.R
    for j in 1:3, i in (j + 1):3
        U[i, j] = 0.0
    end
    return Q, Matrix(U)
end

function PeriodicGrid(A::AbstractMatrix, rc::Float64, nhalo::Int)
    size(A) == (3, 3) || throw(ArgumentError("the cell must be a 3×3 matrix of lattice vectors (columns)"))
    abs(det(A)) > 0 || throw(ArgumentError("degenerate cell"))
    Q, U = _lattice_qr(A)
    Uinv = inv(UpperTriangular(U)) |> Matrix
    dmin = minimum(1 / norm(Uinv[i, :]) for i in 1:3)
    # bounding box of the cell along each rotated axis (C++ compute_grid)
    xg = zeros(3); mg = zeros(Int, 3)
    for i in 3:-1:1
        lo = 0.0; hi = U[i, i]
        for j in (i + 1):3
            lo += min(0.0, U[i, j]); hi += max(0.0, U[i, j])
        end
        mg[i] = max(1, ceil(Int, (hi - lo) / rc))
        xg[i] = lo
    end
    scale = maximum(abs, U)
    diagonal = all(abs(U[i, j]) <= 1e-12 * scale for j in 1:3 for i in 1:3 if i != j)
    tiled = diagonal && all(abs(round(U[d, d] / rc) * rc - U[d, d]) <= 1e-9 * U[d, d] for d in 1:3)
    tiled && (mg .= [round(Int, U[d, d] / rc) for d in 1:3])
    mgrid = Tuple(mg); xgrid0 = Tuple(xg)
    return PeriodicGrid(Matrix{Float64}(A), Q, U, Uinv, abs(det(U)), dmin, rc, mgrid, xgrid0, nhalo,
                        mgrid .+ 2nhalo, xgrid0 .- nhalo * rc, diagonal, tiled)
end

PeriodicGrid(L::NTuple{3,Float64}, rc::Float64, nhalo::Int) = PeriodicGrid(Matrix(Diagonal(collect(L))), rc, nhalo)

# Does the cube (center c, half side h, rotated frame) meet the cell? Slab test
# on the fractional coordinates (conservative).
function _meets_cell(g::PeriodicGrid, c, h::Float64)
    for d in 1:3
        t = g.Uinv[d, 1] * c[1] + g.Uinv[d, 2] * c[2] + g.Uinv[d, 3] * c[3]
        r = h * (abs(g.Uinv[d, 1]) + abs(g.Uinv[d, 2]) + abs(g.Uinv[d, 3]))
        (t + r <= 1e-12 || t - r >= 1 - 1e-12) && return false
    end
    return true
end

# Level-0 grid cell (0-based, clamped) containing x along axis d.
@inline _gridcell(g::PeriodicGrid, x::Float64, d::Int) = clamp(floor(Int, (x - g.xelo[d]) / g.rc), 0, g.mle[d] - 1)

# A box is a halo box when its level-0 cell lies in the image layer.
function _inhalo(g::PeriodicGrid, c)
    for d in 1:3
        k = _gridcell(g, c[d], d)
        (k < g.nhalo || k >= g.mle[d] - g.nhalo) && return true
    end
    return false
end

mutable struct _Forest
    n::Int
    centers::Vector{NTuple{3,Float64}}
    level::Vector{Int}
    parent::Vector{Int}
    child::Vector{NTuple{8,Int}}    # slot = 1 + sx + 2sy + 4sz
    fvals::Vector{Vector{Float64}}  # n³ values per box, x fastest (arena memory)
    boxsize::Vector{Float64}        # boxsize[ilev + 1]
    ar::Arena
    R::Matrix{Float64}              # the density is sampled at R * (box point)
end

_nbox(F::_Forest) = length(F.level)
_isleaf(F::_Forest, b::Int) = F.child[b][1] == 0
_fbox(F::_Forest, b::Int) = F.fvals[b]

function _addbox!(F::_Forest, c::NTuple{3,Float64}, lev::Int, par::Int)
    push!(F.centers, c); push!(F.level, lev); push!(F.parent, par)
    push!(F.child, ntuple(_ -> 0, 8))
    return _nbox(F)
end

# Point-by-point density f(x) -> value, or batch density
# f_batch!(values (1, npts), points (3, npts)) as in `build_tree`.
struct _BatchDensity{B}
    f!::B
end

# Sample the density on the grids of boxes `bs` (appended at the end).
function _sample!(F::_Forest, f, bs, x::Vector{Float64})
    n = F.n; np = n^3
    length(F.fvals) < _nbox(F) && resize!(F.fvals, _nbox(F))
    for b in bs
        F.fvals[b] = alloc(F.ar, np)
    end
    if f isa _BatchDensity
        parallel_foreach(bs, [zeros(3, np) for _ in 1:_nthreads()]) do b, pts
            c = F.centers[b]
            h = F.boxsize[F.level[b] + 1] / 2
            R = F.R
            i = 0
            for iz in 1:n, iy in 1:n, ix in 1:n
                i += 1
                y1 = c[1] + h * x[ix]; y2 = c[2] + h * x[iy]; y3 = c[3] + h * x[iz]
                for d in 1:3
                    pts[d, i] = R[d, 1] * y1 + R[d, 2] * y2 + R[d, 3] * y3
                end
            end
            f.f!(reshape(F.fvals[b], 1, np), pts)
        end
        return F
    end
    parallel_foreach(bs, [zeros(3) for _ in 1:_nthreads()]) do b, pt
        c = F.centers[b]
        h = F.boxsize[F.level[b] + 1] / 2
        v = F.fvals[b]
        R = F.R
        i = 0
        for iz in 1:n, iy in 1:n, ix in 1:n
            y1 = c[1] + h * x[ix]; y2 = c[2] + h * x[iy]; y3 = c[3] + h * x[iz]
            @inbounds for d in 1:3
                pt[d] = R[d, 1] * y1 + R[d, 2] * y2 + R[d, 3] * y3
            end
            @inbounds v[i += 1] = f(pt)
        end
    end
    return F
end

# Subdivide boxes `bs` (all leaves) and sample their children.
function _refine!(F::_Forest, f, bs, x::Vector{Float64})
    isempty(bs) && return Int[]
    new = Int[]
    for b in bs
        lev = F.level[b] + 1
        length(F.boxsize) < lev + 1 && push!(F.boxsize, F.boxsize[end] / 2)
        h = F.boxsize[lev + 1] / 2
        c = F.centers[b]
        F.child[b] = ntuple(8) do slot
            s = slot - 1
            cc = (c[1] + (2 * (s & 1) - 1) * h, c[2] + (2 * ((s >> 1) & 1) - 1) * h,
                  c[3] + (2 * ((s >> 2) & 1) - 1) * h)
            _addbox!(F, cc, lev, b)
        end
        append!(new, F.child[b])
    end
    _sample!(F, f, new, x)
    return new
end

# Tail-coefficient refinement error of one box (C++ `bdmk_box_refine_error`
# with eta = 0): sqrt(Σ_tail c²) / sqrt(#tail), c the Legendre coefficients.
function _tail_error(fb, U::Matrix{Float64}, tail::BitArray{3}, w1::Array{Float64,3}, w2::Array{Float64,3})
    n = size(U, 1)
    v = reshape(fb, n, n, n)
    @inbounds for k in 1:n, j in 1:n, i in 1:n          # x
        s = 0.0
        for m in 1:n
            s += U[i, m] * v[m, j, k]
        end
        w1[i, j, k] = s
    end
    @inbounds for k in 1:n, j in 1:n, i in 1:n          # y
        s = 0.0
        for m in 1:n
            s += U[j, m] * w1[i, m, k]
        end
        w2[i, j, k] = s
    end
    acc = 0.0
    @inbounds for k in 1:n, j in 1:n, i in 1:n          # z
        tail[i, j, k] || continue
        s = 0.0
        for m in 1:n
            s += U[k, m] * w2[i, j, m]
        end
        acc += s * s
    end
    return sqrt(acc / count(tail))
end

function _box_l2sq(fb, w3::Vector{Float64}, rsc::Float64)
    s = 0.0
    @inbounds for i in eachindex(w3)
        s += fb[i] * fb[i] * w3[i]
    end
    return s * rsc
end

# Same-level colleagues (self included): grid neighbours at level 0, children
# of the parent's colleagues within 1.05 boxsizes below.
function _colleagues(F::_Forest, g::PeriodicGrid, levorder::Vector{Vector{Int}})
    nb = _nbox(F)
    ncoll = zeros(Int, nb)
    coll = zeros(Int, 27, nb)
    m1, m2, m3 = g.mle
    for b in levorder[1]                       # level-0 boxes are 1:prod(mle) in grid order
        i = (b - 1) % m1; j = ((b - 1) ÷ m1) % m2; k = (b - 1) ÷ (m1 * m2)
        for dk in -1:1, dj in -1:1, di in -1:1
            (0 <= i + di < m1 && 0 <= j + dj < m2 && 0 <= k + dk < m3) || continue
            coll[ncoll[b] += 1, b] = 1 + (i + di) + m1 * ((j + dj) + m2 * (k + dk))
        end
    end
    for ilev in 1:(length(levorder) - 1), b in levorder[ilev + 1]
        dad = F.parent[b]
        tol = 1.05 * F.boxsize[ilev + 1]
        c = F.centers[b]
        for q in 1:ncoll[dad]
            jb = coll[q, dad]
            _isleaf(F, jb) && continue
            for kb in F.child[jb]
                ck = F.centers[kb]
                (abs(ck[1] - c[1]) <= tol && abs(ck[2] - c[2]) <= tol && abs(ck[3] - c[3]) <= tol) &&
                    (coll[ncoll[b] += 1, b] = kb)
            end
        end
    end
    return ncoll, coll
end

function _levelorder(F::_Forest)
    nl = maximum(F.level)
    lv = [Int[] for _ in 0:nl]
    for b in 1:_nbox(F)
        push!(lv[F.level[b] + 1], b)
    end
    return lv
end

# Forest-wide 2:1 balance (C++ `forest_balance_21`): refine every leaf that a
# box two or more levels finer touches, until none does.
function _balance!(F::_Forest, g::PeriodicGrid, f, x::Vector{Float64})
    while true
        lv = _levelorder(F)
        ncoll, coll = _colleagues(F, g, lv)
        flag = Int[]
        for b in 1:_nbox(F)
            _isleaf(F, b) || continue
            L = F.level[b]
            dist = 0.5 * (F.boxsize[L + 1] + F.boxsize[L + 1] / 2) * 1.05
            c = F.centers[b]
            viol = false
            for q in 1:ncoll[b]
                jb = coll[q, b]
                _isleaf(F, jb) && continue
                for kb in F.child[jb]
                    _isleaf(F, kb) && continue
                    ck = F.centers[kb]
                    if abs(ck[1] - c[1]) <= dist && abs(ck[2] - c[2]) <= dist && abs(ck[3] - c[3]) <= dist
                        viol = true
                        break
                    end
                end
                viol && break
            end
            viol && push!(flag, b)
        end
        isempty(flag) && return F
        _refine!(F, f, flag, x)
    end
end

"""
    build_periodic_forest(f, grid::PeriodicGrid; norder=16, eps=1e-6, min_levels=1, f_batch=nothing)

Adaptive forest over `grid` for the periodic density `f(x::AbstractVector) -> Real`
(or the batch form `f_batch!(values::AbstractMatrix, points::AbstractMatrix)`,
values `(1, npts)`, points `(3, npts)`).
Returns `(ft::FlatTree, fvals::Array{Float64,4}, istarget::BitVector)` with boxes
in level order, `fvals` of size `(n, n, n, nboxes)` and `istarget` false on halo
boxes.
"""
function build_periodic_forest(f, g::PeriodicGrid; norder::Int = 16, eps::Float64 = 1e-6, min_levels::Int = 1,
                               f_batch = nothing)
    f_batch === nothing || (f = _BatchDensity(f_batch))
    f === nothing && throw(ArgumentError("provide a density f or a batch density f_batch"))
    F = _Forest(norder, NTuple{3,Float64}[], Int[], Int[], NTuple{8,Int}[], Vector{Float64}[], [g.rc], Arena(), g.Q)
    try
        return _build_forest!(F, f, g, eps, min_levels)
    finally
        release!(F.ar)
    end
end

function _build_forest!(F::_Forest, f, g::PeriodicGrid, eps::Float64, min_levels::Int)
    n = F.n
    leg = ortho1d(LegendreBasis(), n)
    x = leg.x
    w3 = vec([leg.w[i] * leg.w[j] * leg.w[k] for i in 1:n, j in 1:n, k in 1:n])
    tail = BitArray([i^2 + j^2 + k^2 >= n^2 for i in 1:n, j in 1:n, k in 1:n])
    rc = g.rc
    m1, m2, m3 = g.mle
    for k in 0:(m3 - 1), j in 0:(m2 - 1), i in 0:(m1 - 1)
        _addbox!(F, (g.xelo[1] + (i + 0.5) * rc, g.xelo[2] + (j + 0.5) * rc, g.xelo[3] + (k + 0.5) * rc), 0, 0)
    end
    _sample!(F, f, 1:_nbox(F), x)

    # Adaptive refinement, level by level, against the forest-wide L2 norm
    # estimate rint (the per-box scales follow the C++/Fortran builder).
    rintbs = [_box_l2sq(_fbox(F, b), w3, rc^2 / 8) for b in 1:_nbox(F)]
    rint = sqrt(sum(rintbs))
    level_boxes = collect(1:_nbox(F))
    wss = [(zeros(n, n, n), zeros(n, n, n)) for _ in 1:_nthreads()]
    nlevels = 0
    while true
        rsc = sqrt(1 / rc^3) * rint
        errs = zeros(length(level_boxes))
        parallel_foreach(collect(eachindex(level_boxes)), wss) do i, (w1, w2)
            errs[i] = _tail_error(_fbox(F, level_boxes[i]), leg.U, tail, w1, w2)
        end
        flag = [level_boxes[i] for i in eachindex(level_boxes) if errs[i] > eps * rsc]
        isempty(flag) && break
        new = _refine!(F, f, flag, x)
        nlevels += 1
        rintsq = rint^2 - sum(rintbs[b] for b in flag)
        rintsq = max(rintsq, 0.0)
        rsc_child = F.boxsize[nlevels + 1]^3 / 8
        resize!(rintbs, _nbox(F))
        for b in new
            rintbs[b] = _box_l2sq(_fbox(F, b), w3, rsc_child)
            rintsq += rintbs[b]
        end
        rint = sqrt(rintsq)
        level_boxes = new
    end
    # The narrow cascade needs a plane-wave level: refine one interior box.
    while nlevels < min_levels
        b = findfirst(b -> _isleaf(F, b) && !_inhalo(g, F.centers[b]), 1:_nbox(F))
        b === nothing && throw(ArgumentError("no interior box to refine"))
        _refine!(F, f, [b], x)
        nlevels = maximum(F.level)
    end
    nlevels >= 2 && _balance!(F, g, f, x)
    return _flatten_forest(F, g)
end

# Renumber in level order (level 0 in grid order, then children in parent
# order and slot order) and assemble the FlatTree.
function _flatten_forest(F::_Forest, g::PeriodicGrid)
    nb = _nbox(F)
    order = Int[]
    lvl = collect(1:prod(g.mle))
    while !isempty(lvl)
        append!(order, lvl)
        nxt = Int[]
        for b in lvl
            _isleaf(F, b) || append!(nxt, F.child[b])
        end
        lvl = nxt
    end
    @assert length(order) == nb
    newid = zeros(Int, nb)
    newid[order] = 1:nb
    nl = maximum(F.level)
    level = F.level[order]
    parent = [F.parent[b] == 0 ? 0 : newid[F.parent[b]] for b in order]
    nchild = zeros(Int, nb)
    child = zeros(Int, 8, nb)
    centers = zeros(3, nb)
    for (i, b) in enumerate(order)
        centers[:, i] .= F.centers[b]
        _isleaf(F, b) && continue
        nchild[i] = 8
        child[:, i] .= newid[collect(F.child[b])]
    end
    laddr = zeros(Int, 2, nl + 1)
    for ilev in 0:nl
        r = searchsorted(level, ilev)
        laddr[1, ilev + 1] = first(r); laddr[2, ilev + 1] = last(r)
    end
    # colleagues on the renumbered forest
    G = _Forest(F.n, [Tuple(centers[:, i]) for i in 1:nb], level, parent,
                [nchild[i] == 0 ? ntuple(_ -> 0, 8) : Tuple(child[:, i]) for i in 1:nb],
                Vector{Float64}[], F.boxsize, F.ar, F.R)
    ncoll, coll = _colleagues(G, g, [collect(laddr[1, l]:laddr[2, l]) for l in 1:nl + 1])
    ft = FlatTree(nb, nl, collect(1:nb), laddr, level, parent, nchild, child, ncoll, coll, centers,
                  F.boxsize[1:nl + 1])
    np = F.n^3
    fvals = Array{Float64}(undef, F.n, F.n, F.n, nb)
    parallel_foreach(collect(1:nb), fill(nothing, _nthreads())) do i, _
        copyto!(fvals, (i - 1) * np + 1, F.fvals[order[i]], 1, np)
    end
    # targets: boxes of the central grid that meet the cell
    istarget = BitVector([!_inhalo(g, view(centers, :, i)) && _meets_cell(g, view(centers, :, i), F.boxsize[level[i] + 1] / 2)
                          for i in 1:nb])
    return ft, fvals, istarget
end
