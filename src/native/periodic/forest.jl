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
plus `halo[d]` image layers: `mle = mgrid .+ 2 .* halo` boxes per axis from
`xelo = xgrid0 .- halo .* rc`.

A `slab` grid (quasi-2D: periodic in x and y, free in z) has `A = diag(L₁, L₂, H)`
for the slab `0 ≤ z ≤ H` (`zorg` + that range in the caller's frame), no image
layer in z, and `mgrid[3]` cubes in z from the slab bottom.
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
    halo::NTuple{3,Int}         # image layers per axis
    slab::Bool                  # periodic in x, y only
    zorg::Float64               # slab: z of the slab bottom in the caller's frame
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

function PeriodicGrid(A::AbstractMatrix, rc::Float64, nhalo::Int; slab::Bool = false, zorg::Float64 = 0.0)
    slab && return _slab_grid(A, rc, nhalo, zorg)
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
                        mgrid .+ 2nhalo, xgrid0 .- nhalo * rc, diagonal, tiled, (nhalo, nhalo, nhalo), false, 0.0)
end

# Quasi-2D grid: in-plane cell A₂ = Q₂U₂ (columns a₁, a₂ in the xy plane),
# covered by the bounding box of rc cubes in the rotated frame (tiled when
# rectangular with sides multiple of rc), the slab 0 ≤ z ≤ H by mgrid[3] cubes
# from z = 0, images in x and y only.
function _slab_grid(A::AbstractMatrix, rc::Float64, nhalo::Int, zorg::Float64)
    size(A) == (3, 3) && all(abs(A[3, j]) <= 1e-14 * norm(A) for j in 1:2) && A[1, 3] == A[2, 3] == 0 && A[3, 3] > 0 ||
        throw(ArgumentError("slab cell: in-plane lattice vectors (x, y) and the slab height"))
    Q, U = _lattice_qr(A)
    U[1:2, 3] .= 0.0
    Uinv = Matrix(inv(UpperTriangular(U)))
    H = U[3, 3]
    lo1 = min(0.0, U[1, 2])
    m = (max(1, ceil(Int, (U[1, 1] + abs(U[1, 2])) / rc * (1 - 1e-12))), max(1, ceil(Int, U[2, 2] / rc * (1 - 1e-12))),
         max(1, ceil(Int, H / rc * (1 - 1e-12))))
    diagonal = abs(U[1, 2]) <= 1e-12 * U[1, 1]
    tiled = diagonal && all(abs(round(U[d, d] / rc) * rc - U[d, d]) <= 1e-9 * U[d, d] for d in 1:2)
    tiled && (m = (round(Int, U[1, 1] / rc), round(Int, U[2, 2] / rc), m[3]))
    xg = (lo1, 0.0, 0.0)                                  # z-cells from the slab bottom up
    halo = (nhalo, nhalo, 0)
    dmin = minimum(1 / norm(view(Uinv, i, 1:2)) for i in 1:2)
    return PeriodicGrid(Matrix{Float64}(A), Q, U, Uinv, abs(det(U)), dmin, rc, m, xg, nhalo,
                        m .+ 2 .* halo, xg .- halo .* rc, diagonal, tiled, halo, true, zorg)
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
        (k < g.halo[d] || k >= g.mle[d] - g.halo[d]) && return true
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
    core::Vector{Bool}              # sampled from the density (else interpolated from the core)
    ci::Any                         # nothing, or the _CoreInterp of the non-core boxes
    origin::Vector{UInt8}           # where the values come from: _SAMPLED, _FROMPARENT, _INTERP, _COPY
end

_Forest(n::Int, bs0::Float64, ar::Arena, R::Matrix{Float64}) =
    _Forest(n, NTuple{3,Float64}[], Int[], Int[], NTuple{8,Int}[], Vector{Float64}[], [bs0], ar, R, Bool[], nothing, UInt8[])

const _SAMPLED = 0x00       # the density
const _FROMPARENT = 0x01    # the parent's polynomial
const _INTERP = 0x02        # the core leaves at the periodic images of the nodes
const _COPY = 0x03          # the lattice image (tiled cells)

_nbox(F::_Forest) = length(F.level)
_isleaf(F::_Forest, b::Int) = F.child[b][1] == 0
_fbox(F::_Forest, b::Int) = F.fvals[b]

function _addbox!(F::_Forest, c::NTuple{3,Float64}, lev::Int, par::Int)
    push!(F.centers, c); push!(F.level, lev); push!(F.parent, par)
    push!(F.child, ntuple(_ -> 0, 8))
    push!(F.core, par == 0 ? true : F.core[par])
    push!(F.origin, _SAMPLED)
    return _nbox(F)
end

# Stands in for the density once the adaptive refinement is done (`sample_once`):
# new boxes take their values from the parent's polynomial, which resolves f.
# T[s + 1][i, j] = ℓ_j((x_i + 2s - 1)/2): child (side s) node values from the
# parent's node values along one axis; Tt = transposes.
struct _FromParent
    T::NTuple{2,Matrix{Float64}}
    Tt::NTuple{2,Matrix{Float64}}
end

function _FromParent(n::Int)
    leg = ortho1d(LegendreBasis(), n)
    pv = zeros(n)
    T = ntuple(2) do s1
        P = zeros(n, n)
        for i in 1:n
            legepols!(pv, (leg.x[i] + 2 * (s1 - 1) - 1) / 2, n)
            P[i, :] .= pv
        end
        P * leg.U
    end
    return _FromParent(T, map(m -> Matrix(transpose(m)), T))
end

# Values of child `slot` (0-based, 1 + slot = 1 + sx + 2sy + 4sz) from its parent's.
@inline _parent_to_child!(out, inp, fp::_FromParent, slot::Int, w1, w2) =
    tp3!(out, inp, fp.T[(slot & 1) + 1], fp.Tt[((slot >> 1) & 1) + 1], fp.Tt[((slot >> 2) & 1) + 1], w1, w2, 1.0, false)

function _sample!(F::_Forest, fp::_FromParent, bs, x::Vector{Float64})
    np = F.n^3
    length(F.fvals) < _nbox(F) && resize!(F.fvals, _nbox(F))
    for b in bs
        F.fvals[b] = alloc(F.ar, np)
        F.origin[b] = _FROMPARENT
    end
    parallel_foreach(collect(bs), [(alloc(F.ar, np), alloc(F.ar, np)) for _ in 1:_nthreads()]) do b, (w1, w2)
        P = F.parent[b]
        _parent_to_child!(F.fvals[b], F.fvals[P], fp, findfirst(==(b), F.child[P]) - 1, w1, w2)
    end
    return F
end

# Point-by-point density f(x) -> value, or batch density
# f_batch!(values (1, npts), points (3, npts)) as in `build_tree`.
struct _BatchDensity{B}
    f!::B
end

# Sample the density on the grids of boxes `bs` (appended at the end);
# non-core boxes are interpolated from the core forest.
function _sample!(F::_Forest, f, bs, x::Vector{Float64})
    np = F.n^3
    length(F.fvals) < _nbox(F) && resize!(F.fvals, _nbox(F))
    for b in bs
        F.fvals[b] = alloc(F.ar, np)
    end
    if F.ci === nothing || all(b -> F.core[b], bs)
        return _sample_density!(F, f, bs, x)
    end
    _sample_density!(F, f, [b for b in bs if F.core[b]], x)     # core first: the others read it
    return _interp_sample!(F, [b for b in bs if !F.core[b]], x)
end

function _sample_density!(F::_Forest, f, bs, x::Vector{Float64})
    n = F.n; np = n^3
    isempty(bs) && return F
    if f isa _BatchDensity
        parallel_foreach(bs, [alloc(F.ar, 3, np) for _ in 1:_nthreads()]) do b, pts
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
function _tail_error(fb, U::Matrix{Float64}, Ut::Matrix{Float64}, tail::BitArray{3},
                     c::Vector{Float64}, w1::Vector{Float64}, w2::Vector{Float64})
    tp3!(c, fb, U, Ut, Ut, w1, w2, 1.0, false)
    acc = 0.0
    @inbounds for i in eachindex(c)
        tail[i] && (acc += c[i] * c[i])
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
                _near(g, F.centers[kb], c, tol, false) && (coll[ncoll[b] += 1, b] = kb)
            end
        end
    end
    return ncoll, coll
end

# |a - b| ≤ tol per axis (minimum image along the cell sides if periodic;
# tiled cells only).
@inline function _near(g::PeriodicGrid, a, b, tol::Float64, periodic::Bool)
    for d in 1:3
        dx = a[d] - b[d]
        periodic && (dx -= g.U[d, d] * round(dx / g.U[d, d]))
        abs(dx) <= tol || return false
    end
    return true
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
function _balance!(F::_Forest, g::PeriodicGrid, f, x::Vector{Float64}; img = nothing)
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
                    if _near(g, F.centers[kb], c, dist, false)
                        viol = true
                        break
                    end
                end
                viol && break
            end
            viol && push!(flag, b)
        end
        isempty(flag) && return F
        img === nothing ? _refine!(F, f, flag, x) : _refine_from_images!(F, g, img, f, flag, x)
    end
end

# Tiled cells: refine `flag`, copying children (and samples) from the
# lattice image of a halo box when that image is already refined.
function _refine_from_images!(F::_Forest, g::PeriodicGrid, img::Vector{Int}, f, flag::Vector{Int}, x::Vector{Float64})
    nb = _nbox(F)
    root = zeros(Int, nb)
    for b in 1:nb
        root[b] = F.parent[b] == 0 ? b : root[F.parent[b]]
    end
    # central boxes first, so that their halo images can copy them
    _refine!(F, f, [b for b in flag if img[root[b]] == root[b]], x)
    nb = _nbox(F)
    resize!(root, nb)
    for b in 1:nb
        root[b] = F.parent[b] == 0 ? b : root[F.parent[b]]
    end
    key(b, c) = (F.level[b], ntuple(d -> round(Int, (c[d] - g.xelo[d]) / F.boxsize[F.level[b] + 1] - 0.5), 3))
    index = Dict(key(b, F.centers[b]) => b for b in 1:nb if img[root[b]] == root[b])
    sample = Int[]
    for b in flag
        img[root[b]] == root[b] && continue
        r = root[b]
        src = 0
        if img[r] != r
            sh = F.centers[img[r]] .- F.centers[r]
            src = get(index, key(b, F.centers[b] .+ sh), 0)
        end
        if src == 0 || _isleaf(F, src)
            push!(sample, b)
            continue
        end
        sh = F.centers[b] .- F.centers[src]
        F.child[b] = ntuple(slot -> _addbox!(F, F.centers[F.child[src][slot]] .+ sh, F.level[b] + 1, b), 8)
        resize!(F.fvals, _nbox(F))
        for slot in 1:8
            F.fvals[F.child[b][slot]] = F.fvals[F.child[src][slot]]
            F.origin[F.child[b][slot]] = _COPY
        end
    end
    _refine!(F, f, sample, x)
    return F
end

"""
    build_periodic_forest(f, grid::PeriodicGrid; norder=16, eps=1e-6, min_levels=1, f_batch=nothing,
                          interpolate=true, sample_once=true)

Adaptive forest over `grid` for the periodic density `f(x::AbstractVector) -> Real`
(or the batch form `f_batch!(values::AbstractMatrix, points::AbstractMatrix)`,
values `(1, npts)`, points `(3, npts)`).
Returns `(ft::FlatTree, fvals::Array{Float64,4}, istarget::BitVector, core::BitVector,
origin::Vector{UInt8}, tsample)` with boxes in level order, `fvals` of size
`(n, n, n, nboxes)`, `core` false on the boxes interpolated from the others,
`istarget` false on halo boxes, `origin` where each box's values come from,
and `tsample` the time (s) of the adaptive refinement of the core, the only
stage that evaluates `f` with `sample_once` (later refinements, for the 2:1
balance, take the parent's polynomial).
"""
function build_periodic_forest(f, g::PeriodicGrid; norder::Int = 16, eps::Float64 = 1e-6, min_levels::Int = 1,
                               f_batch = nothing, interpolate::Bool = true, sample_once::Bool = true)
    f_batch === nothing || (f = _BatchDensity(f_batch))
    f === nothing && throw(ArgumentError("provide a density f or a batch density f_batch"))
    F = _Forest(norder, g.rc, Arena(), g.Q)
    try
        return _build_forest!(F, f, g, eps, min_levels, interpolate, sample_once)
    finally
        release!(F.ar)
    end
end

function _build_forest!(F::_Forest, f, g::PeriodicGrid, eps::Float64, min_levels::Int, interpolate::Bool,
                        sample_once::Bool)
    t0 = time_ns()
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
    # Only part of the forest samples the density:
    # - when the central cubes tile the cell, each halo cube is a lattice
    #   translate of a central one; only the central forest is sampled and
    #   refined, and the halo subtrees are copied from their images;
    # - otherwise (with `interpolate`) the core of boxes that meet the cell is
    #   sampled and refined first; every other box takes its values from the
    #   core leaves at the periodic images of its nodes, and refines by the
    #   same test with the thresholds of the core levels.
    # The norm estimate counts every sampled box once per box it stands for,
    # as if the whole extended grid had been refined (like the C++ builder).
    nroot = _nbox(F)
    img = g.tiled ? [_image_root(g, b) for b in 1:nroot] : collect(1:nroot)
    if !g.tiled && interpolate
        for b in 1:nroot
            F.core[b] = _meets_cell(g, F.centers[b], rc / 2)
        end
    end
    active = [b for b in 1:nroot if img[b] == b && F.core[b]]
    mult = zeros(nroot)
    for b in 1:nroot
        mult[img[b]] += 1
    end
    g.tiled || (mult[active] .= nroot / length(active))
    _sample!(F, f, active, x)

    # Adaptive refinement, level by level, against the forest-wide L2 norm
    # estimate rint (the per-box scales follow the C++/Fortran builder).
    rintbs = zeros(nroot)
    wbox = copy(mult)                      # boxes stood for (inherited by children)
    for b in active
        rintbs[b] = _box_l2sq(_fbox(F, b), w3, rc^2 / 8)
    end
    rint = sqrt(sum(wbox .* rintbs))
    level_boxes = active
    wss = [(zeros(n^3), zeros(n^3), zeros(n^3)) for _ in 1:_nthreads()]
    Ut = Matrix(transpose(leg.U))
    thr = Float64[]                        # refinement threshold per level
    function refine_level(boxes, t)
        errs = zeros(length(boxes))
        parallel_foreach(collect(eachindex(boxes)), wss) do i, (c, w1, w2)
            errs[i] = _tail_error(_fbox(F, boxes[i]), leg.U, Ut, tail, c, w1, w2)
        end
        return [boxes[i] for i in eachindex(boxes) if errs[i] > t]
    end
    nlevels = 0
    while true
        push!(thr, eps * sqrt(1 / rc^3) * rint)
        flag = refine_level(level_boxes, thr[end])
        isempty(flag) && break
        new = _refine!(F, f, flag, x)
        nlevels += 1
        rintsq = rint^2 - sum(wbox[b] * rintbs[b] for b in flag)
        rintsq = max(rintsq, 0.0)
        rsc_child = F.boxsize[nlevels + 1]^3 / 8
        resize!(rintbs, _nbox(F)); resize!(wbox, _nbox(F))
        for b in new
            wbox[b] = wbox[F.parent[b]]
            rintbs[b] = _box_l2sq(_fbox(F, b), w3, rsc_child)
            rintsq += wbox[b] * rintbs[b]
        end
        rint = sqrt(rintsq)
        level_boxes = new
    end
    tsample = (time_ns() - t0) / 1e9
    fs = sample_once ? _FromParent(n) : f       # density of the later (balance) refinements
    if !g.tiled && interpolate
        F.ci = _CoreInterp(g, leg)
        level_boxes = [b for b in 1:nroot if !F.core[b]]
        _sample!(F, f, level_boxes, x)
        L = 0
        while !isempty(level_boxes)
            flag = refine_level(level_boxes, 2 * thr[min(L + 1, end)])   # slack: the data are piecewise (core leaves)
            isempty(flag) && break
            level_boxes = _refine!(F, f, flag, x)
            L += 1
        end
        nlevels = maximum(F.level)
    end
    # The narrow cascade needs a plane-wave level: refine one interior box.
    while nlevels < min_levels
        b = findfirst(b -> _isleaf(F, b) && !_inhalo(g, F.centers[b]), 1:_nbox(F))
        b === nothing && throw(ArgumentError("no interior box to refine"))
        _refine!(F, fs, [b], x)
        nlevels = maximum(F.level)
    end
    g.tiled && _copy_images!(F, img)
    maximum(F.level) >= 2 && _balance!(F, g, fs, x; img = g.tiled ? img : nothing)
    return (_flatten_forest(F, g)..., sample_once ? tsample : (time_ns() - t0) / 1e9)
end

# Interpolation of the non-core boxes from the core leaves: their Legendre
# coefficients (computed on demand) and, per box, the nodes grouped by the
# core leaf holding their periodic image; on each leaf the nodes are a subset
# of a tensor grid, evaluated by three small products.
mutable struct _CoreInterp
    g::PeriodicGrid
    leg::Ortho1D
    coef::Dict{Int,Vector{Float64}}
    ws::Any                         # per-thread workspaces (`_interp_workspaces`)
end
_CoreInterp(g::PeriodicGrid, leg::Ortho1D) =
    _CoreInterp(g, leg, Dict{Int,Vector{Float64}}(), _interp_workspaces(length(leg.x)))

function _core_leaf(F::_Forest, g::PeriodicGrid, y)
    b = 1 + _gridcell(g, y[1], 1) + g.mle[1] * (_gridcell(g, y[2], 2) + g.mle[2] * _gridcell(g, y[3], 3))
    while !_isleaf(F, b)
        c = F.centers[b]
        b = F.child[b][1 + (y[1] > c[1]) + 2 * (y[2] > c[2]) + 4 * (y[3] > c[3])]
    end
    return b
end

function _interp_sample!(F::_Forest, bs::Vector{Int}, x::Vector{Float64})
    ci = F.ci::_CoreInterp
    g = ci.g; n = F.n
    # coefficients of the core leaves not seen yet
    need = [b for b in 1:_nbox(F) if F.core[b] && _isleaf(F, b) && !haskey(ci.coef, b)]
    newc = _leaf_coefs(F.ar, n, [F.fvals[b] for b in need], ci.ws)
    for (i, b) in enumerate(need)
        ci.coef[b] = newc[i]
    end
    locate = y -> _core_leaf(F, g, y)
    geom = L -> (F.centers[L], F.boxsize[F.level[L] + 1] / 2)
    coef = L -> ci.coef[L]
    for b in bs
        F.origin[b] = _INTERP
    end
    parallel_foreach(bs, ci.ws) do b, w
        _interp_values!(F.fvals[b], F.centers[b], F.boxsize[F.level[b] + 1] / 2, x, g, n, locate, geom, coef, w)
    end
    return F
end

# Legendre coefficients of value arrays (n³ each).
function _leaf_coefs(ar::Arena, n::Int, vals, ws = _interp_workspaces(n))
    U = ortho1d(LegendreBasis(), n).U
    Ut = Matrix(transpose(U))
    out = [alloc(ar, n^3) for _ in vals]
    parallel_foreach(collect(eachindex(vals)), ws) do i, w
        tp3!(out[i], vals[i], U, Ut, Ut, w.w1, w.w2, 1.0, false)
    end
    return out
end

_interp_workspaces(n::Int) = [(gid = zeros(Int, n^3), grp = Tuple{Int,NTuple{3,Float64}}[], mask = falses(n, 3),
                               pos = zeros(Int, n, 3), w1 = zeros(n^3), w2 = zeros(n^3), out = zeros(n^3),
                               P1t = zeros(n, n), P2 = zeros(n, n), P3 = zeros(n, n), pv = zeros(n))
                              for _ in 1:_nthreads()]

# Values v at the nodes of the box (center c, half side h) from the core:
# `locate(y)` is the core leaf at the in-cell point y, `geom(L)` its (center,
# half side), `coef(L)` its Legendre coefficients. The nodes are grouped by
# the leaf holding their periodic image (the image shift is the same on a
# leaf, which is smaller than the lattice spacing); on each leaf they are a
# subset of a tensor grid, evaluated by three small products.
function _interp_values!(v, c, h::Float64, x::Vector{Float64}, g::PeriodicGrid, n::Int, locate, geom, coef, w)
    Ui, U, slab = g.Uinv, g.U, g.slab
    gid = w.gid; grp = w.grp
    empty!(grp)
    Lc = 0; lo = (0.0, 0.0, 0.0); hi = (0.0, 0.0, 0.0); gc = 0
    q = 0
    @inbounds for iz in 1:n, iy in 1:n, ix in 1:n
        q += 1
        p = (c[1] + h * x[ix], c[2] + h * x[iy], c[3] + h * x[iz])
        t = ntuple(3) do d
            d == 3 && slab && return 0.0                              # free in z
            u = Ui[d, 1] * p[1] + Ui[d, 2] * p[2] + Ui[d, 3] * p[3]
            clamp(u - floor(u), 1e-12, 1 - 1e-12) - u                 # wrapped - unwrapped
        end
        sh = ntuple(d -> U[d, 1] * t[1] + U[d, 2] * t[2] + U[d, 3] * t[3], 3)   # image shift (rotated frame)
        y = (p[1] + sh[1], p[2] + sh[2], p[3] + sh[3])
        # the previous node's leaf holds most images (strictly inside: same leaf as `locate`)
        if !(Lc != 0 && lo[1] < y[1] < hi[1] && lo[2] < y[2] < hi[2] && lo[3] < y[3] < hi[3])
            Lc = locate(y)
            cl, hl = geom(Lc)
            lo = (cl[1] - hl, cl[2] - hl, cl[3] - hl); hi = (cl[1] + hl, cl[2] + hl, cl[3] + hl)
            gc = 0
            for e in eachindex(grp)
                grp[e][1] == Lc && (gc = e; break)
            end
            if gc == 0
                push!(grp, (Lc, sh))
                gc = length(grp)
            end
        end
        gid[q] = gc
    end
    mask = w.mask; pos = w.pos; pv = w.pv
    for (gi, (L, sh)) in enumerate(grp)
        cl, hl = geom(L)
        # index sets of this leaf's nodes along each axis
        fill!(mask, false)
        q = 0
        @inbounds for k in 1:n, j in 1:n, i in 1:n
            q += 1
            gid[q] == gi && (mask[i, 1] = mask[j, 2] = mask[k, 3] = true)
        end
        nr = ntuple(3) do d
            r = 0
            for i in 1:n
                mask[i, d] || continue
                pos[i, d] = (r += 1)
                legepols!(pv, (c[d] + h * x[i] + sh[d] - cl[d]) / hl, n)
                P = d == 1 ? w.P1t : d == 2 ? w.P2 : w.P3
                for a in 1:n
                    d == 1 ? (P[r, a] = pv[a]) : (P[a, r] = pv[a])
                end
            end
            r
        end
        ni, nj, nk = nr
        tp3!(view(w.out, 1:ni * nj * nk), coef(L), view(w.P1t, 1:ni, :), view(w.P2, :, 1:nj), view(w.P3, :, 1:nk),
             w.w1, w.w2, 1.0, false)
        q = 0
        @inbounds for k in 1:n, j in 1:n, i in 1:n
            q += 1
            gid[q] == gi || continue
            v[q] = w.out[pos[i, 1] + ni * ((pos[j, 2] - 1) + nj * (pos[k, 3] - 1))]
        end
    end
    return v
end

# Central level-0 box whose lattice translate the level-0 box b is (tiled cells).
function _image_root(g::PeriodicGrid, b::Int)
    m1, m2 = g.mle[1], g.mle[2]
    e = ((b - 1) % m1, ((b - 1) ÷ m1) % m2, (b - 1) ÷ (m1 * m2))
    c = ntuple(d -> mod(e[d] - g.halo[d], g.mgrid[d]) + g.halo[d], 3)
    return 1 + c[1] + m1 * (c[2] + m2 * c[3])
end

# Give every halo root the subtree (and samples, shared) of its image.
function _copy_images!(F::_Forest, img::Vector{Int})
    function copysub!(src, dst)
        F.fvals[dst] = F.fvals[src]
        F.origin[dst] = _COPY
        _isleaf(F, src) && return
        sh = F.centers[dst] .- F.centers[src]
        F.child[dst] = ntuple(8) do slot
            c = F.child[src][slot]
            _addbox!(F, F.centers[c] .+ sh, F.level[c], dst)
        end
        length(F.fvals) < _nbox(F) && resize!(F.fvals, _nbox(F))
        for slot in 1:8
            copysub!(F.child[src][slot], F.child[dst][slot])
        end
    end
    length(F.fvals) < _nbox(F) && resize!(F.fvals, _nbox(F))
    for b in eachindex(img)
        img[b] == b || copysub!(img[b], b)
    end
    return F
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
                Vector{Float64}[], F.boxsize, F.ar, F.R, F.core[order], nothing, F.origin[order])
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
    return ft, fvals, istarget, BitVector(F.core[order]), F.origin[order]
end
