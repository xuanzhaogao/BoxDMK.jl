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
    target_pot::Union{Nothing,Vector{Float64}}
end

"""
    bdmk_periodic(f, cell; eps=1e-6, norder=16, rc_factor=0.25, nhalo=1, f_batch=nothing,
                  skip_negligible=true, interpolate=true, sample_once=true, timings=nothing)

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
the forest of the C++ builder. With `sample_once`, `f` is evaluated only by
the adaptive refinement of the boxes that meet the cell: the boxes added for
the 2:1 balance take their parent's polynomial and the wide source tree of
non-tiled cells interpolates the forest (values accurate to `eps` instead of
to rounding; `false` samples `f` everywhere, as the C++ code). `targets` (3 × N
points) restricts the computation to the leaves holding them and returns the
potential there in `target_pot`. `timings` (length ≥ 6)
receives wall times (s): [1] sampling (the adaptive refinement that evaluates
`f`; with `sample_once = false` the whole forest), [2] the rest of the forest,
setup and wide source, [3] local
expansions + upward pass, [4] wide part, [5] plane waves, [6] near field, and
if length ≥ 10 the plane-wave split [7] formation, [8] translation,
[9] evaluation, [10] downward pass.
"""
function bdmk_periodic(f, cell; f_batch = nothing, timings::Union{Nothing,Vector{Float64}} = nothing, kwargs...)
    tm = timings === nothing ? zeros(10) : timings
    fill!(tm, 0.0)
    t0 = time_ns()
    plan = periodic_plan(f, cell; f_batch, kwargs...)
    tb = (time_ns() - t0) / 1e9
    res = periodic_solve!(plan, f; f_batch, timings = tm)
    tm[1] = plan.tsample; tm[2] += tb - plan.tsample
    return res
end

"""
    bdmk_q2d(f, L, zlim; eps=1e-6, norder=16, rc=nothing, charged=false, f_batch=nothing, timings=nothing, ...)

Quasi-2D (slab) Coulomb potential: periodic in x and y with the in-plane
lattice `L` (a 2×2 matrix of lattice vectors as columns, a tuple `(L₁, L₂)`
for a rectangle or a number for a square), free in z,

    u(x) = ∫ Σ_{n ∈ ℤ²} f(y) / |x - y - (Ln, 0)| dy,

for a density `f` that vanishes outside the slab `zlim = (z₀, z₁)`, neutral
unless `charged = true` (then in the gauge `u + 2πQ|z|/A → 0` far from the
slab, `Q` the charge per cell, `A` its area). The potential is computed in the
slab (`evaluate_periodic(res, x)` for z₀ ≤ x₃ ≤ z₁). The level-0 boxes have
side `rc ≤ min(d, z₁ - z₀)`, `d` the in-plane spacing of the lattice lines; by
default, for a rectangle the largest such value dividing both sides (the
cubes then tile the cell), else the bound. The real-space (narrow) part is
that of [`bdmk_periodic`](@ref) with images in x and y only; the wide part is
summed in the plane over the modes `k̇ = 2πL⁻ᵀm` and applied in z through
exact Gaussian kernel matrices on the level-0 z-cells, the mode `k̇ = 0`
through the kernel `-2π|z - z'|/A` of all wide Gaussians. Other options and
`timings` as for `bdmk_periodic`; the forest and `res.pot` are in the frame
`(x, y, z - z₀)` rotated so that the first lattice vector lies along x.
"""
bdmk_q2d(f, L, zlim; kwargs...) = bdmk_periodic(f, _slab_cell(L, zlim); slab = true, zorg = Float64(zlim[1]), kwargs...)

"""
    q2d_plan(f, L, zlim; kwargs...) -> PeriodicPlan

Plan of [`bdmk_q2d`](@ref) for repeated solves with [`periodic_solve!`](@ref).
"""
q2d_plan(f, L, zlim; kwargs...) = periodic_plan(f, _slab_cell(L, zlim); slab = true, zorg = Float64(zlim[1]), kwargs...)

function _slab_cell(L, zlim)
    H = zlim[2] - zlim[1]
    H > 0 || throw(ArgumentError("need z₀ < z₁"))
    if L isa AbstractMatrix
        size(L) == (2, 2) && abs(det(L)) > 0 || throw(ArgumentError("the in-plane cell must be a 2×2 matrix of lattice vectors (columns)"))
        return Float64[L[1, 1] L[1, 2] 0; L[2, 1] L[2, 2] 0; 0 0 H]
    end
    L1, L2 = L isa Real ? (L, L) : (L[1], L[2])
    (L1 > 0 && L2 > 0) || throw(ArgumentError("need positive sides"))
    return Matrix{Float64}(Diagonal([L1, L2, H]))
end

_cell_matrix(L::Real) = Matrix(Float64(L) * I, 3, 3)
_cell_matrix(L::Tuple) = Matrix(Diagonal(collect(Float64.(L))))
_cell_matrix(A::AbstractMatrix) = Matrix{Float64}(A)

"""
    PeriodicPlan

Forest and precomputation of [`bdmk_periodic`](@ref) for repeated solves on a
fixed forest (e.g. time stepping): `plan = periodic_plan(f, cell; kwargs...)`,
then `periodic_solve!(plan, f)` for each new density (the first solve uses the
samples taken by the plan). The result of a solve shares the plan's buffers
(`pot`, `fvals`), which the next solve overwrites. The forest is adapted to
the density given to `periodic_plan`; later densities must be resolved by it.
"""
mutable struct PeriodicPlan
    g::PeriodicGrid
    ft::FlatTree
    n::Int
    eps::Float64
    istarget::BitVector
    core::BitVector
    origin::Vector{UInt8}         # where each box's values come from (_SAMPLED, _FROMPARENT, _INTERP, _COPY)
    sample_once::Bool
    copysrc::Vector{Int}          # tiled cells: box whose samples a box copies (0: sampled)
    s::DMKSetup
    wplan::Union{WidePlan,SlabWide}
    wn::Vector{Float64}
    dn::Vector{Float64}
    taylor::NTuple{3,Float64}
    skip::Bool
    fvals::Array{Float64,4}
    pot::Array{Float64,4}
    fresh::Bool                   # fvals hold the samples of the plan's density
    tforest::Float64
    tsample::Float64              # part of tforest evaluating f
    targets::Union{Nothing,Matrix{Float64}}
    iws::Any                      # interpolation workspaces of `_resample!` (created on first use)
end

"""
    periodic_plan(f, cell; eps=1e-6, norder=16, rc_factor=0.25, nhalo=1, f_batch=nothing,
                  skip_negligible=true, interpolate=true, sample_once=true, targets=nothing,
                  near32=nothing) -> PeriodicPlan

Forest (adapted to `f`), tables and buffers of [`bdmk_periodic`](@ref), for
repeated solves with [`periodic_solve!`](@ref); the options are those of
`bdmk_periodic`. With `targets` (3 × N points), only the leaves holding a target are computed
and solves return the potential at the targets (`target_pot`). `near32`
runs the dense same-level near field in single precision (default: when
eps ≥ 1e-6).
"""
function periodic_plan(f, cell; eps::Real = 1e-6, norder::Int = 16, rc_factor::Real = 0.25, nhalo::Int = 1,
                       f_batch = nothing, skip_negligible::Bool = true, interpolate::Bool = true, sample_once::Bool = true,
                       targets::Union{Nothing,AbstractMatrix} = nothing, near32::Union{Nothing,Bool} = nothing,
                       slab::Bool = false, zorg::Real = 0.0, rc::Union{Nothing,Real} = nothing, charged::Bool = false)
    f = _density(f, f_batch, Float64(zorg))
    old_ftz = get_zero_subnormals()
    set_zero_subnormals(true)
    try
        return _periodic_plan(f, _cell_matrix(cell), Float64(eps), norder, Float64(rc_factor), nhalo,
                              skip_negligible, interpolate, sample_once, targets === nothing ? nothing : Matrix{Float64}(targets),
                              near32, slab, Float64(zorg), rc, charged)
    finally
        set_zero_subnormals(old_ftz)
    end
end

function _periodic_plan(f, A::Matrix{Float64}, eps::Float64, n::Int, rc_factor::Float64, nhalo::Int,
                        skip::Bool, interpolate::Bool, sample_once::Bool, targets, near32,
                        slab::Bool = false, zorg::Float64 = 0.0, rcin = nothing, charged::Bool = false)
    nhalo >= 1 || throw(ArgumentError("nhalo must be ≥ 1"))
    0 < rc_factor < 0.95 || throw(ArgumentError("need 0 < rc_factor < 0.95"))
    size(A) == (3, 3) && abs(det(A)) > 0 || throw(ArgumentError("degenerate cell"))
    _, U = _lattice_qr(A)
    Uinv = inv(UpperTriangular(U))
    # (slab: in-plane distance between lattice lines, at most min(|a₁|, |a₂|))
    dmin = slab ? minimum(1 / norm(view(Uinv, i, 1:2)) for i in 1:2) : minimum(1 / norm(Uinv[i, :]) for i in 1:3)
    rc = rcin !== nothing ? Float64(rcin) : slab ? _slab_rc(U, dmin) : dmin / ceil(1 / rc_factor * (1 - 1e-12))
    slab && rc > min(dmin, U[3, 3]) * (1 + 1e-12) &&
        throw(ArgumentError("rc = $rc must not exceed the in-plane lattice spacing and the slab height, min = $(min(dmin, U[3, 3]))"))
    g = PeriodicGrid(A, rc, nhalo; slab, zorg)

    t0 = time_ns()
    ft, fvals, istarget, core, origin, tsample = build_periodic_forest(f, g; norder = n, eps = eps, min_levels = 1,
                                                                       interpolate, sample_once)
    tforest = (time_ns() - t0) / 1e9

    r0, ws, ts = useries_sog(round(Int, -log10(eps)), rc)
    dmin > 2r0 || throw(ArgumentError("cell too small for the SOG"))
    iswide = [sqrt(t * log(1 / eps)) > rc for t in ts]
    any(iswide) || throw(ArgumentError("no wide Gaussians; increase rc_factor"))
    narrow = sort(findall(.!iswide); by = i -> -ts[i])     # δ descending
    dn, wn = ts[narrow], ws[narrow]
    tsw, wsw = ts[iswide], ws[iswide]
    taylor = taylor_coefficients(LaplaceKernel(), r0, wn, dn)
    if targets !== nothing                 # the leaves holding a target, and their ancestors
        size(targets, 1) == 3 || throw(ArgumentError("targets must be 3 × N"))
        istarget = falses(ft.nboxes)
        for j in axes(targets, 2)
            b = _locate_leaf(ft, g, _wrap_into_cell(g, view(targets, :, j)))
            while b > 0 && !istarget[b]
                istarget[b] = true
                b = ft.parent[b]
            end
        end
    end
    # dense same-level near field in single precision at eps ≥ 1e-6 (as the C++ code)
    s = DMKSetup(ft, LegendreBasis(), n, 1, eps, 1, r0, wn, dn, taylor; asym_safety = 1e-4, istarget,
                 near32 = near32 === nothing ? eps >= 0.99e-6 : near32)
    wplan = slab ? SlabWide(g, eps, wide_orders(eps, s.porder, n, rc)..., tsw, wsw; charged, narrow = (dn, wn)) :
                   WidePlan(g, eps, wide_orders(eps, s.porder, n, rc)..., tsw, wsw)
    copysrc = g.tiled ? _copy_sources(ft, g) : zeros(Int, ft.nboxes)
    all(b -> origin[b] != _COPY || copysrc[b] > 0, 1:ft.nboxes) || error("copied box without a source")
    pot = Array{Float64}(undef, n, n, n, ft.nboxes)
    return PeriodicPlan(g, ft, n, eps, istarget, core, origin, sample_once, copysrc, s, wplan, wn, dn, Tuple(taylor), skip,
                        fvals, pot, true, tforest, tsample, targets, nothing)
end

# Tiled cells: for each box below a halo root, the box at the same place
# below its image root (0 if none, then the box samples the density).
function _copy_sources(ft::FlatTree, g::PeriodicGrid)
    nb = ft.nboxes
    root = zeros(Int, nb)
    for b in 1:nb
        root[b] = ft.parent[b] == 0 ? b : root[ft.parent[b]]
    end
    key(b, c) = (ft.level[b], ntuple(d -> round(Int, (c[d] - g.xelo[d]) / ft.boxsize[ft.level[b] + 1] - 0.5), 3))
    index = Dict(key(b, view(ft.centers, :, b)) => b for b in 1:nb if _image_root(g, root[b]) == root[b])
    src = zeros(Int, nb)
    for b in 1:nb
        r = root[b]; ri = _image_root(g, r)
        ri == r && continue
        sh = ft.centers[:, ri] .- ft.centers[:, r]
        src[b] = get(index, key(b, ft.centers[:, b] .+ sh), 0)
    end
    return src
end

"""
    periodic_solve!(plan, f; f_batch=nothing, timings=nothing) -> PeriodicResult

Solve on the plan's forest for the density `f` (resampled unless the plan's
samples are fresh). `timings` as for [`bdmk_periodic`](@ref).
"""
function periodic_solve!(plan::PeriodicPlan, f; f_batch = nothing, timings::Union{Nothing,Vector{Float64}} = nothing)
    f = _density(f, f_batch, plan.g.zorg)
    old_ftz = get_zero_subnormals()
    set_zero_subnormals(true)
    try
        return _periodic_solve!(plan, f, timings === nothing ? zeros(10) : timings)
    finally
        set_zero_subnormals(old_ftz)
    end
end

function _periodic_solve!(plan::PeriodicPlan, f, tm)
    fill!(tm, 0.0)
    g, ft, n, s, eps = plan.g, plan.ft, plan.n, plan.s, plan.eps
    t0 = time_ns()
    tsamp = plan.fresh ? 0.0 : _resample!(plan, f)
    plan.fresh = false
    fvals = plan.fvals
    tm[1] = tsamp
    tderive = (time_ns() - t0) / 1e9 - tsamp     # values derived from the samples

    t0 = time_ns()
    Sn = sum(plan.wn .* (pi .* plan.dn) .^ 1.5)
    set_sources!(s, plan.skip ? _source_boxes(ft, fvals, eps, Sn) : trues(ft.nboxes))
    # Source moments of one period: from the level-0 cubes when they tile the
    # cell (after the upward pass), else from a tree in fractional coordinates
    # refined like the forest (threshold of its level-0 test).
    Msrc = nothing
    if !g.tiled
        w3 = _tensor_weights(n)
        rint0 = sqrt(sum(b -> _box_l2sq(view(fvals, :, :, :, b), w3, g.rc^2 / 8), levelboxes(ft, 0)))
        if plan.sample_once
            ar = Arena()
            try
                Msrc = fractional_moments(_ForestSource(ft, g, fvals, n, ar), g, ft, n, plan.wplan.pf,
                                          eps * g.rc^-1.5 * rint0)
            finally
                release!(ar)
            end
        else
            Msrc = fractional_moments(f, g, ft, n, plan.wplan.pf, eps * g.rc^-1.5 * rint0)
        end
    end
    tm[2] = tderive + (time_ns() - t0) / 1e9
    try
        pot = plan.pot
        fq = _periodic_steps!(pot, s, g, fvals, plan.wplan, Msrc, tm)
        # 3D (fq the mean of f), neutralizing background: remove the k = 0 mode
        # of the real-space parts (narrow Gaussians and local correction), the
        # wide part has none. Slab (fq the charge Q): the gauge u + 2πQ|z|/A → 0
        # (`SlabWide`, the constant of the k̇ = 0 kernel)
        dc = g.slab ? fq * plan.wplan.qconst : fq * (Sn + plan.taylor[1])
        parallel_foreach(collect(1:ft.nboxes), fill(nothing, _nthreads())) do ibox, _
            (ft.nchild[ibox] == 0 && plan.istarget[ibox]) && (view(pot, :, :, :, ibox) .-= dc)
        end
        res = PeriodicResult(ft, g, fvals, pot, plan.istarget, n, nothing)
        plan.targets === nothing && return res
        T = plan.targets
        tp = zeros(size(T, 2))
        parallel_foreach(collect(axes(T, 2)), fill(nothing, _nthreads())) do j, _
            tp[j] = evaluate_periodic(res, view(T, :, j))
        end
        return PeriodicResult(ft, g, fvals, pot, plan.istarget, n, tp)
    finally
        release!(s.arena)
    end
end

# New density on the plan's forest; returns the time spent evaluating f.
# sample_once: f on the leaves of the adaptive refinement (and the level-0
# boxes, whose norm sets the wide-source threshold), then the other boxes
# from the polynomials as when the forest was built: children of the balance
# from their parents, lattice images by copy, non-core boxes from the core
# leaves. Otherwise f on every box that sampled it.
function _resample!(plan::PeriodicPlan, f)
    g, ft, n = plan.g, plan.ft, plan.n
    x = ortho1d(LegendreBasis(), n).x
    fvals = plan.fvals
    org = plan.origin
    nb = ft.nboxes
    nt = fill(nothing, _nthreads())
    copyboxes(bs) = parallel_foreach(bs, nt) do b, _
        view(fvals, :, :, :, b) .= view(fvals, :, :, :, plan.copysrc[b])
    end
    if !plan.sample_once
        t0 = time_ns()
        _sample_flat!(fvals, ft, g, f, _sampled_boxes(plan), x)
        tsamp = (time_ns() - t0) / 1e9
        copyboxes([b for b in 1:nb if plan.copysrc[b] > 0])
        _resample_noncore!(plan, [b for b in 1:nb if !plan.core[b]], x)
        return tsamp
    end
    t0 = time_ns()
    _sample_flat!(fvals, ft, g, f, _sampled_boxes(plan), x)
    tsamp = (time_ns() - t0) / 1e9
    fp = _FromParent(n)
    np = n^3
    ws = [(zeros(np), zeros(np)) for _ in 1:_nthreads()]
    derive(bs) = parallel_foreach(bs, ws) do b, (w1, w2)
        P = ft.parent[b]
        _parent_to_child!(view(fvals, :, :, :, b), view(fvals, :, :, :, P), fp, findfirst(==(b), view(ft.child, :, P)) - 1,
                          w1, w2)
    end
    lev(L, pred) = [b for b in levelboxes(ft, L) if pred(b)]
    copyboxes(lev(0, b -> org[b] == _COPY))
    for L in 1:ft.nlevels                   # parents before children, images after their sources
        derive(lev(L, b -> org[b] == _FROMPARENT && plan.core[b]))
        copyboxes(lev(L, b -> org[b] == _COPY))
    end
    _resample_noncore!(plan, [b for b in 1:nb if org[b] == _INTERP], x)
    for L in 1:ft.nlevels
        derive(lev(L, b -> org[b] == _FROMPARENT && !plan.core[b]))
    end
    return tsamp
end

# Boxes on which a solve samples f, in the order of `sample_points`.
function _sampled_boxes(plan::PeriodicPlan)
    ft, org = plan.ft, plan.origin
    nb = ft.nboxes
    plan.sample_once || return [b for b in 1:nb if plan.core[b] && plan.copysrc[b] == 0]
    return [b for b in 1:nb if org[b] == _SAMPLED &&
            (ft.nchild[b] == 0 || org[ft.child[1, b]] != _SAMPLED || (ft.level[b] == 0 && !plan.g.tiled))]
end

"""
    sample_points(plan) -> P (3 × N)

The points (caller's frame) at which a solve evaluates the density, box by box
on the Legendre grids. `periodic_solve!(plan, v)` with `v[j] = f(P[:, j])` is
the solve for `f`; any values on these points define a density on the plan's
forest (e.g. one known only through a basis, evaluated in batches).
"""
function sample_points(plan::PeriodicPlan)
    ft, g, n = plan.ft, plan.g, plan.n
    x = ortho1d(LegendreBasis(), n).x
    bs = _sampled_boxes(plan)
    np = n^3
    P = zeros(3, np * length(bs))
    R = g.Q
    parallel_foreach(collect(eachindex(bs)), fill(nothing, _nthreads())) do i, _
        b = bs[i]
        h = ft.boxsize[ft.level[b] + 1] / 2
        q = np * (i - 1)
        @inbounds for iz in 1:n, iy in 1:n, ix in 1:n
            q += 1
            y1 = ft.centers[1, b] + h * x[ix]; y2 = ft.centers[2, b] + h * x[iy]; y3 = ft.centers[3, b] + h * x[iz]
            for d in 1:3
                P[d, q] = R[d, 1] * y1 + R[d, 2] * y2 + R[d, 3] * y3
            end
            P[3, q] += g.zorg
        end
    end
    return P
end

# Density given by its values at `sample_points`.
struct _GivenValues{V<:AbstractVector{<:Real}}
    v::V
end

"""
    periodic_solve!(plan, v::AbstractVector; timings=nothing) -> PeriodicResult

Solve for the density with values `v` at `sample_points(plan)` (needs
`sample_once`, the default).
"""
function periodic_solve!(plan::PeriodicPlan, v::AbstractVector{<:Real}; timings::Union{Nothing,Vector{Float64}} = nothing)
    plan.sample_once || throw(ArgumentError("solving from given values needs a plan with sample_once"))
    length(v) == plan.n^3 * length(_sampled_boxes(plan)) ||
        throw(ArgumentError("need one value per sample point ($(plan.n^3 * length(_sampled_boxes(plan))))"))
    plan.fresh = false
    old_ftz = get_zero_subnormals()
    set_zero_subnormals(true)
    try
        return _periodic_solve!(plan, _GivenValues(v), timings === nothing ? zeros(10) : timings)
    finally
        set_zero_subnormals(old_ftz)
    end
end

# Non-core boxes `others` from the core leaves at the images of their nodes.
function _resample_noncore!(plan::PeriodicPlan, others, x)
    g, ft, n, fvals = plan.g, plan.ft, plan.n, plan.fvals
    isempty(others) && return plan
    ar = Arena()
    try
        leaves = [b for b in 1:ft.nboxes if plan.core[b] && ft.nchild[b] == 0]
        plan.iws === nothing && (plan.iws = _interp_workspaces(n))
        cf = _leaf_coefs(ar, n, [view(fvals, :, :, :, b) for b in leaves], plan.iws)
        coefs = Dict(zip(leaves, cf))
        locate = y -> _locate_leaf(ft, g, y)
        geom = L -> (view(ft.centers, :, L), ft.boxsize[ft.level[L] + 1] / 2)
        parallel_foreach(others, plan.iws) do b, w
            _interp_values!(view(fvals, :, :, :, b), view(ft.centers, :, b), ft.boxsize[ft.level[b] + 1] / 2, x, g, n,
                            locate, geom, L -> coefs[L], w)
        end
    finally
        release!(ar)
    end
    return plan
end

# Density samples on the grids of boxes `bs` of a flat forest.
function _sample_flat!(fvals, ft::FlatTree, g::PeriodicGrid, f, bs, x::Vector{Float64})
    n = length(x); np = n^3
    R = g.Q
    if f isa _GivenValues
        parallel_foreach(collect(eachindex(bs)), fill(nothing, _nthreads())) do i, _
            copyto!(view(fvals, :, :, :, bs[i]), 1, f.v, 1 + np * (i - 1), np)
        end
        return fvals
    end
    if f isa _BatchDensity
        parallel_foreach(bs, [zeros(3, np) for _ in 1:_nthreads()]) do b, pts
            h = ft.boxsize[ft.level[b] + 1] / 2
            i = 0
            for iz in 1:n, iy in 1:n, ix in 1:n
                i += 1
                y1 = ft.centers[1, b] + h * x[ix]; y2 = ft.centers[2, b] + h * x[iy]; y3 = ft.centers[3, b] + h * x[iz]
                for d in 1:3
                    pts[d, i] = R[d, 1] * y1 + R[d, 2] * y2 + R[d, 3] * y3
                end
            end
            f.f!(reshape(view(fvals, :, :, :, b), 1, np), pts)
        end
        return fvals
    end
    parallel_foreach(bs, [zeros(3) for _ in 1:_nthreads()]) do b, pt
        h = ft.boxsize[ft.level[b] + 1] / 2
        i = (b - 1) * np
        for iz in 1:n, iy in 1:n, ix in 1:n
            y1 = ft.centers[1, b] + h * x[ix]; y2 = ft.centers[2, b] + h * x[iy]; y3 = ft.centers[3, b] + h * x[iz]
            @inbounds for d in 1:3
                pt[d] = R[d, 1] * y1 + R[d, 2] * y2 + R[d, 3] * y3
            end
            @inbounds fvals[i += 1] = f(pt)
        end
    end
    return fvals
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

# Default level-0 side of a slab (U the rotated cell, dmin the in-plane lattice
# spacing): rectangular cells, the largest rc ≤ min(L₁, L₂, H) dividing both
# sides when one is not much smaller (the cubes then tile the cell);
# otherwise min(dmin, H).
function _slab_rc(U::AbstractMatrix, dmin::Float64)
    rmin = min(dmin, U[3, 3])
    if abs(U[1, 2]) <= 1e-12 * U[1, 1]
        L1, L2 = U[1, 1], U[2, 2]
        for m in max(1, ceil(Int, L1 / rmin * (1 - 1e-12))):floor(Int, 2 * L1 / rmin)
            rc = L1 / m
            abs(round(L2 / rc) * rc - L2) <= 1e-9 * L2 && return rc
        end
    end
    return rmin
end

# ∫|f| over a box (side h) from its node values; w3 the tensor Gauss-Legendre weights.
function _box_l1(fb, w3::Vector{Float64}, h::Float64)
    s = 0.0
    @inbounds for i in eachindex(w3)
        s += abs(fb[i]) * w3[i]
    end
    return s * (h / 2)^3
end

# The density as given (point by point or batch), in the solver frame: the
# slab frame starts at z = zorg of the caller's frame.
function _density(f, f_batch, zorg::Float64)
    if f_batch !== nothing
        zorg == 0 && return _BatchDensity(f_batch)
        return _BatchDensity((v, p) -> begin
            p3 = view(p, 3, :)
            p3 .+= zorg
            f_batch(v, p)
            p3 .-= zorg
            v
        end)
    end
    f === nothing && throw(ArgumentError("provide a density f or a batch density f_batch"))
    zorg == 0 && return f
    return _ZShifted(f, zorg)
end

struct _ZShifted{F}
    f::F
    z0::Float64
end
function (s::_ZShifted)(p)
    p[3] += s.z0
    v = s.f(p)
    p[3] -= s.z0
    return v
end

_tensor_weights(n::Int) = (w = ortho1d(LegendreBasis(), n).w; vec([w[i] * w[j] * w[k] for i in 1:n, j in 1:n, k in 1:n]))

# Mean of f over the cell from the root-cell moments (u-measure).
_cell_mean(P::WidePlan, M::Array{Float64,3}) =
    sum(M[1 + P.pf * s1, 1 + P.pf * s2, 1 + P.pf * s3] for s1 in 0:(P.mt[1] - 1), s2 in 0:(P.mt[2] - 1), s3 in 0:(P.mt[3] - 1)) / prod(P.mt)

function _periodic_steps!(pot4, s::DMKSetup, g::PeriodicGrid, fvals::Array{Float64,4}, wplan, Msrc, tm)
    ft, n, p = s.ft, s.n, s.porder
    nb = ft.nboxes
    fv = reshape(fvals, n, n, n, 1, nb)
    pot = reshape(pfill!(pot4), n, n, n, 1, nb)
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
    return fmean
end

# Central level-0 box (0-based central grid indices) of each level-0 box, or nothing.
function _central_index(g::PeriodicGrid, ibox::Int)
    m1, m2 = g.mle[1], g.mle[2]
    i = (ibox - 1) % m1 - g.halo[1]
    j = ((ibox - 1) ÷ m1) % m2 - g.halo[2]
    k = (ibox - 1) ÷ (m1 * m2) - g.halo[3]
    (0 <= i < g.mgrid[1] && 0 <= j < g.mgrid[2] && 0 <= k < g.mgrid[3]) || return nothing
    return (i, j, k)
end

# Root-cell moments -> modes -> u_wide Legendre coefficients on the central
# cubes, added to the proxy potentials of nonleaf boxes (spread by the
# downward pass) and evaluated directly on leaf boxes. For a tiled cell the
# moments are those of the central cubes (proxy charges of the upward pass).
# Returns the mean of f over the cell (slab: the total charge).
function _wide_inject!(pot, pp, pc, nodeidx, s::DMKSetup, g::PeriodicGrid, fv, P, Msrc)
    ft, p = s.ft, s.porder
    pe = P.pe
    boxes = [(ibox, c) for ibox in levelboxes(ft, 0) for c in (_central_index(g, ibox),) if c !== nothing]
    wks = make_workspaces(() -> (w1 = alloc(s.arena, max(p, s.n)^3), w2 = alloc(s.arena, max(p, s.n)^3),
                                  w3 = alloc(s.arena, p^3)))
    # (M and C are assigned once: the threaded closures capture them, and a
    # variable assigned more than once would be boxed)
    M = Msrc === nothing ? _level0_moments(pc, nodeidx, s, g, fv, P.pf, boxes, wks) : Msrc
    P isa SlabWide && !P.charged && _check_neutral(P, M, s, g, fv)
    C = _wide_coefs(P, M, s.arena)
    parallel_foreach([b for b in boxes if s.istarget[b[1]]], wks) do (ibox, (i, j, k)), w
        cb = reshape(view(w.w3, 1:p^3), p, p, p)
        fill!(cb, 0.0)
        _cube_coefs!(cb, C, P, i, j, k)
        if ft.nchild[ibox] == 0
            tp3!(view(pot, :, :, :, 1, ibox), cb, s.poteval, s.potevalT, s.potevalT, w.w1, w.w2, 1.0, true)
        else
            vec(view(pp, :, 1, 1, nodeidx[ibox])) .+= vec(cb)
        end
    end
    return P isa SlabWide ? _slab_charge(P, M) : _cell_mean(P, M)
end

# Legendre moments (u-measure) of the central level-0 cubes, from their proxy
# charges (nonleaf) or grid values (leaf).
function _level0_moments(pc, nodeidx, s::DMKSetup, g::PeriodicGrid, fv, pf::Int, boxes, wks)
    ft, p = s.ft, s.porder
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
    return M
end

_wide_coefs(P::SlabWide, M, ar::Arena) = slab_coefficients(P, M, ar)
_wide_coefs(P::WidePlan, M, ar::Arena) = wide_coefficients(P, wide_modes(P, M, ar), ar)

# Coefficients of the central cube (i, j, k): natural order C[r₁, r₂, r₃] (slab)
# or reversed Ct[r₃, r₂, r₁].
function _cube_coefs!(cb, C::Array{Float64,3}, P::SlabWide, i::Int, j::Int, k::Int)
    pe = P.pe
    if !P.tiled                                    # reversed order (oblique cells)
        @inbounds for l3 in 1:pe, l2 in 1:pe, l1 in 1:pe
            cb[l1, l2, l3] = C[pe * k + l3, pe * j + l2, pe * i + l1]
        end
        return cb
    end
    @inbounds for l3 in 1:pe, l2 in 1:pe, l1 in 1:pe
        cb[l1, l2, l3] = C[pe * i + l1, pe * j + l2, pe * k + l3]
    end
    return cb
end
function _cube_coefs!(cb, C::Array{Float64,3}, P::WidePlan, i::Int, j::Int, k::Int)
    pe = P.pe
    if P.iE !== nothing                            # natural order (FFT over the cubes)
        @inbounds for l3 in 1:pe, l2 in 1:pe, l1 in 1:pe
            cb[l1, l2, l3] = C[pe * i + l1, pe * j + l2, pe * k + l3]
        end
        return cb
    end
    @inbounds for l3 in 1:pe, l2 in 1:pe, l1 in 1:pe
        cb[l1, l2, l3] = C[pe * k + l3, pe * j + l2, pe * i + l1]
    end
    return cb
end

# A slab density must be neutral (the 2-periodic sum diverges otherwise).
function _check_neutral(P::SlabWide, M, s::DMKSetup, g::PeriodicGrid, fv)
    ft = s.ft
    qtot = _slab_charge(P, M)
    w3 = _tensor_weights(s.n)
    lv = [b for b in 1:ft.nboxes if ft.nchild[b] == 0 && !_inhalo(g, view(ft.centers, :, b))]
    l1 = zeros(length(lv))
    parallel_foreach(collect(eachindex(lv)), fill(nothing, _nthreads())) do i, _
        l1[i] = _box_l1(view(fv, :, :, :, 1, lv[i]), w3, ft.boxsize[ft.level[lv[i]] + 1])
    end
    qabs = sum(l1)
    abs(qtot) <= max(10 * s.eps, 1e-12) * qabs ||
        throw(ArgumentError("a quasi-2D density must be neutral (total charge $qtot, ∫|f| ≈ $qabs)"))
    return nothing
end

# Rotated-frame image in the cell of the point x (original frame).
function _wrap_into_cell(g::PeriodicGrid, x::AbstractVector)
    if g.slab                                    # periodic in x, y; z inside the slab
        z = x[3] - g.zorg
        0 <= z <= g.U[3, 3] || throw(ArgumentError("z = $(x[3]) is outside the slab"))
        y = transpose(g.Q) * [x[1], x[2], 0.0]
        t = [g.Uinv[1, 1] * y[1] + g.Uinv[1, 2] * y[2], g.Uinv[2, 2] * y[2]]
        t .-= floor.(t)
        return [g.U[1, 1] * t[1] + g.U[1, 2] * t[2], g.U[2, 2] * t[2], z]
    end
    t = g.Uinv * (transpose(g.Q) * x)            # fractional coordinates
    t .-= floor.(t)
    return g.U * t
end

"""
    evaluate_periodic(res::PeriodicResult, x::AbstractVector) -> Float64

Potential at any point `x` (wrapped into the cell), by interpolation on the
leaf containing it.
"""
function evaluate_periodic(res::PeriodicResult, x::AbstractVector)
    g, ft, n = res.grid, res.ft, res.norder
    y = _wrap_into_cell(g, x)
    ibox = _locate_leaf(ft, g, y)
    res.istarget[ibox] || throw(ArgumentError("the potential was not computed at $x (not a target)"))
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
