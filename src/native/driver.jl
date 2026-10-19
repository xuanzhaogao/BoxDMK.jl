# Pure-Julia box-DMK solver (port of Fortran `bdmk`, 3D).
#
# Internal box data is stored as (n, n, n, nd, nboxes) arrays in level order;
# proxy data as (porder^3, nd, [ncomp,] nboxes).

struct DMKSetup
    ft::FlatTree
    nd::Int
    n::Int
    porder::Int
    ifpgh::Int
    eps::Float64
    deltas::Vector{Float64}
    wdeltas::Vector{Float64}
    r0::Float64
    ipwaddr::Matrix{Int}        # (2, ilev - _NLEVSTART + 1), levels _NLEVSTART:nlevels+1
    idelta::Vector{Int}         # idelta[ilev+1]
    ifpwexp::BitVector
    nlistpw::Vector{Int}
    listpw::Matrix{Int}
    list1_ptr::Vector{Int}
    list1::Vector{Int}
    inp::Ortho1D                # input basis tables (order n)
    D1::Matrix{Float64}         # values -> first derivative values on [-1,1]
    D2::Matrix{Float64}
    D1T::Matrix{Float64}
    D2T::Matrix{Float64}
    den2pc::Matrix{Float64}     # values -> proxy coefficients (porder × n)
    den2pcT::Matrix{Float64}
    poteval::Matrix{Float64}    # proxy coefficients -> values (n × porder)
    potevalT::Matrix{Float64}
    p2c::NTuple{2,Matrix{Float64}}
    p2cT::NTuple{2,Matrix{Float64}}
    c2p::NTuple{2,Matrix{Float64}}
    c2pT::NTuple{2,Matrix{Float64}}
    # leaf children: c2p ∘ den2pc (porder × n) and poteval ∘ p2c (n × porder)
    c2pf::NTuple{2,Matrix{Float64}}
    c2pfT::NTuple{2,Matrix{Float64}}
    p2cf::NTuple{2,Matrix{Float64}}
    p2cfT::NTuple{2,Matrix{Float64}}
    prox::Ortho1D
    loctabs::Vector{Vector{Union{Nothing,LocTab1D}}}   # [ilev+1][id]
    pwlevels::Vector{Union{Nothing,PWLevel}}            # [ilev+1]
    taylor::NTuple{3,Float64}
    istarget::BitVector         # boxes whose potential is wanted (halo boxes are source-only)
    issource::BitVector         # boxes whose density matters (outgoing expansions, near field)
    need_mp::BitVector          # boxes whose outgoing expansion some target reads
    arena::Arena                # large buffers of this solve
end

_lvidx(ilev::Int) = ilev - _NLEVSTART + 1
ipwstart(s::DMKSetup, ilev::Int) = s.ipwaddr[1, _lvidx(ilev)]
ipwrange(s::DMKSetup, ilev::Int) = s.ipwaddr[1, _lvidx(ilev)]:s.ipwaddr[2, _lvidx(ilev)]

_nthreads() = Threads.nthreads()

function DMKSetup(tree::BoxTree, nd::Int, kernel::AbstractKernel, eps::Float64, ifpgh::Int)
    ft = FlatTree(tree)
    r0, wdeltas, deltas = sog_nodes(kernel, eps, ft.boxsize[1], ft.nlevels, tree.norder)
    taylor = taylor_coefficients(kernel, r0, wdeltas, deltas)
    return DMKSetup(ft, tree.basis, tree.norder, nd, eps, ifpgh, r0, wdeltas, deltas, taylor)
end

# Setup on a flat tree (or forest) for the SOG Σ wdeltas exp(-r²/deltas) with
# local correction `taylor`. `asym_safety` tightens the asymptotic-expansion
# criterion (4δ/bs²)³ <= eps * asym_safety; `istarget` marks the boxes whose
# potential is wanted (the others only act as sources); `issource` the boxes
# whose density is used in the far (plane-wave) and near fields; a nonleaf box
# should be a source if any leaf below it is. The upward pass covers all boxes.
function DMKSetup(ft::FlatTree, basis::AbstractBasis, n::Int, nd::Int, eps::Float64, ifpgh::Int,
                  r0::Float64, wdeltas::Vector{Float64}, deltas::Vector{Float64}, taylor;
                  asym_safety::Float64 = 1.0, istarget::BitVector = trues(ft.nboxes),
                  issource::BitVector = trues(ft.nboxes))
    nl = ft.nlevels
    bs0 = ft.boxsize[1]
    nde = length(deltas)

    nlv = _lvidx(nl + 1)
    ipw = zeros(Int, nlv)
    for i in 1:nde
        ipw[_lvidx(find_npwlevel(eps, nl, ft.boxsize, deltas[i]))] += 1
    end
    ipwaddr = zeros(Int, 2, nlv)
    istart = 1
    for l in 1:nlv
        ipwaddr[1, l] = istart
        ipwaddr[2, l] = istart + ipw[l] - 1
        istart += ipw[l]
    end

    idelta = zeros(Int, nl + 1)
    for ilev in 0:nl, k in nde:-1:1
        (deltas[k] * 4 / ft.boxsize[ilev + 1]^2)^3 <= eps * asym_safety && (idelta[ilev + 1] += 1)
    end

    ifpwexp = find_pwexp_boxes(ft)
    nlistpw, listpw = compute_listpw(ft)
    list1_ptr, list1 = compute_list1(ft)
    need_mp = falses(ft.nboxes)
    for ibox in 1:ft.nboxes
        (ifpwexp[ibox] && istarget[ibox]) || continue
        need_mp[ibox] = issource[ibox]
        for j in 1:nlistpw[ibox]
            jbox = listpw[j, ibox]
            issource[jbox] && (need_mp[jbox] = true)
        end
    end

    porder = select_porder(eps)
    prox = ortho1d(LegendreBasis(), porder)
    p2c, c2p = coefs_translation_matrices(LegendreBasis(), porder)
    inp = ortho1d(basis, n)

    p = zeros(n)
    vtmp = zeros(n, porder)
    for k in 1:porder
        polyvals!(p, basis, prox.x[k], n)
        vtmp[:, k] .= p .* prox.w[k]
    end
    den2pc = Matrix(transpose(vtmp * prox.V)) * inp.U
    pl = zeros(porder)
    poteval = zeros(n, porder)
    for i in 1:n
        legepols!(pl, inp.x[i], porder)
        poteval[i, :] .= pl
    end

    # Local tables: only levels with leaves, only the Gaussians they use.
    loctabs = [Union{Nothing,LocTab1D}[nothing for _ in 1:nde] for _ in 0:nl]
    q = LocTabQuad(basis, n)
    jobs = Tuple{Int,Int}[]
    for ilev in 0:nl
        any(ibox -> isleafbox(ft, ibox) && istarget[ibox], levelboxes(ft, ilev)) || continue
        for id in ipwaddr[1, _lvidx(ilev)]:(nde - idelta[ilev + 1])
            push!(jobs, (ilev, id))
        end
    end
    parallel_foreach(jobs, [nothing for _ in 1:_nthreads()]) do (ilev, id), _
        loctabs[ilev + 1][id] = mk_loctab_all(eps, basis, n, deltas[id], ft.boxsize[ilev + 1], q)
    end

    pwlevels = Union{Nothing,PWLevel}[nothing for _ in 0:nl]
    pwjobs = Int[]
    for ilev in 0:nl
        r = ipwaddr[1, _lvidx(ilev)]:ipwaddr[2, _lvidx(ilev)]
        isempty(r) && continue
        any(ibox -> ifpwexp[ibox], levelboxes(ft, ilev)) || continue
        push!(pwjobs, ilev)
    end
    parallel_foreach(pwjobs, [nothing for _ in 1:_nthreads()]) do ilev, _
        r = ipwaddr[1, _lvidx(ilev)]:ipwaddr[2, _lvidx(ilev)]
        pwlevels[ilev + 1] = PWLevel(eps, ilev, porder, ft.boxsize[ilev + 1], bs0,
                                     @view(deltas[r]), @view(wdeltas[r]), prox,
                                     (den2pc .* (ft.boxsize[ilev + 1] / 2), poteval))
    end

    D1 = inp.Vp * inp.U
    D2 = inp.Vpp * inp.U

    return DMKSetup(ft, nd, n, porder, ifpgh, eps, deltas, wdeltas, r0, ipwaddr, idelta, ifpwexp,
                    nlistpw, listpw, list1_ptr, list1, inp, D1, D2, Matrix(transpose(D1)), Matrix(transpose(D2)), den2pc, Matrix(transpose(den2pc)),
                    poteval, Matrix(transpose(poteval)), p2c, map(m -> Matrix(transpose(m)), p2c),
                    c2p, map(m -> Matrix(transpose(m)), c2p),
                    map(m -> m * den2pc, c2p), map(m -> Matrix(transpose(m * den2pc)), c2p),
                    map(m -> poteval * m, p2c), map(m -> Matrix(transpose(poteval * m)), p2c), prox,
                    loctabs, pwlevels, Tuple(taylor), istarget, issource, need_mp, Arena())
end

leafboxes(ft::FlatTree) = [ibox for ibox in 1:ft.nboxes if ft.nchild[ibox] == 0]
targetleaves(s::DMKSetup) = [ibox for ibox in 1:s.ft.nboxes if s.ft.nchild[ibox] == 0 && s.istarget[ibox]]

function _laplacian!(out, f, s::DMKSetup, sc2)
    n = s.n
    apply_dim!(out, f, s.D2, s.D2T, 1, n, sc2, false)
    apply_dim!(out, f, s.D2, s.D2T, 2, n, sc2, true)
    apply_dim!(out, f, s.D2, s.D2T, 3, n, sc2, true)
    return out
end

const _HESS_DIRS = ((1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3))

# STEPS 2 and 8 together: the local Taylor correction c0 f + c1 Δf + c2 Δ²f
# and the asymptotic expansion of the narrowest Gaussians, a0 f + a2 Δf + a4 Δ²f
# (gradient/hessian: first two terms of each), are both linear in the same
# derivatives of the density, so they are applied in one pass per leaf.
function step2_local_expansions!(pot, grad, hess, s::DMKSetup, fv)
    ft, n, nd, ifpgh = s.ft, s.n, s.nd, s.ifpgh
    c0, c1, c2 = s.taylor
    asym = [s.idelta[ilev + 1] > 0 ? _asym_coefs(s, s.idelta[ilev + 1]) : (0.0, 0.0, 0.0) for ilev in 0:ft.nlevels]
    wss = make_workspaces(() -> ntuple(_ -> alloc(s.arena, n, n, n), 5))
    parallel_foreach(targetleaves(s), wss) do ibox, (fl, fl2, gd, gld, tmp)
        ilev = ft.level[ibox]
        a0, a2, a4 = asym[ilev + 1]
        b0 = c0 + a0; b1 = c1 + a2; b2 = c2 + a4
        sc = 2.0 / ft.boxsize[ilev + 1]
        sc2 = sc * sc
        for ind in 1:nd
            f = view(fv, :, :, :, ind, ibox)
            _laplacian!(fl, f, s, sc2)
            _laplacian!(fl2, fl, s, sc2)
            pb = view(pot, :, :, :, ind, ibox)
            @inbounds @simd for i in eachindex(pb)
                pb[i] = b0 * f[i] + b1 * fl[i] + b2 * fl2[i]
            end
            if ifpgh >= 2
                for d in 1:3
                    apply_dim!(gd, f, s.D1, s.D1T, d, n, sc, false)
                    _laplacian!(gld, gd, s, sc2)
                    gb = view(grad, :, :, :, ind, d, ibox)
                    @inbounds @simd for i in eachindex(gb)
                        gb[i] = b0 * gd[i] + b1 * gld[i]
                    end
                end
            end
            if ifpgh >= 3
                for m in 1:6
                    d1, d2 = _HESS_DIRS[m]
                    if d1 == d2
                        apply_dim!(gd, f, s.D2, s.D2T, d1, n, sc2, false)
                    else
                        apply_dim!(tmp, f, s.D1, s.D1T, d1, n, sc, false)
                        apply_dim!(gd, tmp, s.D1, s.D1T, d2, n, sc, false)
                    end
                    _laplacian!(gld, gd, s, sc2)
                    hb = view(hess, :, :, :, ind, m, ibox)
                    @inbounds @simd for i in eachindex(hb)
                        hb[i] = b0 * gd[i] + b1 * gld[i]
                    end
                end
            end
        end
    end
    return nothing
end

_childsides(slot::Int) = (((slot - 1) & 1) + 1, (((slot - 1) >> 1) & 1) + 1, (((slot - 1) >> 2) & 1) + 1)

struct TPWork
    w1::Vector{Float64}
    w2::Vector{Float64}
end
TPWork(m::Int, ar::Arena) = TPWork(alloc(ar, m), alloc(ar, m))

# Proxy data is stored only for nonleaf boxes (and the root): leaves form and
# evaluate plane waves directly from/to their grid values, and their proxy
# charges/potentials are transient. nodeidx maps a box to its column.
function node_index(s::DMKSetup)
    idx = zeros(Int, s.ft.nboxes)
    k = 0
    for ibox in 1:s.ft.nboxes
        (s.ft.nchild[ibox] > 0 || ibox == 1) && (idx[ibox] = (k += 1))
    end
    return idx, k
end

# Box output component c: 1 = pot, 2:4 = grad, 5:10 = hess.
@inline function _outview(pot, grad, hess, ind, c, ibox)
    c == 1 && return view(pot, :, :, :, ind, ibox)
    c <= 4 && return view(grad, :, :, :, ind, c - 1, ibox)
    return view(hess, :, :, :, ind, c - 4, ibox)
end

struct ProxyOuts{A}
    pp::A
    ind::Int
    k::Int
end
@inline Base.getindex(o::ProxyOuts, m::Int) = view(o.pp, :, o.ind, m, o.k)

# Output m (1 = pot, 2:4 = grad, 5:10 = hess) of one box and density, as a
# pointer to its n³ contiguous values (type stable, unlike a tuple of views).
struct BoxOuts{A,G,H}
    pot::A
    grad::G
    hess::H
    ind::Int
    ibox::Int
end

@inline function outptr(o::BoxOuts, m::Int)
    n3 = size(o.pot, 1) * size(o.pot, 2) * size(o.pot, 3)
    nd = size(o.pot, 4)
    m == 1 && return pointer(o.pot) + 8 * n3 * ((o.ind - 1) + nd * (o.ibox - 1))
    m <= 4 && return pointer(o.grad) + 8 * n3 * ((o.ind - 1) + nd * ((m - 2) + 3 * (o.ibox - 1)))
    return pointer(o.hess) + 8 * n3 * ((o.ind - 1) + nd * ((m - 5) + 6 * (o.ibox - 1)))
end
@inline outptr(o::ProxyOuts, m::Int) = pointer(o.pp, 1 + size(o.pp, 1) * ((o.ind - 1) + size(o.pp, 2) * ((m - 1) + size(o.pp, 3) * (o.k - 1))))
@inline outptr(o::Tuple{AbstractArray{Float64}}, m::Int) = pointer(o[1])

function step3_upward!(pc, nodeidx, s::DMKSetup, fv, wss::Vector{TPWork})
    ft, n, nd, p = s.ft, s.n, s.nd, s.porder
    for ilev in ft.nlevels:-1:0
        boxes = [ibox for ibox in levelboxes(ft, ilev) if nodeidx[ibox] > 0]
        sc3 = ilev < ft.nlevels ? (ft.boxsize[ilev + 2] / 2)^3 : 0.0
        parallel_foreach(boxes, wss) do ibox, w
            k = nodeidx[ibox]
            for ind in 1:nd
                out = view(pc, :, ind, k)
                if ft.nchild[ibox] == 0
                    _leaf_charge!(out, s, fv, ind, ibox, w)
                else
                    # nonleaf children from their proxy charges, leaf children
                    # straight from their grid values
                    qn = slot -> (c = ft.child[slot, ibox]; (c == 0 || nodeidx[c] == 0) ? nothing : view(pc, :, ind, nodeidx[c]))
                    ql = slot -> (c = ft.child[slot, ibox]; (c == 0 || nodeidx[c] > 0) ? nothing : view(fv, :, :, :, ind, c))
                    children2parent!(out, qn, s.c2p, s.c2pT, s.c2pT, p, p, w.w1, w.w2, false)
                    children2parent!(out, ql, s.c2pf, s.c2pfT, s.c2pfT, n, p, w.w1, w.w2, true, sc3)
                end
            end
        end
    end
    return pc
end

@inline function _leaf_charge!(out, s::DMKSetup, fv, ind::Int, ibox::Int, w::TPWork)
    sc3 = (s.ft.boxsize[s.ft.level[ibox] + 1] / 2)^3
    tp3!(out, view(fv, :, :, :, ind, ibox), s.den2pc, s.den2pcT, s.den2pcT, w.w1, w.w2, sc3, false)
end

# proxy potential (component c) of a leaf -> values on its grid
@inline function _proxy_eval!(out, phi, s::DMKSetup, w::TPWork)
    tp3!(out, phi, s.poteval, s.potevalT, s.potevalT, w.w1, w.w2, 1.0, true)
end

function steps456_planewave!(pot, grad, hess, pp, pc, nodeidx, s::DMKSetup, fv, tpw::Vector{TPWork}, tm::Vector{Float64})
    ft, nd, p, ifpgh, eps = s.ft, s.nd, s.porder, s.ifpgh, s.eps
    ncomb = _ncombos(ifpgh)
    bs0 = ft.boxsize[1]
    # Fat Gaussians (coarser than the root box) act on the root only.
    for ilev in _NLEVSTART:-1
        r = ipwrange(s, ilev)
        isempty(r) && continue
        L = PWLevel(eps, ilev, p, bs0, bs0, @view(s.deltas[r]), @view(s.wdeltas[r]), s.prox)
        w = PWWork(L.npw, max(p, s.n), s.arena, ifpgh, true)
        for ind in 1:nd
            proxy2pw!(w.locr, w.loci, view(pc, :, ind, nodeidx[1]), L, L.prox, w, 0, PWLayout(L), true)
            pw2proxy!(ProxyOuts(pp, ind, nodeidx[1]), w.locr, w.loci, L, L.prox, ifpgh, w)
        end
    end

    mpidx = zeros(Int, ft.nboxes)
    # Workspaces and expansion storage are sized once for the busiest level
    # (all levels >= 0 share the same number of plane waves).
    npw = pw_terms(eps, 0)
    nexp = npw * npw * (npw ÷ 2)
    maxboxes = maximum(ilev -> max(count(ibox -> s.need_mp[ibox], levelboxes(ft, ilev)),
                                   count(ibox -> s.ifpwexp[ibox] && s.istarget[ibox], levelboxes(ft, ilev))), 0:ft.nlevels)
    ncols = npw * (npw ÷ 2)
    cb = min(ncols, cld(2048, npw))           # ~16 KB of a box's expansion per block
    nblk = cld(ncols, cb)
    mpr = alloc(s.arena, npw, cb, nd, maxboxes, nblk)
    mpi = alloc(s.arena, npw, cb, nd, maxboxes, nblk)
    locr = alloc(s.arena, npw, cb, nd, maxboxes, nblk)
    loci = alloc(s.arena, npw, cb, nd, maxboxes, nblk)
    lay = PWLayout(npw, cb, npw * cb * nd * maxboxes)
    wss = make_workspaces(() -> PWWork(npw, max(p, s.n), s.arena, ifpgh))
    accs = [zeros(2 * 64) for _ in 1:_nthreads()]
    tidx = zeros(Int, ft.nboxes)
    for ilev in 0:ft.nlevels
        # sources: expansions some target reads; targets: wanted boxes
        boxes = [ibox for ibox in levelboxes(ft, ilev) if s.need_mp[ibox]]
        tboxes = [ibox for ibox in levelboxes(ft, ilev) if s.ifpwexp[ibox] && s.istarget[ibox]]
        isempty(tboxes) && continue
        L = s.pwlevels[ilev + 1]
        if L !== nothing
            nbl = length(tboxes)
            for (k, ibox) in enumerate(boxes)
                mpidx[ibox] = k
            end
            for (k, ibox) in enumerate(tboxes)
                tidx[ibox] = k
            end
            @assert L.nexp == nexp
            # outgoing expansions, premultiplied by the kernel's Fourier transform
            mpr1 = vec(mpr); mpi1 = vec(mpi); locr1 = vec(locr); loci1 = vec(loci)
            t0 = time_ns()
            parallel_foreach(boxes, wss) do ibox, w
                k = mpidx[ibox]
                for ind in 1:nd
                    off0 = npw * cb * ((ind - 1) + nd * (k - 1))
                    if ft.nchild[ibox] == 0
                        proxy2pw!(mpr1, mpi1, view(fv, :, :, :, ind, ibox), L, L.leaf, w, off0, lay, true)
                    else
                        proxy2pw!(mpr1, mpi1, view(pc, :, ind, nodeidx[ibox]), L, L.prox, w, off0, lay, true)
                    end
                end
            end
            t1 = time_ns(); tm[5] += (t1 - t0) / 1e9
            lst = _m2l_list(s, tboxes, mpidx, ft.boxsize[ilev + 1])
            nchunk = min(ncols, 8 * _nthreads())
            ranges = [(1 + div((i - 1) * ncols, nchunk)):div(i * ncols, nchunk) for i in 1:nchunk]
            parallel_foreach(ranges, accs) do cols, acc
                m2l_columns!(locr, loci, mpr, mpi, lst, L, nd, nbl, cols, acc)
            end
            t2 = time_ns(); tm[6] += (t2 - t1) / 1e9
            parallel_foreach(tboxes, wss) do ibox, w
                k = tidx[ibox]
                for ind in 1:nd
                    off0 = npw * cb * ((ind - 1) + nd * (k - 1))
                    if ft.nchild[ibox] == 0
                        pw2proxy!(BoxOuts(pot, grad, hess, ind, ibox), locr1, loci1, L, L.leaf, ifpgh, w, off0, lay)
                    else
                        pw2proxy!(ProxyOuts(pp, ind, nodeidx[ibox]), locr1, loci1, L, L.prox, ifpgh, w, off0, lay)
                    end
                end
            end
            tm[8] += (time_ns() - t2) / 1e9
        end
        # Downward pass: parent proxy potential -> children. Leaf children get
        # their (now complete) proxy potential evaluated on their grid directly.
        t3 = time_ns()
        parents = [ibox for ibox in tboxes if ft.nchild[ibox] > 0]
        parallel_foreach(parents, tpw) do ibox, w
            for ind in 1:nd, c in 1:ncomb
                phi = view(pp, :, ind, c, nodeidx[ibox])
                on = slot -> (ch = ft.child[slot, ibox]; (ch == 0 || nodeidx[ch] == 0) ? nothing : view(pp, :, ind, c, nodeidx[ch]))
                ol = slot -> (ch = ft.child[slot, ibox]; (ch == 0 || nodeidx[ch] > 0) ? nothing : outptr(BoxOuts(pot, grad, hess, ind, ch), c))
                parent2children!(on, phi, s.p2c, s.p2cT, s.p2cT, p, p, w.w1, w.w2)
                parent2children!(ol, phi, s.p2cf, s.p2cfT, s.p2cfT, p, s.n, w.w1, w.w2)
            end
        end
        tm[9] += (time_ns() - t3) / 1e9
    end
    return nothing
end

function _m2l_list(s::DMKSetup, tboxes, mpidx, xmin)
    ft = s.ft
    srcs = [NTuple{4,Int}[] for _ in tboxes]
    for (t, ibox) in enumerate(tboxes)
        s.need_mp[ibox] && push!(srcs[t], (mpidx[ibox], 0, 0, 0))
        for j in 1:s.nlistpw[ibox]
            jbox = s.listpw[j, ibox]
            s.need_mp[jbox] || continue
            dx = round(Int, (ft.centers[1, ibox] - ft.centers[1, jbox]) / xmin)
            dy = round(Int, (ft.centers[2, ibox] - ft.centers[2, jbox]) / xmin)
            dz = round(Int, (ft.centers[3, ibox] - ft.centers[3, jbox]) / xmin)
            push!(srcs[t], (mpidx[jbox], dx, dy, dz))
        end
    end
    return M2LList(srcs)
end

@inline _loctab_slot(t::Float64, c::Float64, bs::Float64) = round(Int, (t - c) / bs) + NLOCTAB + 1

function step7_local!(pot, grad, hess, s::DMKSetup, fv)
    ft, n = s.ft, s.n
    wss = make_workspaces(() -> (LocalWork(n, s.arena), alloc(s.arena, n^3), alloc(s.arena, n^3), zeros(Int, 27), Int[]))
    parallel_foreach(targetleaves(s), wss) do ibox, ws
        if s.ifpgh == 1
            _local_pot_box!(pot, s, fv, ibox, ws)
        else
            _local_pgh_box!(pot, grad, hess, s, fv, ibox, ws[1])
        end
    end
    return nothing
end

@inline _ldelta(s::DMKSetup, ilev::Int) = length(s.deltas) - s.idelta[ilev + 1]

@inline function _local_jstart(s::DMKSetup, ibox::Int, jbox::Int, ilev::Int, jlev::Int)
    if s.ifpwexp[jbox] && jbox == ibox
        return ipwstart(s, jlev + 1)
    elseif jlev < ilev
        return ipwstart(s, ilev)
    else
        return ipwstart(s, jlev)
    end
end

@inline _slot_klen(T::LocTab1D, slot::Int, n::Int) = max(0, T.ind[2, n + 1, slot] - T.ind[1, n + 1, slot] + 1)

# Fortran evaluates a (source, Gaussian) pair densely iff every 1D pattern is
# nonempty and their lengths sum to at least n; otherwise sparsely (or not at all).
@inline function _pair_mode(T::LocTab1D, sx::Int, sy::Int, sz::Int, n::Int)
    nx = _slot_klen(T, sx, n); ny = _slot_klen(T, sy, n); nz = _slot_klen(T, sz, n)
    (nx == 0 || ny == 0 || nz == 0) && return 0
    return nx + ny + nz >= n ? 2 : 1
end

# Potential-only near field of one leaf. Same-level dense interactions are
# evaluated target-centrically: pot += w Σ_dz Tz(Σ_dy Ty(Σ_dx Tx f_s)), which
# shares the y and z stages among the sources; everything else per pair.
function _local_pot_box!(pot, s::DMKSetup, fv, ibox::Int, (lw, Abuf, Bbuf, stencil, others))
    ft, n, nd = s.ft, s.n, s.nd
    ilev = ft.level[ibox]
    iend = _ldelta(s, ilev)
    tabsl = s.loctabs[ilev + 1]
    bsl = ft.boxsize[ilev + 1]
    fill!(stencil, 0)
    empty!(others)
    for jj in s.list1_ptr[ibox]:(s.list1_ptr[ibox + 1] - 1)
        jbox = s.list1[jj]
        s.issource[jbox] || continue
        if ft.level[jbox] == ilev
            dx = round(Int, (ft.centers[1, jbox] - ft.centers[1, ibox]) / bsl)
            dy = round(Int, (ft.centers[2, jbox] - ft.centers[2, ibox]) / bsl)
            dz = round(Int, (ft.centers[3, jbox] - ft.centers[3, ibox]) / bsl)
            stencil[1 + (dx + 1) + 3 * (dy + 1) + 9 * (dz + 1)] = jbox
        else
            push!(others, jbox)
        end
    end
    selfpw_end = s.ifpwexp[ibox] ? ipwstart(s, ilev + 1) : 0
    nn = n * n
    for id in ipwstart(s, ilev):iend
        T = tabsl[id]::LocTab1D
        wd = s.wdeltas[id]
        for ind in 1:nd
            potb = view(pot, :, :, :, ind, ibox)
            GC.@preserve potb fv T Abuf Bbuf begin
                pA = pointer(Abuf); pB = pointer(Bbuf)
                for kz in 1:3
                    # slot of the table: source at -d relative offset -> slot 7 - 4d
                    sz = NLOCTAB + 1 - 4 * (kz - 2)
                    haveB = false
                    for ky in 1:3
                        sy = NLOCTAB + 1 - 4 * (ky - 2)
                        haveA = false
                        for kx in 1:3
                            jbox = stencil[kx + 3 * (ky - 1) + 9 * (kz - 1)]
                            jbox == 0 && continue
                            (jbox == ibox && id < selfpw_end) && continue
                            sx = NLOCTAB + 1 - 4 * (kx - 2)
                            mode = _pair_mode(T, sx, sy, sz, n)
                            mode == 0 && continue
                            if mode == 1
                                local_sparse!((potb,), view(fv, :, :, :, ind, jbox), wd, T, sx, sy, sz, n, 1, lw)
                                continue
                            end
                            # stage x: A += Tx' f
                            gemm!(pA, n, pointer(T.tabT, (sx - 1) * nn + 1), n,
                                  pointer(fv, 1 + nn * n * ((ind - 1) + nd * (jbox - 1))), n,
                                  n, nn, n, 1.0, haveA)
                            haveA = true
                        end
                        haveA || continue
                        # stage y: B[:, :, k] += A[:, :, k] * Ty
                        gemm_batch!(pB, n, nn, pA, n, nn, pointer(T.tab, (sy - 1) * nn + 1), n,
                                    n, n, n, n, 1.0, haveB)
                        haveB = true
                    end
                    haveB || continue
                    # stage z: pot += w B * Tz
                    gemm!(pointer(potb), nn, pB, nn, pointer(T.tab, (sz - 1) * nn + 1), n, nn, n, n, wd, true)
                end
            end
        end
    end
    # Coarser/finer neighbours and Gaussians not handled above, pair by pair.
    for jbox in others
        jlev = ft.level[jbox]
        bs = ft.boxsize[jlev + 1] / 4
        sx = _loctab_slot(ft.centers[1, ibox], ft.centers[1, jbox], bs)
        sy = _loctab_slot(ft.centers[2, ibox], ft.centers[2, jbox], bs)
        sz = _loctab_slot(ft.centers[3, ibox], ft.centers[3, jbox], bs)
        for id in _local_jstart(s, ibox, jbox, ilev, jlev):iend
            T = tabsl[id]::LocTab1D
            mode = _pair_mode(T, sx, sy, sz, n)
            mode == 0 && continue
            for ind in 1:nd
                potb = view(pot, :, :, :, ind, ibox)
                f = view(fv, :, :, :, ind, jbox)
                if mode == 2
                    tp3!(potb, f, view(T.tabT, :, :, sx), view(T.tab, :, :, sy), view(T.tab, :, :, sz),
                         lw.w1, lw.w2, s.wdeltas[id], true)
                else
                    local_sparse!((potb,), f, s.wdeltas[id], T, sx, sy, sz, n, 1, lw)
                end
            end
        end
    end
    return nothing
end

function _local_pgh_box!(pot, grad, hess, s::DMKSetup, fv, ibox::Int, lw::LocalWork)
    ft, n, nd, ifpgh = s.ft, s.n, s.nd, s.ifpgh
    ncomb = _ncombos(ifpgh)
    ilev = ft.level[ibox]
    iend = _ldelta(s, ilev)
    tabsl = s.loctabs[ilev + 1]
    for jj in s.list1_ptr[ibox]:(s.list1_ptr[ibox + 1] - 1)
        jbox = s.list1[jj]
        s.issource[jbox] || continue
        jlev = ft.level[jbox]
        bs = ft.boxsize[jlev + 1] / 4
        sx = _loctab_slot(ft.centers[1, ibox], ft.centers[1, jbox], bs)
        sy = _loctab_slot(ft.centers[2, ibox], ft.centers[2, jbox], bs)
        sz = _loctab_slot(ft.centers[3, ibox], ft.centers[3, jbox], bs)
        for id in _local_jstart(s, ibox, jbox, ilev, jlev):iend
            T = tabsl[id]::LocTab1D
            for ind in 1:nd
                local_sparse!(BoxOuts(pot, grad, hess, ind, ibox), view(fv, :, :, :, ind, jbox),
                              s.wdeltas[id], T, sx, sy, sz, n, ncomb, lw)
            end
        end
    end
    return nothing
end


function _asym_coefs(s::DMKSetup, nas::Int)
    nde = length(s.deltas)
    sqrtpi = sqrt(pi)
    c0 = 0.0; c2 = 0.0; c4 = 0.0
    for i in 1:nas
        delta = s.deltas[nde - i + 1]
        wd = s.wdeltas[nde - i + 1]
        d0 = (sqrtpi * sqrt(delta))^3
        c0 += d0 * wd
        c2 += d0 * delta / 4 * wd
        c4 += d0 * delta^2 / 32 * wd
    end
    return c0, c2, c4
end

# Leaves with stored proxy potential (only a leaf root): proxy -> grid values.
function step9_proxy2pot!(pot, grad, hess, pp, nodeidx, s::DMKSetup, wss::Vector{TPWork})
    ft, n, nd, p = s.ft, s.n, s.nd, s.porder
    ncomb = _ncombos(s.ifpgh)
    leaves = [ibox for ibox in 1:ft.nboxes if ft.nchild[ibox] == 0 && nodeidx[ibox] > 0]
    parallel_foreach(leaves, wss) do ibox, w
        for ind in 1:nd, c in 1:ncomb
            _proxy_eval!(_outview(pot, grad, hess, ind, c, ibox), view(pp, :, ind, c, nodeidx[ibox]), s, w)
        end
    end
    return nothing
end

# Leaf containing `x` (descending from the root by coordinate comparison).
function _locate_leaf_flat(ft::FlatTree, x::AbstractVector{Float64})
    ibox = 1
    while ft.nchild[ibox] > 0
        slot = 1
        x[1] > ft.centers[1, ibox] && (slot += 1)
        x[2] > ft.centers[2, ibox] && (slot += 2)
        x[3] > ft.centers[3, ibox] && (slot += 4)
        c = ft.child[slot, ibox]
        c == 0 && break
        ibox = c
    end
    return ibox
end

# Interpolation weights l_i(t) = Σ_m U[m, i] P_m(t) at t ∈ [-1, 1].
function _interp_weights!(l, pv, basis, U, t, n)
    polyvals!(pv, basis, t, n)
    for i in 1:n
        acc = 0.0
        for m in 1:n
            acc += U[m, i] * pv[m]
        end
        l[i] = acc
    end
    return l
end

function evaluate_targets_native(s::DMKSetup, basis, targets::Matrix{Float64}, arrs...)
    ft, n, nd = s.ft, s.n, s.nd
    nt = size(targets, 2)
    outs = map(arrs) do a
        ncomp = ndims(a) == 6 ? size(a, 5) : 1
        zeros(nd, ncomp, nt)
    end
    wss = [(zeros(n), zeros(n), zeros(n), zeros(n)) for _ in 1:_nthreads()]
    parallel_foreach(1:nt, wss) do it, (lx, ly, lz, pv)
        x = view(targets, :, it)
        ibox = _locate_leaf_flat(ft, x)
        sc = 2.0 / ft.boxsize[ft.level[ibox] + 1]
        _interp_weights!(lx, pv, basis, s.inp.U, (x[1] - ft.centers[1, ibox]) * sc, n)
        _interp_weights!(ly, pv, basis, s.inp.U, (x[2] - ft.centers[2, ibox]) * sc, n)
        _interp_weights!(lz, pv, basis, s.inp.U, (x[3] - ft.centers[3, ibox]) * sc, n)
        for (a, o) in zip(arrs, outs)
            ncomp = size(o, 2)
            for c in 1:ncomp, ind in 1:nd
                v = ndims(a) == 6 ? view(a, :, :, :, ind, c, ibox) : view(a, :, :, :, ind, ibox)
                acc = 0.0
                @inbounds for k in 1:n
                    acck = 0.0
                    for j in 1:n
                        accj = 0.0
                        @simd for i in 1:n
                            accj += lx[i] * v[i, j, k]
                        end
                        acck += ly[j] * accj
                    end
                    acc += lz[k] * acck
                end
                o[ind, c, it] = acc
            end
        end
    end
    return outs
end

"""
    bdmk_native(tree, fvals, kernel; eps=1e-6, grad=false, hess=false, targets=nothing, timings=nothing)

Pure-Julia box-DMK volume potential solver (3D). `timings`, if given a
`Vector{Float64}` of length ≥ 10, receives wall times (s): [1] precomputation,
[2] local Taylor + asymptotic expansions, [3] proxy upward pass, [4] plane-wave
part (total), of which [5] formation, [6] translation, [8] evaluation,
[9] downward pass; [7] near field, [10] output and targets.
"""
function bdmk_native(tree::BoxTree, fvals::AbstractArray{<:Real,3}, kernel::AbstractKernel;
                     eps = 1e-6, grad::Bool = false, hess::Bool = false, targets = nothing,
                     timings::Union{Nothing,Vector{Float64}} = nothing)
    old_ftz = get_zero_subnormals()
    set_zero_subnormals(true)
    try
        return _bdmk_native(tree, fvals, kernel, Float64(eps), grad, hess, targets, timings)
    finally
        set_zero_subnormals(old_ftz)
    end
end

# Box data live in the solve's arena except what is returned to the caller.
_boxarray(s::DMKSetup, returned::Bool, dims::Int...) = returned ? Array{Float64}(undef, dims...) : alloc(s.arena, dims...)

function _bdmk_native(tree::BoxTree, fvals::AbstractArray{<:Real,3}, kernel::AbstractKernel, eps::Float64,
                      grad::Bool, hess::Bool, targets, timings)
    nd, np = _check_solver_inputs(tree, fvals)
    ifpgh = hess ? 3 : (grad ? 2 : 1)
    tm = timings === nothing ? zeros(10) : timings
    fill!(tm, 0.0)

    t0 = time_ns()
    s = DMKSetup(tree, nd, kernel, eps, ifpgh)
    try
        return _bdmk_steps(s, tree, fvals, nd, np, ifpgh, targets, tm, t0)
    finally
        release!(s.arena)
    end
end

function _bdmk_steps(s::DMKSetup, tree::BoxTree, fvals, nd::Int, np::Int, ifpgh::Int, targets, tm, t0)
    ft, n, p = s.ft, s.n, s.porder
    nb = ft.nboxes
    # Box data are (n, n, n, nd, nboxes); for nd = 1 that is the layout of fvals.
    fv = if nd == 1 && fvals isa Array{Float64,3}
        reshape(fvals, n, n, n, 1, nb)
    else
        a = alloc(s.arena, n, n, n, nd, nb)
        a3 = reshape(a, np, nd, nb)
        parallel_foreach(1:nb, fill(nothing, _nthreads())) do ibox, _
            for ind in 1:nd
                @inbounds for j in 1:np
                    a3[j, ind, ibox] = fvals[ind, j, ibox]
                end
            end
        end
        a
    end
    ng = ifpgh >= 2 ? nb : 0
    nh = ifpgh >= 3 ? nb : 0
    # STEP 2 overwrites every leaf; only nonleaf boxes need zeros.
    pot = _boxarray(s, nd == 1, n, n, n, nd, nb)
    gradv = _boxarray(s, false, n, n, n, nd, 3, ng)
    hessv = _boxarray(s, false, n, n, n, nd, 6, nh)
    for ibox in 1:nb
        ft.nchild[ibox] == 0 && continue
        fill!(view(pot, :, :, :, :, ibox), 0.0)
        ng > 0 && fill!(view(gradv, :, :, :, :, :, ibox), 0.0)
        nh > 0 && fill!(view(hessv, :, :, :, :, :, ibox), 0.0)
    end
    tm[1] = (time_ns() - t0) / 1e9

    t0 = time_ns()
    step2_local_expansions!(pot, gradv, hessv, s, fv)
    tm[2] = (time_ns() - t0) / 1e9

    t0 = time_ns()
    nodeidx, nnode = node_index(s)
    pc = alloc(s.arena, p^3, nd, nnode)
    tpw = make_workspaces(() -> TPWork(max(p, n)^3, s.arena))
    step3_upward!(pc, nodeidx, s, fv, tpw)
    tm[3] = (time_ns() - t0) / 1e9

    t0 = time_ns()
    pp = pfill!(alloc(s.arena, p^3, nd, _ncombos(ifpgh), nnode))
    steps456_planewave!(pot, gradv, hessv, pp, pc, nodeidx, s, fv, tpw, tm)
    tm[4] = (time_ns() - t0) / 1e9

    t0 = time_ns()
    step7_local!(pot, gradv, hessv, s, fv)
    tm[7] = (time_ns() - t0) / 1e9


    step9_proxy2pot!(pot, gradv, hessv, pp, nodeidx, s, tpw)

    t0 = time_ns()
    pot_out = _unflatten(pot, ft, nd, np)
    grad_out = ifpgh >= 2 ? _unflatten(gradv, ft, nd, np) : nothing
    hess_out = ifpgh >= 3 ? _unflatten(hessv, ft, nd, np) : nothing
    target_pot = target_grad = target_hess = nothing
    if targets !== nothing
        tp = _normalize_targets(targets, tree)
        arrs = ifpgh == 1 ? (pot,) : (ifpgh == 2 ? (pot, gradv) : (pot, gradv, hessv))
        outs = evaluate_targets_native(s, tree.basis, tp, arrs...)
        target_pot = outs[1][:, 1, :]
        ifpgh >= 2 && (target_grad = outs[2])
        ifpgh >= 3 && (target_hess = outs[3])
    end
    tm[10] = (time_ns() - t0) / 1e9
    return SolverResult(pot_out, grad_out, hess_out, target_pot, target_grad, target_hess)
end

function _unflatten(a::Array{Float64,5}, ft::FlatTree, nd::Int, np::Int)
    nd == 1 && return reshape(a, 1, np, ft.nboxes)
    out = Array{Float64}(undef, nd, np, ft.nboxes)
    a3 = reshape(a, np, nd, ft.nboxes)
    parallel_foreach(1:ft.nboxes, fill(nothing, _nthreads())) do ibox, _
        for ind in 1:nd
            @inbounds for j in 1:np
                out[ind, j, ibox] = a3[j, ind, ibox]
            end
        end
    end
    return out
end

function _unflatten(a::Array{Float64,6}, ft::FlatTree, nd::Int, np::Int)
    nc = size(a, 5)
    out = Array{Float64}(undef, nd, nc, np, ft.nboxes)
    a4 = reshape(a, np, nd, nc, ft.nboxes)
    parallel_foreach(1:ft.nboxes, fill(nothing, _nthreads())) do ibox, _
        for c in 1:nc, ind in 1:nd
            @inbounds for j in 1:np
                out[ind, c, j, ibox] = a4[j, ind, c, ibox]
            end
        end
    end
    return out
end
