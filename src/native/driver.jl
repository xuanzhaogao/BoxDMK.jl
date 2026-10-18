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
    den2pc::Matrix{Float64}     # values -> proxy coefficients (porder × n)
    den2pcT::Matrix{Float64}
    poteval::Matrix{Float64}    # proxy coefficients -> values (n × porder)
    potevalT::Matrix{Float64}
    p2c::NTuple{2,Matrix{Float64}}
    p2cT::NTuple{2,Matrix{Float64}}
    c2p::NTuple{2,Matrix{Float64}}
    c2pT::NTuple{2,Matrix{Float64}}
    prox::Ortho1D
    loctabs::Vector{Vector{Union{Nothing,LocTab1D}}}   # [ilev+1][id]
    pwlevels::Vector{Union{Nothing,PWLevel}}            # [ilev+1]
    taylor::NTuple{3,Float64}
end

_lvidx(ilev::Int) = ilev - _NLEVSTART + 1
ipwstart(s::DMKSetup, ilev::Int) = s.ipwaddr[1, _lvidx(ilev)]
ipwrange(s::DMKSetup, ilev::Int) = s.ipwaddr[1, _lvidx(ilev)]:s.ipwaddr[2, _lvidx(ilev)]

_nthreads() = Threads.nthreads()

function DMKSetup(tree::BoxTree, nd::Int, kernel::AbstractKernel, eps::Float64, ifpgh::Int)
    ft = FlatTree(tree)
    n = tree.norder
    basis = tree.basis
    nl = ft.nlevels
    bs0 = ft.boxsize[1]

    r0, wdeltas, deltas = sog_nodes(kernel, eps, bs0, nl, n)
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
        (deltas[k] * 4 / ft.boxsize[ilev + 1]^2)^3 <= eps && (idelta[ilev + 1] += 1)
    end

    ifpwexp = find_pwexp_boxes(ft)
    nlistpw, listpw = compute_listpw(ft)
    list1_ptr, list1 = compute_list1(ft)

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
        any(ibox -> isleafbox(ft, ibox), levelboxes(ft, ilev)) || continue
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
                                     @view(deltas[r]), @view(wdeltas[r]), prox)
    end

    taylor = taylor_coefficients(kernel, r0, wdeltas, deltas)
    D1 = inp.Vp * inp.U
    D2 = inp.Vpp * inp.U

    return DMKSetup(ft, nd, n, porder, ifpgh, eps, deltas, wdeltas, r0, ipwaddr, idelta, ifpwexp,
                    nlistpw, listpw, list1_ptr, list1, inp, D1, D2, den2pc, Matrix(transpose(den2pc)),
                    poteval, Matrix(transpose(poteval)), p2c, map(m -> Matrix(transpose(m)), p2c),
                    c2p, map(m -> Matrix(transpose(m)), c2p), prox,
                    loctabs, pwlevels, taylor)
end

# Density derivative data on the leaves (STEP 2 inputs reused by STEP 8).
struct DensityDerivs
    flv::Array{Float64,5}           # Δf
    fl2v::Array{Float64,5}          # Δ²f
    g::Array{Float64,6}             # ∇f      (n,n,n,nd,3,nb) or empty
    gl::Array{Float64,6}            # Δ∇f
    h::Array{Float64,6}             # ∇∇f     (n,n,n,nd,6,nb) or empty
    hl::Array{Float64,6}            # Δ∇∇f
end

leafboxes(ft::FlatTree) = [ibox for ibox in 1:ft.nboxes if ft.nchild[ibox] == 0]

function _laplacian!(out, f, D2, n, sc2)
    apply_dim!(out, f, D2, 1, n, sc2, false)
    apply_dim!(out, f, D2, 2, n, sc2, true)
    apply_dim!(out, f, D2, 3, n, sc2, true)
    return out
end

const _HESS_DIRS = ((1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3))

function step2_taylor!(pot, grad, hess, s::DMKSetup, fv)
    ft, n, nd, ifpgh = s.ft, s.n, s.nd, s.ifpgh
    nb = ft.nboxes
    c0, c1, c2 = s.taylor
    flv = zeros(n, n, n, nd, nb)
    fl2v = zeros(n, n, n, nd, nb)
    ng = ifpgh >= 2 ? nb : 0
    nh = ifpgh >= 3 ? nb : 0
    g = zeros(n, n, n, nd, 3, ng); gl = zeros(n, n, n, nd, 3, ng)
    h = zeros(n, n, n, nd, 6, nh); hl = zeros(n, n, n, nd, 6, nh)
    wss = [zeros(n, n, n) for _ in 1:_nthreads()]
    parallel_foreach(leafboxes(ft), wss) do ibox, tmp
        sc = 2.0 / ft.boxsize[ft.level[ibox] + 1]
        sc2 = sc * sc
        for ind in 1:nd
            f = view(fv, :, :, :, ind, ibox)
            fl = view(flv, :, :, :, ind, ibox)
            fl2 = view(fl2v, :, :, :, ind, ibox)
            _laplacian!(fl, f, s.D2, n, sc2)
            _laplacian!(fl2, fl, s.D2, n, sc2)
            pb = view(pot, :, :, :, ind, ibox)
            @inbounds @simd for i in eachindex(pb)
                pb[i] = c0 * f[i] + c1 * fl[i] + c2 * fl2[i]
            end
            if ifpgh >= 2
                for d in 1:3
                    gd = view(g, :, :, :, ind, d, ibox)
                    gld = view(gl, :, :, :, ind, d, ibox)
                    apply_dim!(gd, f, s.D1, d, n, sc, false)
                    _laplacian!(gld, gd, s.D2, n, sc2)
                    gb = view(grad, :, :, :, ind, d, ibox)
                    @inbounds @simd for i in eachindex(gb)
                        gb[i] = c0 * gd[i] + c1 * gld[i]
                    end
                end
            end
            if ifpgh >= 3
                for m in 1:6
                    d1, d2 = _HESS_DIRS[m]
                    hm = view(h, :, :, :, ind, m, ibox)
                    hlm = view(hl, :, :, :, ind, m, ibox)
                    if d1 == d2
                        apply_dim!(hm, f, s.D2, d1, n, sc2, false)
                    else
                        apply_dim!(tmp, f, s.D1, d1, n, sc, false)
                        apply_dim!(hm, tmp, s.D1, d2, n, sc, false)
                    end
                    _laplacian!(hlm, hm, s.D2, n, sc2)
                    hb = view(hess, :, :, :, ind, m, ibox)
                    @inbounds @simd for i in eachindex(hb)
                        hb[i] = c0 * hm[i] + c1 * hlm[i]
                    end
                end
            end
        end
    end
    return DensityDerivs(flv, fl2v, g, gl, h, hl)
end

_childsides(slot::Int) = (((slot - 1) & 1) + 1, (((slot - 1) >> 1) & 1) + 1, (((slot - 1) >> 2) & 1) + 1)

struct TPWork
    w1::Vector{Float64}
    w2::Vector{Float64}
end
TPWork(m::Int) = TPWork(zeros(m), zeros(m))

function step3_upward!(pc, s::DMKSetup, fv)
    ft, n, nd, p = s.ft, s.n, s.nd, s.porder
    m = max(p, n)^3
    wss = [TPWork(m) for _ in 1:_nthreads()]
    for ilev in ft.nlevels:-1:0
        sc3 = (ft.boxsize[ilev + 1] / 2)^3
        parallel_foreach(collect(levelboxes(ft, ilev)), wss) do ibox, w
            for ind in 1:nd
                out = view(pc, :, ind, ibox)
                if ft.nchild[ibox] == 0
                    tp3!(out, view(fv, :, :, :, ind, ibox), s.den2pc, s.den2pcT, s.den2pcT, w.w1, w.w2, sc3, false)
                else
                    fill!(out, 0.0)
                    for slot in 1:8
                        jbox = ft.child[slot, ibox]
                        jbox == 0 && continue
                        sx, sy, sz = _childsides(slot)
                        tp3!(out, view(pc, :, ind, jbox), s.c2p[sx], s.c2pT[sy], s.c2pT[sz], w.w1, w.w2, 1.0, true)
                    end
                end
            end
        end
    end
    return pc
end

# Proxy outputs of one box/density for the derivative combos.
@inline function _proxy_outs(ppot, pgrad, phess, ind, ibox, ifpgh)
    o1 = view(ppot, :, ind, ibox)
    ifpgh == 1 && return (o1,)
    g = ntuple(d -> view(pgrad, :, ind, d, ibox), 3)
    ifpgh == 2 && return (o1, g...)
    h = ntuple(m -> view(phess, :, ind, m, ibox), 6)
    return (o1, g..., h...)
end

function steps456_planewave!(ppot, pgrad, phess, s::DMKSetup, pc)
    ft, nd, p, ifpgh, eps = s.ft, s.nd, s.porder, s.ifpgh, s.eps
    bs0 = ft.boxsize[1]
    # Fat Gaussians (coarser than the root box) act on the root only.
    for ilev in _NLEVSTART:-1
        r = ipwrange(s, ilev)
        isempty(r) && continue
        L = PWLevel(eps, ilev, p, bs0, bs0, @view(s.deltas[r]), @view(s.wdeltas[r]), s.prox)
        w = PWWork(L.npw, p)
        for ind in 1:nd
            proxy2pw!(w.locr, w.loci, view(pc, :, ind, 1), L, p, w)
            w.locr .*= L.kernelft
            w.loci .*= L.kernelft
            pw2proxy!(_proxy_outs(ppot, pgrad, phess, ind, 1, ifpgh), w.locr, w.loci, L, p, ifpgh, w)
        end
    end

    mpidx = zeros(Int, ft.nboxes)
    tpw = [TPWork(p^3) for _ in 1:_nthreads()]
    for ilev in 0:ft.nlevels
        boxes = [ibox for ibox in levelboxes(ft, ilev) if s.ifpwexp[ibox]]
        isempty(boxes) && continue
        L = s.pwlevels[ilev + 1]
        if L !== nothing
            for (k, ibox) in enumerate(boxes)
                mpidx[ibox] = k
            end
            mpr = Array{Float64}(undef, L.nexp, nd, length(boxes))
            mpi = Array{Float64}(undef, L.nexp, nd, length(boxes))
            wss = [PWWork(L.npw, p) for _ in 1:_nthreads()]
            parallel_foreach(boxes, wss) do ibox, w
                k = mpidx[ibox]
                for ind in 1:nd
                    proxy2pw!(view(mpr, :, ind, k), view(mpi, :, ind, k), view(pc, :, ind, ibox), L, p, w)
                end
            end
            xmin = ft.boxsize[ilev + 1]
            parallel_foreach(boxes, wss) do ibox, w
                k = mpidx[ibox]
                for ind in 1:nd
                    locr, loci = w.locr, w.loci
                    copyto!(locr, view(mpr, :, ind, k))
                    copyto!(loci, view(mpi, :, ind, k))
                    for j in 1:s.nlistpw[ibox]
                        jbox = s.listpw[j, ibox]
                        dx = round(Int, (ft.centers[1, ibox] - ft.centers[1, jbox]) / xmin)
                        dy = round(Int, (ft.centers[2, ibox] - ft.centers[2, jbox]) / xmin)
                        dz = round(Int, (ft.centers[3, ibox] - ft.centers[3, jbox]) / xmin)
                        kj = mpidx[jbox]
                        shiftpw_add!(locr, loci, view(mpr, :, ind, kj), view(mpi, :, ind, kj), L, dx, dy, dz)
                    end
                    kft = L.kernelft
                    @inbounds @simd for e in eachindex(kft)
                        locr[e] *= kft[e]
                        loci[e] *= kft[e]
                    end
                    pw2proxy!(_proxy_outs(ppot, pgrad, phess, ind, ibox, ifpgh), locr, loci, L, p, ifpgh, w)
                end
            end
        end
        # Downward pass: parent proxy potential -> children.
        parents = [ibox for ibox in boxes if ft.nchild[ibox] > 0]
        parallel_foreach(parents, tpw) do ibox, w
            for slot in 1:8
                jbox = ft.child[slot, ibox]
                jbox == 0 && continue
                sx, sy, sz = _childsides(slot)
                A, B, C = s.p2c[sx], s.p2cT[sy], s.p2cT[sz]
                for ind in 1:nd
                    tp3!(view(ppot, :, ind, jbox), view(ppot, :, ind, ibox), A, B, C, w.w1, w.w2, 1.0, true)
                    if ifpgh >= 2
                        for d in 1:3
                            tp3!(view(pgrad, :, ind, d, jbox), view(pgrad, :, ind, d, ibox), A, B, C, w.w1, w.w2, 1.0, true)
                        end
                    end
                    if ifpgh >= 3
                        for m in 1:6
                            tp3!(view(phess, :, ind, m, jbox), view(phess, :, ind, m, ibox), A, B, C, w.w1, w.w2, 1.0, true)
                        end
                    end
                end
            end
        end
    end
    return nothing
end

@inline _loctab_slot(t::Float64, c::Float64, bs::Float64) = round(Int, (t - c) / bs) + NLOCTAB + 1

function step7_local!(pot, grad, hess, s::DMKSetup, fv)
    ft, n = s.ft, s.n
    wss = [(LocalWork(n), zeros(n^3), zeros(n^3), zeros(Int, 27), Int[]) for _ in 1:_nthreads()]
    parallel_foreach(leafboxes(ft), wss) do ibox, ws
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
                                f = view(fv, :, :, :, ind, jbox)
                                tabs = (view(T.tab, :, :, sx), view(T.tab, :, :, sy), view(T.tab, :, :, sz))
                                tabsT = (view(T.tabT, :, :, sx), view(T.tabT, :, :, sy), view(T.tabT, :, :, sz))
                                inds = (view(T.ind, :, :, sx), view(T.ind, :, :, sy), view(T.ind, :, :, sz))
                                local_pot!(potb, f, wd, tabs, tabsT, inds, n, lw)
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
            tabs = (view(T.tab, :, :, sx), view(T.tab, :, :, sy), view(T.tab, :, :, sz))
            tabsT = (view(T.tabT, :, :, sx), view(T.tabT, :, :, sy), view(T.tabT, :, :, sz))
            inds = (view(T.ind, :, :, sx), view(T.ind, :, :, sy), view(T.ind, :, :, sz))
            for ind in 1:nd
                local_pot!(view(pot, :, :, :, ind, ibox), view(fv, :, :, :, ind, jbox), s.wdeltas[id],
                           tabs, tabsT, inds, n, lw)
            end
        end
    end
    return nothing
end

function _local_pgh_box!(pot, grad, hess, s::DMKSetup, fv, ibox::Int, lw::LocalWork)
    ft, n, nd, ifpgh = s.ft, s.n, s.nd, s.ifpgh
    ilev = ft.level[ibox]
    iend = _ldelta(s, ilev)
    tabsl = s.loctabs[ilev + 1]
    for jj in s.list1_ptr[ibox]:(s.list1_ptr[ibox + 1] - 1)
        jbox = s.list1[jj]
        jlev = ft.level[jbox]
        bs = ft.boxsize[jlev + 1] / 4
        sx = _loctab_slot(ft.centers[1, ibox], ft.centers[1, jbox], bs)
        sy = _loctab_slot(ft.centers[2, ibox], ft.centers[2, jbox], bs)
        sz = _loctab_slot(ft.centers[3, ibox], ft.centers[3, jbox], bs)
        for id in _local_jstart(s, ibox, jbox, ilev, jlev):iend
            T = tabsl[id]::LocTab1D
            inds = (view(T.ind, :, :, sx), view(T.ind, :, :, sy), view(T.ind, :, :, sz))
            tabs = ntuple(3) do d
                A = d == 1 ? T.tab : (d == 2 ? T.tabx : T.tabxx)
                (view(A, :, :, sx), view(A, :, :, sy), view(A, :, :, sz))
            end
            for ind in 1:nd
                local_pgh!(_box_outs(pot, grad, hess, ind, ibox, ifpgh), view(fv, :, :, :, ind, jbox),
                           s.wdeltas[id], tabs, inds, n, ifpgh, lw)
            end
        end
    end
    return nothing
end

@inline function _box_outs(pot, grad, hess, ind, ibox, ifpgh)
    o1 = view(pot, :, :, :, ind, ibox)
    g = ntuple(d -> view(grad, :, :, :, ind, d, ibox), 3)
    ifpgh == 2 && return (o1, g...)
    h = ntuple(m -> view(hess, :, :, :, ind, m, ibox), 6)
    return (o1, g..., h...)
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

function step8_asymptotic!(pot, grad, hess, s::DMKSetup, fv, dd::DensityDerivs)
    ft = s.ft
    for ilev in 0:ft.nlevels
        nas = s.idelta[ilev + 1]
        nas > 0 || continue
        coefs = _asym_coefs(s, nas)
        boxes = [ibox for ibox in levelboxes(ft, ilev) if ft.nchild[ibox] == 0]
        parallel_foreach(boxes, [nothing for _ in 1:_nthreads()]) do ibox, _
            _asym_box!(pot, grad, hess, s, fv, dd, coefs, ibox)
        end
    end
    return nothing
end

function _asym_box!(pot, grad, hess, s::DMKSetup, fv, dd::DensityDerivs, (c0, c2, c4), ibox::Int)
    for ind in 1:s.nd
        pb = view(pot, :, :, :, ind, ibox)
        f = view(fv, :, :, :, ind, ibox)
        fl = view(dd.flv, :, :, :, ind, ibox)
        fl2 = view(dd.fl2v, :, :, :, ind, ibox)
        @inbounds @simd for i in eachindex(pb)
            pb[i] += c0 * f[i] + c2 * fl[i] + c4 * fl2[i]
        end
        if s.ifpgh >= 2
            for d in 1:3
                gb = view(grad, :, :, :, ind, d, ibox)
                g = view(dd.g, :, :, :, ind, d, ibox)
                gl = view(dd.gl, :, :, :, ind, d, ibox)
                @inbounds @simd for i in eachindex(gb)
                    gb[i] += c0 * g[i] + c2 * gl[i]
                end
            end
        end
        if s.ifpgh >= 3
            for m in 1:6
                hb = view(hess, :, :, :, ind, m, ibox)
                h = view(dd.h, :, :, :, ind, m, ibox)
                hl = view(dd.hl, :, :, :, ind, m, ibox)
                @inbounds @simd for i in eachindex(hb)
                    hb[i] += c0 * h[i] + c2 * hl[i]
                end
            end
        end
    end
    return nothing
end

function step9_proxy2pot!(pot, grad, hess, s::DMKSetup, ppot, pgrad, phess)
    ft, n, nd, p, ifpgh = s.ft, s.n, s.nd, s.porder, s.ifpgh
    PE = s.poteval
    PET = s.potevalT
    wss = [TPWork(max(n, p)^3) for _ in 1:_nthreads()]
    parallel_foreach(leafboxes(ft), wss) do ibox, w
        for ind in 1:nd
            tp3!(view(pot, :, :, :, ind, ibox), view(ppot, :, ind, ibox), PE, PET, PET, w.w1, w.w2, 1.0, true)
            if ifpgh >= 2
                for d in 1:3
                    tp3!(view(grad, :, :, :, ind, d, ibox), view(pgrad, :, ind, d, ibox), PE, PET, PET, w.w1, w.w2, 1.0, true)
                end
            end
            if ifpgh >= 3
                for m in 1:6
                    tp3!(view(hess, :, :, :, ind, m, ibox), view(phess, :, ind, m, ibox), PE, PET, PET, w.w1, w.w2, 1.0, true)
                end
            end
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
`Vector{Float64}` of length ≥ 10, receives the wall time of precomputation and
STEPS 2–9 (Fortran `tottimeinfo` order) plus target evaluation.
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

function _bdmk_native(tree::BoxTree, fvals::AbstractArray{<:Real,3}, kernel::AbstractKernel, eps::Float64,
                      grad::Bool, hess::Bool, targets, timings)
    nd, np = _check_solver_inputs(tree, fvals)
    ifpgh = hess ? 3 : (grad ? 2 : 1)
    tm = timings === nothing ? zeros(10) : timings
    fill!(tm, 0.0)

    t0 = time_ns()
    s = DMKSetup(tree, nd, kernel, eps, ifpgh)
    ft, n, p = s.ft, s.n, s.porder
    nb = ft.nboxes
    fv = Array{Float64,5}(undef, n, n, n, nd, nb)
    fv3 = reshape(fv, np, nd, nb)
    for ibox in 1:nb, ind in 1:nd
        src = ft.perm[ibox]
        @inbounds for j in 1:np
            fv3[j, ind, ibox] = fvals[ind, j, src]
        end
    end
    ng = ifpgh >= 2 ? nb : 0
    nh = ifpgh >= 3 ? nb : 0
    pot = zeros(n, n, n, nd, nb)
    gradv = zeros(n, n, n, nd, 3, ng)
    hessv = zeros(n, n, n, nd, 6, nh)
    tm[1] = (time_ns() - t0) / 1e9

    t0 = time_ns()
    dd = step2_taylor!(pot, gradv, hessv, s, fv)
    tm[2] = (time_ns() - t0) / 1e9

    t0 = time_ns()
    pc = zeros(p^3, nd, nb)
    step3_upward!(pc, s, fv)
    tm[3] = (time_ns() - t0) / 1e9

    t0 = time_ns()
    ppot = zeros(p^3, nd, nb)
    pgrad = zeros(p^3, nd, 3, ng)
    phess = zeros(p^3, nd, 6, nh)
    steps456_planewave!(ppot, pgrad, phess, s, pc)
    tm[4] = (time_ns() - t0) / 1e9

    t0 = time_ns()
    step7_local!(pot, gradv, hessv, s, fv)
    tm[7] = (time_ns() - t0) / 1e9

    t0 = time_ns()
    step8_asymptotic!(pot, gradv, hessv, s, fv, dd)
    tm[8] = (time_ns() - t0) / 1e9

    t0 = time_ns()
    step9_proxy2pot!(pot, gradv, hessv, s, ppot, pgrad, phess)
    tm[9] = (time_ns() - t0) / 1e9

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
    out = zeros(nd, np, ft.nboxes)
    a3 = reshape(a, np, nd, ft.nboxes)
    for ibox in 1:ft.nboxes, ind in 1:nd
        dst = ft.perm[ibox]
        @inbounds for j in 1:np
            out[ind, j, dst] = a3[j, ind, ibox]
        end
    end
    return out
end

function _unflatten(a::Array{Float64,6}, ft::FlatTree, nd::Int, np::Int)
    nc = size(a, 5)
    out = zeros(nd, nc, np, ft.nboxes)
    a4 = reshape(a, np, nd, nc, ft.nboxes)
    for ibox in 1:ft.nboxes, c in 1:nc, ind in 1:nd
        dst = ft.perm[ibox]
        @inbounds for j in 1:np
            out[ind, c, j, dst] = a4[j, ind, c, ibox]
        end
    end
    return out
end
