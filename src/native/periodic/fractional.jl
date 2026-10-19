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
    F = _Forest(n, NTuple{3,Float64}[], Int[], Int[], NTuple{8,Int}[], Vector{Float64}[], [1.0], Arena(), R)
    try
        return _fractional_moments!(F, f, g, ft, Ur, mt, pf, thr)
    finally
        release!(F.ar)
    end
end

function _fractional_moments!(F::_Forest, f, g::PeriodicGrid, ft::FlatTree, Ur::Matrix{Float64}, mt, pf::Int, thr::Float64)
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
    wss = [(zeros(n, n, n), zeros(n, n, n), zeros(3)) for _ in 1:_nthreads()]
    while true
        flag = zeros(Bool, length(level_boxes))
        parallel_foreach(collect(eachindex(level_boxes)), wss) do i, (w1, w2, y)
            b = level_boxes[i]
            h = F.boxsize[F.level[b] + 1]
            if _tail_error(_fbox(F, b), leg.U, tail, w1, w2) > thr
                flag[i] = true
                return
            end
            # finest main-forest leaf at the 27 points of the cell (slightly inside)
            c = F.centers[b]
            size = edge * h
            for oz in -1:1, oy in -1:1, ox in -1:1
                u1 = c[1] + 0.49 * h * ox; u2 = c[2] + 0.49 * h * oy; u3 = c[3] + 0.49 * h * oz
                for d in 1:3
                    y[d] = Ur[d, 1] * u1 + Ur[d, 2] * u2 + Ur[d, 3] * u3
                end
                leaf = _locate_leaf(ft, g, y)
                if ft.boxsize[ft.level[leaf] + 1] < size / 2
                    flag[i] = true
                    return
                end
            end
        end
        any(flag) || break
        level_boxes = _refine!(F, f, level_boxes[flag], x)
    end

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
