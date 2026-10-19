# Level-ordered flat tree plus the box-DMK interaction lists
# (Fortran `bdmk_find_all_pwexp_boxes`, `bdmk_compute_all_listpw`,
# `bdmk_compute_all_modified_list1`).

# Boxes keep the BoxTree numbering (box 1 is the root); `levelorder` lists
# them sorted by level and laddr gives each level's range in it.
struct FlatTree
    nboxes::Int
    nlevels::Int
    levelorder::Vector{Int}
    laddr::Matrix{Int}          # (2, nlevels+1): range of each level in levelorder
    level::Vector{Int}
    parent::Vector{Int}         # 0 for the root
    nchild::Vector{Int}
    child::Matrix{Int}          # (8, nboxes), slot = 1 + sx + 2sy + 4sz; 0 = none
    ncoll::Vector{Int}
    coll::Matrix{Int}           # (27, nboxes)
    centers::Matrix{Float64}    # (3, nboxes)
    boxsize::Vector{Float64}    # boxsize[ilev+1]
end

levelboxes(ft::FlatTree, ilev::Int) = view(ft.levelorder, ft.laddr[1, ilev + 1]:ft.laddr[2, ilev + 1])
isleafbox(ft::FlatTree, ibox::Int) = ft.nchild[ibox] == 0

function FlatTree(tree::BoxTree)
    tree.ndim == 3 || throw(ArgumentError("the native solver supports ndim = 3 only"))
    nb = nboxes(tree)
    nl = tree.nlevels
    tree.level[1] == 0 || throw(ArgumentError("box 1 must be the root"))
    order = sortperm(collect(1:nb); by = ibox -> (tree.level[ibox], ibox))
    laddr = zeros(Int, 2, nl + 1)
    sorted_levels = tree.level[order]
    for ilev in 0:nl
        r = searchsorted(sorted_levels, ilev)
        laddr[1, ilev + 1] = first(r)
        laddr[2, ilev + 1] = last(r)
    end

    centers = Matrix{Float64}(tree.centers)
    level = copy(tree.level)
    parent = copy(tree.parent)
    nchild = zeros(Int, nb)
    child = zeros(Int, 8, nb)
    ncoll = zeros(Int, nb)
    coll = zeros(Int, 27, nb)
    for ibox in 1:nb
        for c in @view tree.children[:, ibox]
            c == 0 && continue
            slot = 1
            centers[1, c] > centers[1, ibox] && (slot += 1)
            centers[2, c] > centers[2, ibox] && (slot += 2)
            centers[3, c] > centers[3, ibox] && (slot += 4)
            child[slot, ibox] = c
            nchild[ibox] += 1
        end
        cl = tree.colleagues[ibox]
        ncoll[ibox] = length(cl)
        for (j, c) in enumerate(cl)
            coll[j, ibox] = c
        end
    end
    boxsize = Float64.(tree.boxsize)
    return FlatTree(nb, nl, order, laddr, level, parent, nchild, child, ncoll, coll, centers, boxsize)
end

# Boxes that need plane-wave expansions: the roots (every level-0 box of a
# forest), every nonleaf box, and every leaf with a nonleaf colleague.
function find_pwexp_boxes(ft::FlatTree)
    ifpwexp = falses(ft.nboxes)
    for ibox in levelboxes(ft, 0)
        ifpwexp[ibox] = true
    end
    for ilev in 1:ft.nlevels, ibox in levelboxes(ft, ilev)
        if ft.nchild[ibox] == 0
            for j in 1:ft.ncoll[ibox]
                if ft.nchild[ft.coll[j, ibox]] > 0
                    ifpwexp[ibox] = true
                    break
                end
            end
        else
            ifpwexp[ibox] = true
        end
    end
    return ifpwexp
end

# Plane-wave interaction list: colleagues other than the box itself, unless
# both boxes are leaves.
function compute_listpw(ft::FlatTree)
    nlistpw = zeros(Int, ft.nboxes)
    listpw = zeros(Int, 27, ft.nboxes)
    for ibox in 1:ft.nboxes, i in 1:ft.ncoll[ibox]
        jbox = ft.coll[i, ibox]
        if jbox != ibox && (ft.nchild[jbox] > 0 || ft.nchild[ibox] > 0)
            nlistpw[ibox] += 1
            listpw[nlistpw[ibox], ibox] = jbox
        end
    end
    return nlistpw, listpw
end

# Modified list 1 (near field) of each leaf, in CSR form.
function compute_list1(ft::FlatTree)
    nb = ft.nboxes
    lists = [Int[] for _ in 1:nb]
    if nb == 1
        push!(lists[1], 1)
    else
        for ilev in 0:ft.nlevels, ibox in levelboxes(ft, ilev)
            ft.nchild[ibox] == 0 || continue
            list = lists[ibox]
            for i in 1:ft.ncoll[ibox]
                jbox = ft.coll[i, ibox]
                if ft.nchild[jbox] == 0
                    push!(list, jbox)
                else
                    distest = 1.05 * (ft.boxsize[ilev + 1] + ft.boxsize[ilev + 2]) / 2
                    for j in 1:8
                        kbox = ft.child[j, jbox]
                        (kbox > 0 && ft.nchild[kbox] == 0) || continue
                        _within(ft, kbox, ibox, distest) && push!(list, kbox)
                    end
                end
            end
            dad = ft.parent[ibox]
            dad == 0 && continue
            for i in 1:ft.ncoll[dad]
                jbox = ft.coll[i, dad]
                ft.nchild[jbox] == 0 || continue
                distest = 1.05 * (ft.boxsize[ilev + 1] + ft.boxsize[ilev]) / 2
                _within(ft, jbox, ibox, distest) && push!(list, jbox)
            end
        end
    end
    ptr = ones(Int, nb + 1)
    for ibox in 1:nb
        ptr[ibox + 1] = ptr[ibox] + length(lists[ibox])
    end
    flat = Vector{Int}(undef, ptr[end] - 1)
    for ibox in 1:nb
        copyto!(flat, ptr[ibox], lists[ibox], 1, length(lists[ibox]))
    end
    return ptr, flat
end

@inline function _within(ft::FlatTree, a::Int, b::Int, distest::Float64)
    for k in 1:3
        abs(ft.centers[k, a] - ft.centers[k, b]) >= distest && return false
    end
    return true
end
