const _FORTRAN_CALLBACK_LOCK = ReentrantLock()
const _FORTRAN_CALLBACK_FUNC = Ref{Any}(nothing)
const _FORTRAN_CALLBACK_NDIM = Ref{Cint}(0)
const _FORTRAN_CALLBACK_SHIFT = Ref{Float64}(0.0)
const _FORTRAN_CALLBACK_ERROR = Ref{Any}(nothing)
const _FORTRAN_CALLBACK_PTR = Ref{Ptr{Cvoid}}(C_NULL)
const _FORTRAN_CALLBACK_POINT = Vector{Float64}(undef, 10)  # pre-allocated point buffer
const _FORTRAN_TREE_REGISTRY = WeakKeyDict{Any, Any}()

struct FortranTreeData{B<:AbstractBasis}
    tree::BoxTree{Float64, B}
    fvals::Array{Float64, 3}
    itree::Vector{Cint}
    iptr::Vector{Cint}
    ltree::Cint
    centers::Matrix{Float64}
    boxsize::Vector{Float64}
end

Base.length(::FortranTreeData) = 2

function Base.iterate(data::FortranTreeData, state::Int = 1)
    state == 1 && return data.tree, 2
    state == 2 && return data.fvals, 3
    return nothing
end

function _fortran_libboxdmk_path()
    path = _resolve_fortran_library_path(; must_exist = true)
    return path
end

function _fortran_solve_libboxdmk_path()
    path = _resolve_fortran_solve_library_path(; must_exist = true)
    return path
end

function _fortran_kernel(kernel::AbstractKernel)
    if kernel isa LaplaceKernel
        return Cint(1), 0.0
    elseif kernel isa YukawaKernel
        return Cint(0), Float64(kernel.beta)
    elseif kernel isa SqrtLaplaceKernel
        return Cint(2), 0.0
    end

    throw(ArgumentError("unsupported kernel type $(typeof(kernel))"))
end

function _fortran_basis(basis::AbstractBasis)
    if basis isa LegendreBasis
        return Cint(0)
    elseif basis isa ChebyshevBasis
        return Cint(1)
    end

    throw(ArgumentError("unsupported basis type $(typeof(basis))"))
end

function _clear_fortran_callback_state!()
    _FORTRAN_CALLBACK_FUNC[] = nothing
    _FORTRAN_CALLBACK_NDIM[] = 0
    _FORTRAN_CALLBACK_SHIFT[] = 0.0
    _FORTRAN_CALLBACK_ERROR[] = nothing
    return nothing
end

function _throw_fortran_callback_error!()
    err = _FORTRAN_CALLBACK_ERROR[]
    err === nothing && return nothing
    _FORTRAN_CALLBACK_ERROR[] = nothing
    throw(first(err))
end

function _fortran_callback(
    nd::Cint,
    xyz::Ptr{Cdouble},
    dpars::Ptr{Cdouble},
    zpars::Ptr{ComplexF64},
    ipars::Ptr{Cint},
    fout::Ptr{Cdouble},
)::Cvoid
    _ = dpars
    _ = zpars
    _ = ipars

    f = _FORTRAN_CALLBACK_FUNC[]
    ndim = Int(_FORTRAN_CALLBACK_NDIM[])
    shift = _FORTRAN_CALLBACK_SHIFT[]

    if f === nothing
        for i in 1:Int(nd)
            unsafe_store!(fout, 0.0, i)
        end
        return
    end

    point = _FORTRAN_CALLBACK_POINT
    @inbounds for dim in 1:ndim
        point[dim] = unsafe_load(xyz, dim) + shift
    end

    values = ndim < length(point) ? f(@view(point[1:ndim])) : f(point)

    if values isa Number
        unsafe_store!(fout, Float64(values), 1)
        return
    end

    @inbounds for i in 1:Int(nd)
        unsafe_store!(fout, Float64(values[i]), i)
    end

    return
end

const _FORTRAN_HANDLES = Dict{String, Ptr{Cvoid}}()

# The Fortran reference library is optional: it is only opened on first use.
function _fortran_sym(path::AbstractString, sym::Symbol)
    handle = get!(() -> Libdl.dlopen(path), _FORTRAN_HANDLES, path)
    return Libdl.dlsym(handle, sym)
end

function __init__()
    _FORTRAN_CALLBACK_PTR[] = @cfunction(
        _fortran_callback,
        Cvoid,
        (Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{ComplexF64}, Ptr{Cint}, Ptr{Cdouble}),
    )
end

function _fortran_vol_tree_mem!(
    ndim::Cint,
    ipoly::Cint,
    iperiod::Cint,
    eps::Cdouble,
    zk::ComplexF64,
    boxlen::Cdouble,
    norder::Cint,
    iptype::Cint,
    eta::Cdouble,
    nd::Cint,
    dpars::Vector{Float64},
    zpars::Vector{ComplexF64},
    ipars::Vector{Cint},
    ifnewtree::Cint,
    nboxes::Ref{Cint},
    nlevels::Ref{Cint},
    ltree::Ref{Cint},
    rintl::Vector{Float64},
)
    ndim_ref = Ref{Cint}(ndim)
    ipoly_ref = Ref{Cint}(ipoly)
    iperiod_ref = Ref{Cint}(iperiod)
    eps_ref = Ref{Cdouble}(eps)
    zk_ref = Ref{ComplexF64}(zk)
    boxlen_ref = Ref{Cdouble}(boxlen)
    norder_ref = Ref{Cint}(norder)
    iptype_ref = Ref{Cint}(iptype)
    eta_ref = Ref{Cdouble}(eta)
    nd_ref = Ref{Cint}(nd)
    ifnewtree_ref = Ref{Cint}(ifnewtree)

    ccall(
        _fortran_sym(_fortran_libboxdmk_path(), :boxdmk_vol_tree_mem),
        Cvoid,
        (
            Ref{Cint},
            Ref{Cint},
            Ref{Cint},
            Ref{Cdouble},
            Ref{ComplexF64},
            Ref{Cdouble},
            Ref{Cint},
            Ref{Cint},
            Ref{Cdouble},
            Ptr{Cvoid},
            Ref{Cint},
            Ptr{Cdouble},
            Ptr{ComplexF64},
            Ptr{Cint},
            Ref{Cint},
            Ref{Cint},
            Ref{Cint},
            Ref{Cint},
            Ptr{Cdouble},
        ),
        ndim_ref,
        ipoly_ref,
        iperiod_ref,
        eps_ref,
        zk_ref,
        boxlen_ref,
        norder_ref,
        iptype_ref,
        eta_ref,
        _FORTRAN_CALLBACK_PTR[],
        nd_ref,
        dpars,
        zpars,
        ipars,
        ifnewtree_ref,
        nboxes,
        nlevels,
        ltree,
        rintl,
    )
    return nothing
end

function _fortran_vol_tree_build!(
    ndim::Cint,
    ipoly::Cint,
    iperiod::Cint,
    eps::Cdouble,
    zk::ComplexF64,
    boxlen::Cdouble,
    norder::Cint,
    iptype::Cint,
    eta::Cdouble,
    nd::Cint,
    dpars::Vector{Float64},
    zpars::Vector{ComplexF64},
    ipars::Vector{Cint},
    rintl::Vector{Float64},
    nboxes::Ref{Cint},
    nlevels::Ref{Cint},
    ltree::Ref{Cint},
    itree::Vector{Cint},
    iptr::Vector{Cint},
    centers::Matrix{Float64},
    boxsize::Vector{Float64},
    fvals::Array{Float64, 3},
)
    ndim_ref = Ref{Cint}(ndim)
    ipoly_ref = Ref{Cint}(ipoly)
    iperiod_ref = Ref{Cint}(iperiod)
    eps_ref = Ref{Cdouble}(eps)
    zk_ref = Ref{ComplexF64}(zk)
    boxlen_ref = Ref{Cdouble}(boxlen)
    norder_ref = Ref{Cint}(norder)
    iptype_ref = Ref{Cint}(iptype)
    eta_ref = Ref{Cdouble}(eta)
    nd_ref = Ref{Cint}(nd)

    ccall(
        _fortran_sym(_fortran_libboxdmk_path(), :boxdmk_vol_tree_build),
        Cvoid,
        (
            Ref{Cint},
            Ref{Cint},
            Ref{Cint},
            Ref{Cdouble},
            Ref{ComplexF64},
            Ref{Cdouble},
            Ref{Cint},
            Ref{Cint},
            Ref{Cdouble},
            Ptr{Cvoid},
            Ref{Cint},
            Ptr{Cdouble},
            Ptr{ComplexF64},
            Ptr{Cint},
            Ptr{Cdouble},
            Ref{Cint},
            Ref{Cint},
            Ref{Cint},
            Ptr{Cint},
            Ptr{Cint},
            Ptr{Cdouble},
            Ptr{Cdouble},
            Ptr{Cdouble},
        ),
        ndim_ref,
        ipoly_ref,
        iperiod_ref,
        eps_ref,
        zk_ref,
        boxlen_ref,
        norder_ref,
        iptype_ref,
        eta_ref,
        _FORTRAN_CALLBACK_PTR[],
        nd_ref,
        dpars,
        zpars,
        ipars,
        rintl,
        nboxes,
        nlevels,
        ltree,
        itree,
        iptr,
        centers,
        boxsize,
        fvals,
    )
    return nothing
end

function _fortran_bdmk!(
    nd::Cint,
    ndim::Cint,
    eps::Cdouble,
    ikernel::Cint,
    beta::Cdouble,
    ipoly::Cint,
    norder::Cint,
    npbox_value::Cint,
    nboxes::Cint,
    nlevels::Cint,
    ltree::Cint,
    itree::Vector{Cint},
    iptr::Vector{Cint},
    centers::Matrix{Float64},
    boxsize::Vector{Float64},
    fvals::Array{Float64, 3},
    ifpgh::Cint,
    pot,
    grad,
    hess,
    ntarg::Cint,
    targs,
    ifpghtarg::Cint,
    pote,
    grade,
    hesse,
    tottimeinfo::Vector{Float64},
)
    nd_ref = Ref{Cint}(nd)
    ndim_ref = Ref{Cint}(ndim)
    eps_ref = Ref{Cdouble}(eps)
    ikernel_ref = Ref{Cint}(ikernel)
    beta_ref = Ref{Cdouble}(beta)
    ipoly_ref = Ref{Cint}(ipoly)
    norder_ref = Ref{Cint}(norder)
    npbox_ref = Ref{Cint}(npbox_value)
    nboxes_ref = Ref{Cint}(nboxes)
    nlevels_ref = Ref{Cint}(nlevels)
    ltree_ref = Ref{Cint}(ltree)
    ifpgh_ref = Ref{Cint}(ifpgh)
    ntarg_ref = Ref{Cint}(ntarg)
    ifpghtarg_ref = Ref{Cint}(ifpghtarg)

    ccall(
        _fortran_sym(_fortran_solve_libboxdmk_path(), :boxdmk_bdmk),
        Cvoid,
        (
            Ref{Cint},
            Ref{Cint},
            Ref{Cdouble},
            Ref{Cint},
            Ref{Cdouble},
            Ref{Cint},
            Ref{Cint},
            Ref{Cint},
            Ref{Cint},
            Ref{Cint},
            Ref{Cint},
            Ptr{Cint},
            Ptr{Cint},
            Ptr{Cdouble},
            Ptr{Cdouble},
            Ptr{Cdouble},
            Ref{Cint},
            Ptr{Cdouble},
            Ptr{Cdouble},
            Ptr{Cdouble},
            Ref{Cint},
            Ptr{Cdouble},
            Ref{Cint},
            Ptr{Cdouble},
            Ptr{Cdouble},
            Ptr{Cdouble},
            Ptr{Cdouble},
        ),
        nd_ref,
        ndim_ref,
        eps_ref,
        ikernel_ref,
        beta_ref,
        ipoly_ref,
        norder_ref,
        npbox_ref,
        nboxes_ref,
        nlevels_ref,
        ltree_ref,
        itree,
        iptr,
        centers,
        boxsize,
        fvals,
        ifpgh_ref,
        pot,
        grad,
        hess,
        ntarg_ref,
        targs,
        ifpghtarg_ref,
        pote,
        grade,
        hesse,
        tottimeinfo,
    )
    return nothing
end

_fortran_parent(raw::Integer) = raw < 0 ? 0 : Int(raw)

function _unpack_fortran_tree(
    itree::Vector{Cint},
    iptr::Vector{Cint},
    centers::Matrix{Float64},
    boxsize::Vector{Float64},
    nboxes::Int,
    nlevels::Int,
    basis::B,
    norder::Int,
) where {B<:AbstractBasis}
    ndim = size(centers, 1)
    mc = 2^ndim
    mnbors = 3^ndim
    level = Vector{Int}(undef, nboxes)
    parent = Vector{Int}(undef, nboxes)
    children = zeros(Int, mc, nboxes)
    colleagues = [Int[] for _ in 1:nboxes]

    ilevel_ptr = Int(iptr[2])
    iparent_ptr = Int(iptr[3])
    nchild_ptr = Int(iptr[4])
    ichild_ptr = Int(iptr[5])
    ncoll_ptr = Int(iptr[6])
    coll_ptr = Int(iptr[7])

    for ibox in 1:nboxes
        level[ibox] = Int(itree[ilevel_ptr + ibox - 1])
        parent[ibox] = _fortran_parent(itree[iparent_ptr + ibox - 1])

        if Int(itree[nchild_ptr + ibox - 1]) > 0
            for ichild in 1:mc
                children[ichild, ibox] = _fortran_parent(itree[ichild_ptr + (ibox - 1) * mc + ichild - 1])
            end
        end

        ncoll = max(0, Int(itree[ncoll_ptr + ibox - 1]))
        for icoll in 1:ncoll
            jbox = Int(itree[coll_ptr + (ibox - 1) * mnbors + icoll - 1])
            jbox > 0 && push!(colleagues[ibox], jbox)
        end
    end

    return BoxTree(
        ndim,
        nlevels,
        centers,
        boxsize,
        parent,
        children,
        colleagues,
        level,
        basis,
        norder,
    )
end

function _matches_fortran_tree(tree::BoxTree, data::FortranTreeData)
    return tree.ndim == data.tree.ndim &&
        tree.nlevels == data.tree.nlevels &&
        tree.norder == data.tree.norder &&
        tree.basis === data.tree.basis &&
        tree.centers === data.tree.centers &&
        tree.boxsize === data.tree.boxsize &&
        tree.parent === data.tree.parent &&
        tree.children === data.tree.children &&
        tree.colleagues === data.tree.colleagues &&
        tree.level === data.tree.level
end

function _fortran_registry_lookup(tree::BoxTree, fvals)
    data = get(_FORTRAN_TREE_REGISTRY, fvals, nothing)
    data isa FortranTreeData || throw(ArgumentError("fvals must come from build_tree_fortran or pass the FortranTreeData wrapper directly"))
    _matches_fortran_tree(tree, data) || throw(ArgumentError("tree does not match the Fortran data associated with fvals"))
    return data
end

function _fortran_output_flag(grad::Bool, hess::Bool)
    return Cint(hess ? 3 : (grad ? 2 : 1))
end

function _fortran_level_order(tree::BoxTree)
    order = collect(1:nboxes(tree))
    sort!(order; by = ibox -> (tree.level[ibox], ibox))
    return order
end

function _is_identity_order(order::AbstractVector{<:Integer})
    for (index, value) in pairs(order)
        value == index || return false
    end
    return true
end

function _reorder_tree(tree::BoxTree, order::AbstractVector{<:Integer})
    _is_identity_order(order) && return tree

    nb = nboxes(tree)
    length(order) == nb || throw(ArgumentError("order length must equal the number of boxes"))
    inverse = zeros(Int, nb)
    for (new_box, old_box) in pairs(order)
        inverse[old_box] = new_box
    end

    centers = tree.centers[:, order]
    parent = zeros(Int, nb)
    children = zeros(Int, size(tree.children, 1), nb)
    colleagues = Vector{Vector{Int}}(undef, nb)
    level = tree.level[order]

    for new_box in 1:nb
        old_box = order[new_box]
        old_parent = tree.parent[old_box]
        parent[new_box] = old_parent == 0 ? 0 : inverse[old_parent]

        for child_slot in 1:size(children, 1)
            old_child = tree.children[child_slot, old_box]
            children[child_slot, new_box] = old_child == 0 ? 0 : inverse[old_child]
        end

        colleagues[new_box] = [inverse[old_peer] for old_peer in tree.colleagues[old_box]]
    end

    return BoxTree(
        tree.ndim,
        tree.nlevels,
        centers,
        copy(tree.boxsize),
        parent,
        children,
        colleagues,
        level,
        tree.basis,
        tree.norder,
    )
end

function pack_tree_fortran(tree::BoxTree, fvals)
    _check_solver_inputs(tree, fvals)

    order = _fortran_level_order(tree)
    packed_tree = _reorder_tree(tree, order)
    packed_fvals = _is_identity_order(order) && fvals isa Array{Float64, 3} ? copy(fvals) : Float64.(fvals[:, :, order])

    ndim = packed_tree.ndim
    nb = nboxes(packed_tree)
    nl = packed_tree.nlevels
    mc = 2^ndim
    mnbors = 3^ndim

    ltree = Cint((4 + mc + mnbors) * nb + 2 * (nl + 1))
    iptr = Vector{Cint}(undef, 8)
    iptr[1] = 1
    iptr[2] = Cint(2 * (nl + 1) + 1)
    iptr[3] = Cint(iptr[2] + nb)
    iptr[4] = Cint(iptr[3] + nb)
    iptr[5] = Cint(iptr[4] + nb)
    iptr[6] = Cint(iptr[5] + mc * nb)
    iptr[7] = Cint(iptr[6] + nb)
    iptr[8] = Cint(iptr[7] + mnbors * nb)

    itree = fill(Cint(-1), Int(ltree))

    next_box = 1
    for level in 0:nl
        level_count = count(==(level), packed_tree.level)
        start_index = next_box
        end_index = next_box + level_count - 1
        itree[2 * level + 1] = Cint(start_index)
        itree[2 * level + 2] = Cint(end_index)
        next_box = end_index + 1
    end

    level_ptr = Int(iptr[2])
    parent_ptr = Int(iptr[3])
    nchild_ptr = Int(iptr[4])
    child_ptr = Int(iptr[5])
    ncoll_ptr = Int(iptr[6])
    coll_ptr = Int(iptr[7])

    for ibox in 1:nb
        itree[level_ptr + ibox - 1] = Cint(packed_tree.level[ibox])
        itree[parent_ptr + ibox - 1] = Cint(packed_tree.parent[ibox] == 0 ? -1 : packed_tree.parent[ibox])

        nchildren = 0
        for child_slot in 1:mc
            child_box = packed_tree.children[child_slot, ibox]
            if child_box == 0
                itree[child_ptr + (ibox - 1) * mc + child_slot - 1] = Cint(-1)
            else
                itree[child_ptr + (ibox - 1) * mc + child_slot - 1] = Cint(child_box)
                nchildren += 1
            end
        end
        itree[nchild_ptr + ibox - 1] = Cint(nchildren)

        ncoll = min(length(packed_tree.colleagues[ibox]), mnbors)
        itree[ncoll_ptr + ibox - 1] = Cint(ncoll)
        for j in 1:mnbors
            value = j <= ncoll ? packed_tree.colleagues[ibox][j] : -1
            itree[coll_ptr + (ibox - 1) * mnbors + j - 1] = Cint(value)
        end
    end

    shift = packed_tree.boxsize[1] / 2
    centers = copy(packed_tree.centers)
    centers .-= shift
    boxsize = copy(packed_tree.boxsize)

    data = FortranTreeData(packed_tree, packed_fvals, itree, iptr, ltree, centers, boxsize)
    _FORTRAN_TREE_REGISTRY[packed_fvals] = data
    return data
end

"""
    build_tree_fortran(f, kernel, basis; ndim=3, norder=6, eps=1e-6, boxlen=1.0, nd=1, eta=1.0)

Build an adaptive tree using the Fortran library. Returns a `FortranTreeData`
wrapper that can also be destructured as `(tree, fvals)` to match the Julia API.
"""
function build_tree_fortran(
    f,
    kernel::AbstractKernel,
    basis::AbstractBasis;
    ndim = 3,
    norder = 6,
    eps = 1e-6,
    boxlen = 1.0,
    nd = 1,
    eta = 1.0,
)
    _fortran_libboxdmk_path()

    ndim_int = Int(ndim)
    ndim_int > 0 || throw(ArgumentError("ndim must be positive"))
    nd_int = Int(nd)
    nd_int > 0 || throw(ArgumentError("nd must be positive"))
    norder_int = _check_basis_order(norder)

    eps_value = Float64(eps)
    eps_value > 0 || throw(ArgumentError("eps must be positive"))
    boxlen_value = Float64(boxlen)
    boxlen_value > 0 || throw(ArgumentError("boxlen must be positive"))

    ikernel, _ = _fortran_kernel(kernel)
    ipoly = _fortran_basis(basis)
    iperiod = Cint(0)
    iptype = Cint(2)
    ifnewtree = Cint(0)
    eta_value = Float64(eta)
    zk = ComplexF64(30.0, 0.0)

    dpars = zeros(Float64, 1)
    zpars = zeros(ComplexF64, 1)
    ipars = zeros(Cint, 10)
    ipars[1] = Cint(ndim_int)
    ipars[2] = ikernel
    ipars[5] = iperiod
    ipars[10] = Cint(1)

    nboxes = Ref{Cint}(0)
    nlevels = Ref{Cint}(0)
    ltree = Ref{Cint}(0)
    rintl = zeros(Float64, 201)
    shift = boxlen_value / 2

    lock(_FORTRAN_CALLBACK_LOCK)
    try
        _clear_fortran_callback_state!()
        _FORTRAN_CALLBACK_FUNC[] = f
        _FORTRAN_CALLBACK_NDIM[] = Cint(ndim_int)
        _FORTRAN_CALLBACK_SHIFT[] = shift

        _fortran_vol_tree_mem!(
            Cint(ndim_int),
            ipoly,
            iperiod,
            Cdouble(eps_value),
            zk,
            Cdouble(boxlen_value),
            Cint(norder_int),
            iptype,
            Cdouble(eta_value),
            Cint(nd_int),
            dpars,
            zpars,
            ipars,
            ifnewtree,
            nboxes,
            nlevels,
            ltree,
            rintl,
        )
        _throw_fortran_callback_error!()

        nboxes_int = Int(nboxes[])
        nlevels_int = Int(nlevels[])
        npbox_value = npbox(norder_int, ndim_int)

        itree = Vector{Cint}(undef, Int(ltree[]))
        iptr = Vector{Cint}(undef, 8)
        centers_fortran = Matrix{Float64}(undef, ndim_int, nboxes_int)
        boxsize = Vector{Float64}(undef, nlevels_int + 1)
        fvals = Array{Float64, 3}(undef, nd_int, npbox_value, nboxes_int)

        _fortran_vol_tree_build!(
            Cint(ndim_int),
            ipoly,
            iperiod,
            Cdouble(eps_value),
            zk,
            Cdouble(boxlen_value),
            Cint(norder_int),
            iptype,
            Cdouble(eta_value),
            Cint(nd_int),
            dpars,
            zpars,
            ipars,
            rintl,
            nboxes,
            nlevels,
            ltree,
            itree,
            iptr,
            centers_fortran,
            boxsize,
            fvals,
        )
        _throw_fortran_callback_error!()

        centers_julia = copy(centers_fortran)
        centers_julia .+= shift
        tree = _unpack_fortran_tree(
            itree,
            iptr,
            centers_julia,
            copy(boxsize),
            nboxes_int,
            nlevels_int,
            basis,
            norder_int,
        )
        data = FortranTreeData(tree, fvals, itree, iptr, ltree[], centers_fortran, boxsize)
        _FORTRAN_TREE_REGISTRY[fvals] = data
        return data
    finally
        _clear_fortran_callback_state!()
        unlock(_FORTRAN_CALLBACK_LOCK)
    end
end

"""
    bdmk_fortran(tree, fvals, kernel; eps=1e-6, grad=false, hess=false, targets=nothing)

Solve using the Fortran library. Accepts either the `FortranTreeData` wrapper or
the destructured `(tree, fvals)` returned by `build_tree_fortran`.
"""
function bdmk_fortran(
    data::FortranTreeData,
    kernel::AbstractKernel;
    eps = 1e-6,
    grad = false,
    hess = false,
    targets = nothing,
)
    eps_value = Float64(eps)
    eps_value > 0 || throw(ArgumentError("eps must be positive"))

    tree = data.tree
    nd, np = _check_solver_inputs(tree, data.fvals)
    ikernel, beta = _fortran_kernel(kernel)
    ipoly = _fortran_basis(tree.basis)
    ifpgh = _fortran_output_flag(grad, hess)
    ifpghtarg = targets === nothing ? Cint(0) : ifpgh
    nboxes_int = nboxes(tree)
    nhess_value = nhess(tree.ndim)

    pot = zeros(Float64, nd, np, nboxes_int)
    grad_buffer = if Int(ifpgh) >= 2
        zeros(Float64, nd, tree.ndim, np, nboxes_int)
    else
        zeros(Float64, 1)
    end
    hess_buffer = if Int(ifpgh) >= 3
        zeros(Float64, nd, nhess_value, np, nboxes_int)
    else
        zeros(Float64, 1)
    end

    if targets === nothing
        ntarg = Cint(1)
        targs = zeros(Float64, tree.ndim, 1)
        pote = zeros(Float64, nd, 1)
        grade = zeros(Float64, 1)
        hesse = zeros(Float64, 1)
    else
        targs = _normalize_targets(targets, tree)
        targs = copy(targs)
        targs .-= data.boxsize[1] / 2
        ntarg = Cint(size(targs, 2))
        pote = zeros(Float64, nd, Int(ntarg))
        grade = if Int(ifpghtarg) >= 2
            zeros(Float64, nd, tree.ndim, Int(ntarg))
        else
            zeros(Float64, 1)
        end
        hesse = if Int(ifpghtarg) >= 3
            zeros(Float64, nd, nhess_value, Int(ntarg))
        else
            zeros(Float64, 1)
        end
    end

    tottimeinfo = zeros(Float64, 20)
    _fortran_bdmk!(
        Cint(nd),
        Cint(tree.ndim),
        Cdouble(eps_value),
        ikernel,
        Cdouble(beta),
        ipoly,
        Cint(tree.norder),
        Cint(np),
        Cint(nboxes_int),
        Cint(tree.nlevels),
        data.ltree,
        data.itree,
        data.iptr,
        data.centers,
        data.boxsize,
        data.fvals,
        ifpgh,
        pot,
        grad_buffer,
        hess_buffer,
        ntarg,
        targs,
        ifpghtarg,
        pote,
        grade,
        hesse,
        tottimeinfo,
    )

    grad_out = grad ? grad_buffer : nothing
    hess_out = hess ? hess_buffer : nothing
    target_pot = targets === nothing ? nothing : pote
    target_grad = targets === nothing || !grad ? nothing : grade
    target_hess = targets === nothing || !hess ? nothing : hesse

    return SolverResult(pot, grad_out, hess_out, target_pot, target_grad, target_hess)
end

function bdmk_fortran(
    tree::BoxTree,
    fvals,
    kernel::AbstractKernel;
    eps = 1e-6,
    grad = false,
    hess = false,
    targets = nothing,
)
    _check_solver_inputs(tree, fvals)
    data = get(_FORTRAN_TREE_REGISTRY, fvals, nothing)
    if !(data isa FortranTreeData && _matches_fortran_tree(tree, data))
        data = pack_tree_fortran(tree, fvals)
    end
    return bdmk_fortran(data, kernel; eps = eps, grad = grad, hess = hess, targets = targets)
end
