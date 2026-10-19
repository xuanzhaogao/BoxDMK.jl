function _check_tensor_product_inputs(out, mats, vals, n::Int, ndim::Int, nd::Int)
    n > 0 || throw(ArgumentError("tensor size n must be positive"))
    ndim > 0 || throw(ArgumentError("ndim must be positive"))
    nd > 0 || throw(ArgumentError("nd must be positive"))
    length(mats) == ndim || throw(ArgumentError("expected $ndim matrices, got $(length(mats))"))

    ngrid = n^ndim
    size(vals) == (nd, ngrid) || throw(DimensionMismatch("vals must have size ($nd, $ngrid)"))
    size(out) == (nd, ngrid) || throw(DimensionMismatch("out must have size ($nd, $ngrid)"))

    for (d, mat) in pairs(mats)
        size(mat) == (n, n) || throw(DimensionMismatch("matrix $d must have size ($n, $n)"))
    end

    return ngrid
end

_tensor_apply_workspace(::Type{T}, nd::Int, n::Int, ndim::Int) where {T} =
    _rect_tensor_apply_workspace(T, nd, n, n, ndim)

function _tensor_product_apply_dim!(dest, src, mat, n::Int, ndim::Int, nd::Int, dim::Int)
    workspace = _tensor_apply_workspace(eltype(dest), nd, n, ndim)
    return _tensor_product_apply_dim!(dest, src, mat, n, ndim, nd, dim, workspace)
end

function _tensor_product_apply_dim!(dest, src, mat, n::Int, ndim::Int, nd::Int, dim::Int, workspace)
    ndim == 3 && return _tensor_product_apply_dim_3d!(dest, src, mat, n, nd, dim, workspace)
    return _tensor_product_apply_dim!(dest, src, mat, n, ndim, nd, dim, workspace.tmp_perm, workspace.result_2d)
end

function _tensor_product_apply_dim_3d!(dest, src, mat, n::Int, nd::Int, dim::Int)
    workspace = _tensor_apply_workspace(eltype(dest), nd, n, 3)
    return _tensor_product_apply_dim_3d!(dest, src, mat, n, nd, dim, workspace)
end

function _tensor_product_apply_dim_3d!(dest, src, mat, n::Int, nd::Int, dim::Int, workspace)
    1 <= dim <= 3 || throw(ArgumentError("3D tensor dimension must be 1, 2, or 3"))
    size(mat) == (n, n) || throw(DimensionMismatch("matrix must have size ($n, $n)"))
    size(src) == (nd, n^3) || throw(DimensionMismatch("src must have size ($nd, $(n^3))"))
    size(dest) == (nd, n^3) || throw(DimensionMismatch("dest must have size ($nd, $(n^3))"))

    _check_rect_tensor_3d_workspace(workspace, n, n)

    packed_src = @view(workspace.packed_src[1:(n^3)])
    packed_dest = @view(workspace.packed_dest[1:(n^3)])
    ff = @view(workspace.ff[1:(n^2), 1:n])
    fft = @view(workspace.fft[1:n, 1:(n * n)])
    mat_t = transpose(mat)

    for iv in 1:nd
        copyto!(packed_src, @view(src[iv, :]))

        if dim == 1
            src_2d = reshape(packed_src, n, n * n)
            dest_2d = reshape(packed_dest, n, n * n)
            mul!(dest_2d, mat, src_2d)
        elseif dim == 2
            src_3d = reshape(packed_src, n, n, n)
            ff_3d = reshape(ff, n, n, n)
            @inbounds for i3 in 1:n, i1 in 1:n, i2 in 1:n
                ff_3d[i2, i1, i3] = src_3d[i1, i2, i3]
            end

            mul!(fft, mat, reshape(ff, n, n * n))

            fft_3d = reshape(fft, n, n, n)
            dest_3d = reshape(packed_dest, n, n, n)
            @inbounds for i3 in 1:n, i2 in 1:n, i1 in 1:n
                dest_3d[i1, i2, i3] = fft_3d[i2, i1, i3]
            end
        else
            src_2d = reshape(packed_src, n * n, n)
            dest_2d = reshape(packed_dest, n * n, n)
            mul!(dest_2d, src_2d, mat_t)
        end

        copyto!(@view(dest[iv, :]), packed_dest)
    end

    return dest
end

function _tensor_product_apply_dim!(
    dest,
    src,
    mat,
    n::Int,
    ndim::Int,
    nd::Int,
    dim::Int,
    tmp_perm::AbstractMatrix,
    result_2d::AbstractMatrix,
)
    prefix = n^(dim - 1)
    suffix = n^(ndim - dim)
    mat_t = transpose(mat)

    for iv in 1:nd
        if prefix == 1
            src_flat = reshape(@view(src[iv, :]), n, suffix)
            dest_flat = reshape(@view(dest[iv, :]), n, suffix)
            mul!(dest_flat, mat, src_flat)
        elseif suffix == 1
            src_2d = reshape(@view(src[iv, :]), prefix, n)
            dest_2d = reshape(@view(dest[iv, :]), prefix, n)
            mul!(dest_2d, src_2d, mat_t)
        else
            src_3d = reshape(@view(src[iv, :]), prefix, n, suffix)
            rows = prefix * suffix
            tmp_2d = @view(tmp_perm[1:rows, 1:n])
            tmp_3d = reshape(tmp_2d, prefix, suffix, n)
            permutedims!(tmp_3d, src_3d, (1, 3, 2))
            result_2d_view = @view(result_2d[1:rows, 1:n])
            mul!(result_2d_view, tmp_2d, mat_t)
            result_3d = reshape(result_2d_view, prefix, suffix, n)
            dest_3d = reshape(@view(dest[iv, :]), prefix, n, suffix)
            permutedims!(dest_3d, result_3d, (1, 3, 2))
        end
    end

    return dest
end

function tensor_product_apply!(out, mats::Tuple, vals, n, ndim, nd)
    workspace = _tensor_apply_workspace(eltype(out), Int(nd), Int(n), Int(ndim))
    return tensor_product_apply!(out, mats, vals, n, ndim, nd, workspace)
end

function tensor_product_apply!(out, mats::Tuple, vals, n, ndim, nd, workspace)
    n_int = Int(n)
    ndim_int = Int(ndim)
    nd_int = Int(nd)
    _check_tensor_product_inputs(out, mats, vals, n_int, ndim_int, nd_int)

    ngrid = n_int^ndim_int
    middle_rows = _rect_tensor_middle_rows(n_int, n_int, ndim_int)
    size(workspace.scratch_a, 1) == nd_int || throw(DimensionMismatch("workspace scratch_a must have $nd_int rows"))
    size(workspace.scratch_b, 1) == nd_int || throw(DimensionMismatch("workspace scratch_b must have $nd_int rows"))
    size(workspace.scratch_a, 2) >= ngrid || throw(DimensionMismatch("workspace scratch_a must have at least $ngrid columns"))
    size(workspace.scratch_b, 2) >= ngrid || throw(DimensionMismatch("workspace scratch_b must have at least $ngrid columns"))
    size(workspace.tmp_perm, 1) >= middle_rows || throw(DimensionMismatch("workspace tmp_perm must have at least $middle_rows rows"))
    size(workspace.tmp_perm, 2) >= n_int || throw(DimensionMismatch("workspace tmp_perm must have at least $n_int columns"))
    size(workspace.result_2d, 1) >= middle_rows || throw(DimensionMismatch("workspace result_2d must have at least $middle_rows rows"))
    size(workspace.result_2d, 2) >= n_int || throw(DimensionMismatch("workspace result_2d must have at least $n_int columns"))

    src = vals
    src_slot = 0

    for dim in 1:ndim_int
        dest, dest_slot = if dim == ndim_int && src !== out
            (out, 3)
        elseif src_slot == 1
            (@view(workspace.scratch_b[:, 1:ngrid]), 2)
        else
            (@view(workspace.scratch_a[:, 1:ngrid]), 1)
        end

        _tensor_product_apply_dim!(dest, src, mats[dim], n_int, ndim_int, nd_int, dim, workspace)
        src = dest
        src_slot = dest_slot
    end

    src === out || copyto!(out, src)
    return out
end

function _child_half_nodes(parent_nodes::AbstractVector, side::Int)
    side == 0 && return 0.5 .* parent_nodes .- 0.5
    side == 1 && return 0.5 .* parent_nodes .+ 0.5
    throw(ArgumentError("child side must be 0 (left) or 1 (right)"))
end

function p2c_transform(basis::AbstractBasis, norder, ndim)
    norder_int = _check_basis_order(norder)
    ndim_int = Int(ndim)
    ndim_int > 0 || throw(ArgumentError("ndim must be positive"))

    parent_nodes, _ = nodes_and_weights(basis, norder_int)
    left_nodes = _child_half_nodes(parent_nodes, 0)
    right_nodes = _child_half_nodes(parent_nodes, 1)
    left_interp = interpolation_matrix(basis, parent_nodes, left_nodes)
    right_interp = interpolation_matrix(basis, parent_nodes, right_nodes)

    nchildren = 2^ndim_int
    p2c = Array{Float64,4}(undef, norder_int, norder_int, ndim_int, nchildren)

    for ichild in 1:nchildren
        child_bits = ichild - 1

        for d in 1:ndim_int
            bit = (child_bits >> (d - 1)) & 0x1
            p2c[:, :, d, ichild] .= bit == 0 ? left_interp : right_interp
        end
    end

    return p2c
end

function c2p_transform(basis::AbstractBasis, norder, ndim)
    p2c = p2c_transform(basis, norder, ndim)
    c2p = Array{Float64,4}(undef, size(p2c)...)

    for ichild in axes(p2c, 4), d in axes(p2c, 3)
        c2p[:, :, d, ichild] .= transpose(p2c[:, :, d, ichild])
    end

    return c2p
end

# Workspace helpers for tensor_product_apply! (used by the tree builder).
function _rect_tensor_scratch_cols(from_order::Int, to_order::Int, ndim::Int)
    ndim == 1 && return max(from_order, to_order)
    return maximum(to_order^dim * from_order^(ndim - dim) for dim in 1:(ndim - 1))
end
function _rect_tensor_middle_rows(from_order::Int, to_order::Int, ndim::Int)
    ndim <= 2 && return 0
    return maximum(to_order^(dim - 1) * from_order^(ndim - dim) for dim in 2:(ndim - 1))
end
function _rect_tensor_apply_workspace(::Type{T}, nd::Int, from_order::Int, to_order::Int, ndim::Int) where {T}
    scratch_cols = _rect_tensor_scratch_cols(from_order, to_order, ndim)
    middle_rows = _rect_tensor_middle_rows(from_order, to_order, ndim)
    tensor_in_len = ndim == 3 ? from_order^3 : 0
    tensor_out_len = ndim == 3 ? to_order^3 : 0
    ff_rows = ndim == 3 ? from_order^2 : 0
    fft_cols = ndim == 3 ? to_order * from_order : 0
    ff2_rows = ndim == 3 ? to_order^2 : 0

    return (
        scratch_a = Matrix{T}(undef, nd, scratch_cols),
        scratch_b = Matrix{T}(undef, nd, scratch_cols),
        tmp_perm = Matrix{T}(undef, middle_rows, from_order),
        result_2d = Matrix{T}(undef, middle_rows, to_order),
        packed_src = Vector{T}(undef, tensor_in_len),
        packed_dest = Vector{T}(undef, tensor_out_len),
        ff = Matrix{T}(undef, ff_rows, to_order),
        fft = Matrix{T}(undef, from_order, fft_cols),
        ff2 = Matrix{T}(undef, ff2_rows, from_order),
    )
end
function _check_rect_tensor_3d_workspace(workspace, from_order::Int, to_order::Int)
    for name in (:packed_src, :packed_dest, :ff, :fft, :ff2)
        hasproperty(workspace, name) || throw(ArgumentError("3D tensor workspace must provide $name"))
    end

    length(workspace.packed_src) >= from_order^3 ||
        throw(DimensionMismatch("workspace packed_src must have at least $(from_order^3) entries"))
    length(workspace.packed_dest) >= to_order^3 ||
        throw(DimensionMismatch("workspace packed_dest must have at least $(to_order^3) entries"))
    size(workspace.ff, 1) >= from_order^2 ||
        throw(DimensionMismatch("workspace ff must have at least $(from_order^2) rows"))
    size(workspace.ff, 2) >= to_order ||
        throw(DimensionMismatch("workspace ff must have at least $to_order columns"))
    size(workspace.fft, 1) >= from_order ||
        throw(DimensionMismatch("workspace fft must have at least $from_order rows"))
    size(workspace.fft, 2) >= to_order * from_order ||
        throw(DimensionMismatch("workspace fft must have at least $(to_order * from_order) columns"))
    size(workspace.ff2, 1) >= to_order^2 ||
        throw(DimensionMismatch("workspace ff2 must have at least $(to_order^2) rows"))
    size(workspace.ff2, 2) >= from_order ||
        throw(DimensionMismatch("workspace ff2 must have at least $from_order columns"))

    return workspace
end
