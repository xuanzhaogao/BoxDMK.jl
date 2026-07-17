"""
    PotentialField(tree, result)

Attach a `SolverResult` to its `BoxTree` so the solved potential (and, when
computed, gradient/hessian) can be evaluated at arbitrary points of the
domain `[0, boxlen]^ndim`.

The field is callable: `field(x)` evaluates the potential at a point `x`
(returns a scalar for `nd == 1`, a length-`nd` vector otherwise), and
`field(X)` evaluates at each column of a `(ndim, ntarg)` matrix. Use
[`evaluate_potential`](@ref), [`evaluate_gradient`](@ref), and
[`evaluate_hessian`](@ref) for output with explicit `(nd, ...)` shapes.

Evaluation locates the leaf box containing the point by descending the tree,
then evaluates the box's tensor-product polynomial interpolant.
"""
struct PotentialField{T<:Real, B<:AbstractBasis, TP<:Real}
    tree::BoxTree{T,B}
    pot::Array{TP,3}
    grad::Union{Nothing, Array{TP,4}}
    hess::Union{Nothing, Array{TP,4}}
    nodes::Vector{Float64}        # cached 1D basis nodes on [-1, 1]
end

function PotentialField(tree::BoxTree, result::SolverResult)
    np = npbox(tree.norder, tree.ndim)
    size(result.pot, 2) == np || throw(DimensionMismatch("result.pot must have size (_, $np, _)"))
    size(result.pot, 3) == nboxes(tree) || throw(DimensionMismatch("result.pot must have size (_, _, $(nboxes(tree)))"))

    nodes, _ = nodes_and_weights(tree.basis, tree.norder)
    return PotentialField(tree, result.pot, result.grad, result.hess, nodes)
end

ndensities(field::PotentialField) = size(field.pot, 1)

# Locate the leaf box containing `point` by descending from the root. Boxes
# subdivide fully into 2^ndim children, so the child with the nearest center
# is the one containing the point (ties on shared faces are equivalent).
function _locate_leaf(tree::BoxTree, point::AbstractVector{<:Real})
    length(point) == tree.ndim || throw(DimensionMismatch("point must have length $(tree.ndim)"))
    _box_contains_point(tree, 1, point) || throw(ArgumentError("point $point lies outside the tree domain"))

    ibox = 1
    while !isleaf(tree, ibox)
        best = 0
        best_dist = Inf

        for child in @view tree.children[:, ibox]
            child == 0 && continue

            dist = 0.0
            for dim in 1:tree.ndim
                delta = Float64(point[dim]) - tree.centers[dim, child]
                dist += delta * delta
            end

            if dist < best_dist
                best = child
                best_dist = dist
            end
        end

        ibox = best
    end

    return ibox
end

function _field_interpolation_rows(field::PotentialField, ibox::Int, point::AbstractVector{<:Real})
    tree = field.tree
    local_point = _target_local_coordinates(tree, ibox, point)
    return ntuple(dim -> vec(interpolation_matrix(tree.basis, field.nodes, [local_point[dim]])), tree.ndim)
end

"""
    evaluate_potential(field, x) -> Vector (nd,)
    evaluate_potential(field, X) -> Matrix (nd, ntarg)

Evaluate the potential at a point `x` or at each column of `X`.
"""
function evaluate_potential(field::PotentialField, point::AbstractVector{<:Real})
    ibox = _locate_leaf(field.tree, point)
    rows = _field_interpolation_rows(field, ibox, point)

    out = Vector{eltype(field.pot)}(undef, ndensities(field))
    for idensity in eachindex(out)
        out[idensity] = _evaluate_tensor_values(@view(field.pot[idensity, :, ibox]), rows)
    end
    return out
end

function evaluate_potential(field::PotentialField, points::AbstractMatrix{<:Real})
    target_points = _normalize_targets(points, field.tree)
    out = Matrix{eltype(field.pot)}(undef, ndensities(field), size(target_points, 2))
    for itarg in axes(target_points, 2)
        out[:, itarg] = evaluate_potential(field, @view(target_points[:, itarg]))
    end
    return out
end

function _require_field_component(values, name::String, keyword::String)
    values === nothing && throw(ArgumentError("field has no $name; run bdmk with $keyword = true"))
    return values
end

"""
    evaluate_gradient(field, x) -> Matrix (nd, ndim)
    evaluate_gradient(field, X) -> Array (nd, ndim, ntarg)

Evaluate the potential gradient. Requires the solve to have run with
`grad = true`.
"""
function evaluate_gradient(field::PotentialField, point::AbstractVector{<:Real})
    grad = _require_field_component(field.grad, "gradient data", "grad")
    ibox = _locate_leaf(field.tree, point)
    rows = _field_interpolation_rows(field, ibox, point)

    out = Matrix{eltype(grad)}(undef, ndensities(field), field.tree.ndim)
    for idensity in axes(out, 1), dim in axes(out, 2)
        out[idensity, dim] = _evaluate_tensor_values(@view(grad[idensity, dim, :, ibox]), rows)
    end
    return out
end

function evaluate_gradient(field::PotentialField, points::AbstractMatrix{<:Real})
    grad = _require_field_component(field.grad, "gradient data", "grad")
    target_points = _normalize_targets(points, field.tree)
    out = Array{eltype(grad),3}(undef, ndensities(field), field.tree.ndim, size(target_points, 2))
    for itarg in axes(target_points, 2)
        out[:, :, itarg] = evaluate_gradient(field, @view(target_points[:, itarg]))
    end
    return out
end

"""
    evaluate_hessian(field, x) -> Matrix (nd, nhess)
    evaluate_hessian(field, X) -> Array (nd, nhess, ntarg)

Evaluate the potential hessian components (xx, yy, zz, xy, xz, yz in 3D).
Requires the solve to have run with `hess = true`.
"""
function evaluate_hessian(field::PotentialField, point::AbstractVector{<:Real})
    hess = _require_field_component(field.hess, "hessian data", "hess")
    ibox = _locate_leaf(field.tree, point)
    rows = _field_interpolation_rows(field, ibox, point)

    out = Matrix{eltype(hess)}(undef, ndensities(field), nhess(field.tree.ndim))
    for idensity in axes(out, 1), component in axes(out, 2)
        out[idensity, component] = _evaluate_tensor_values(@view(hess[idensity, component, :, ibox]), rows)
    end
    return out
end

function evaluate_hessian(field::PotentialField, points::AbstractMatrix{<:Real})
    hess = _require_field_component(field.hess, "hessian data", "hess")
    target_points = _normalize_targets(points, field.tree)
    out = Array{eltype(hess),3}(undef, ndensities(field), nhess(field.tree.ndim), size(target_points, 2))
    for itarg in axes(target_points, 2)
        out[:, :, itarg] = evaluate_hessian(field, @view(target_points[:, itarg]))
    end
    return out
end

function (field::PotentialField)(point::AbstractVector{<:Real})
    values = evaluate_potential(field, point)
    return ndensities(field) == 1 ? values[1] : values
end

function (field::PotentialField)(points::AbstractMatrix{<:Real})
    values = evaluate_potential(field, points)
    return ndensities(field) == 1 ? vec(values) : values
end
