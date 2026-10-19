function _check_solver_inputs(tree::BoxTree, fvals)
    nd = size(fvals, 1)
    np = npbox(tree.norder, tree.ndim)
    size(fvals, 2) == np || throw(DimensionMismatch("fvals must have size (_, $np, _)"))
    size(fvals, 3) == nboxes(tree) || throw(DimensionMismatch("fvals must have size (_, _, $(nboxes(tree)))"))
    return nd, np
end

function _normalize_targets(targets, tree::BoxTree)
    if targets isa AbstractVector
        length(targets) == tree.ndim || throw(DimensionMismatch("target vector must have length $(tree.ndim)"))
        return reshape(Float64.(collect(targets)), tree.ndim, 1)
    elseif targets isa AbstractMatrix
        if size(targets, 1) == tree.ndim
            return Float64.(targets)
        elseif size(targets, 2) == tree.ndim
            return permutedims(Float64.(targets))
        end
    end

    throw(DimensionMismatch("targets must be an AbstractVector of length $(tree.ndim) or an AbstractMatrix with one dimension equal to $(tree.ndim)"))
end

function _box_contains_point(tree::BoxTree, ibox::Int, point::AbstractVector{<:Real})
    halfsize = tree.boxsize[tree.level[ibox] + 1] / 2
    tol = 128 * eps(Float64) * max(tree.boxsize[1], 1.0)

    for dim in 1:tree.ndim
        abs(point[dim] - tree.centers[dim, ibox]) <= halfsize + tol || return false
    end

    return true
end

function _target_local_coordinates(tree::BoxTree, ibox::Int, point::AbstractVector{<:Real})
    boxsize = tree.boxsize[tree.level[ibox] + 1]
    return [2 * (Float64(point[dim]) - Float64(tree.centers[dim, ibox])) / Float64(boxsize) for dim in 1:tree.ndim]
end

function _evaluate_tensor_values(values, rows)
    n = length(first(rows))
    tensor = reshape(values, ntuple(_ -> n, length(rows))...)
    accum = zero(eltype(values))

    for idx in CartesianIndices(tensor)
        weight = one(eltype(values))

        for dim in 1:length(rows)
            weight *= rows[dim][idx[dim]]
        end

        accum += weight * tensor[idx]
    end

    return accum
end

"""
    bdmk(tree, fvals, kernel; eps=1e-6, grad=false, hess=false, targets=nothing, timings=nothing)

Volume potential of the density `fvals` (shape `(nd, norder^3, nboxes)`) on
`tree`, computed by the pure-Julia box-DMK solver. Returns a `SolverResult`.
"""
function bdmk(
    tree::BoxTree,
    fvals::AbstractArray{<:Real,3},
    kernel::AbstractKernel;
    eps = 1e-6,
    grad = false,
    hess = false,
    targets = nothing,
    timings = nothing,
)
    eps_value = Float64(eps)
    eps_value > 0 || throw(ArgumentError("eps must be positive"))
    return bdmk_native(tree, fvals, kernel; eps = eps_value, grad = Bool(grad), hess = Bool(hess),
                       targets = targets, timings = timings)
end
