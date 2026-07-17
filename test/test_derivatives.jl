using BoxDMK
using Test

function derivative_box_coordinates(tree::BoxDMK.BoxTree, ibox::Integer)
    nodes, _ = BoxDMK.nodes_and_weights(tree.basis, tree.norder)
    np = length(nodes)^tree.ndim
    coords = Matrix{Float64}(undef, tree.ndim, np)
    halfsize = tree.boxsize[tree.level[ibox] + 1] / 2
    idx = 1

    for point in Iterators.product(ntuple(_ -> nodes, tree.ndim)...)
        for d in 1:tree.ndim
            coords[d, idx] = tree.centers[d, ibox] + halfsize * point[d]
        end
        idx += 1
    end

    return coords
end

@testset "Potential Derivatives" begin
    basis = LegendreBasis()

    f_trilinear(x) = [x[1] * x[2] * x[3]]
    tree_grad, pot_grad = build_tree(
        f_trilinear,
        LaplaceKernel(),
        basis;
        ndim = 3,
        norder = 6,
        eps = 1e-6,
        boxlen = 1.0,
        nd = 1,
    )

    grad = Array{Float64}(undef, 1, tree_grad.ndim, size(pot_grad, 2), size(pot_grad, 3))
    BoxDMK.compute_gradient!(grad, pot_grad, tree_grad, basis)

    for ibox in BoxDMK.leaves(tree_grad)
        coords = derivative_box_coordinates(tree_grad, ibox)
        @test grad[1, 1, :, ibox] ≈ coords[2, :] .* coords[3, :] atol = 1e-10
        @test grad[1, 2, :, ibox] ≈ coords[1, :] .* coords[3, :] atol = 1e-10
        @test grad[1, 3, :, ibox] ≈ coords[1, :] .* coords[2, :] atol = 1e-10
    end

    f_quadratic(x) = [x[1]^2 + 2 * x[2]^2 + 3 * x[3]^2]
    tree_hess, pot_hess = build_tree(
        f_quadratic,
        LaplaceKernel(),
        basis;
        ndim = 3,
        norder = 6,
        eps = 1e-6,
        boxlen = 1.0,
        nd = 1,
    )

    hess = Array{Float64}(undef, 1, BoxDMK.nhess(tree_hess.ndim), size(pot_hess, 2), size(pot_hess, 3))
    BoxDMK.compute_hessian!(hess, pot_hess, tree_hess, basis)

    # atol is a vector-norm bound over npbox entries; per-entry roundoff of
    # ~1e-11 from the spectral second-derivative stack pushes the norm past
    # 1e-10 on some BLAS/FP configurations, so allow 1e-8.
    for ibox in BoxDMK.leaves(tree_hess)
        @test hess[1, 1, :, ibox] ≈ fill(2.0, size(hess, 3)) atol = 1e-8
        @test hess[1, 2, :, ibox] ≈ fill(4.0, size(hess, 3)) atol = 1e-8
        @test hess[1, 3, :, ibox] ≈ fill(6.0, size(hess, 3)) atol = 1e-8
        @test hess[1, 4, :, ibox] ≈ fill(0.0, size(hess, 3)) atol = 1e-8
        @test hess[1, 5, :, ibox] ≈ fill(0.0, size(hess, 3)) atol = 1e-8
        @test hess[1, 6, :, ibox] ≈ fill(0.0, size(hess, 3)) atol = 1e-8
    end
end
