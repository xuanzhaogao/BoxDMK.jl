using BoxDMK
using Test
using LinearAlgebra

function _field_test_setup(; grad = true, hess = true)
    f(x) = exp(-100 * ((x[1] - 0.5)^2 + (x[2] - 0.5)^2 + (x[3] - 0.5)^2))
    tree, fvals = build_tree(
        f,
        LaplaceKernel(),
        LegendreBasis();
        ndim = 3,
        norder = 6,
        eps = 1e-3,
        boxlen = 1.0,
        nd = 1,
    )

    # Columns 1-4 are interior to leaf boxes; columns 5-6 lie on box
    # boundaries (domain center / face-aligned), where the leaf choice is
    # ambiguous and grad/hess only agree to solver accuracy.
    targets = [
        0.31  0.72  0.05  0.95  0.5   0.5;
        0.64  0.28  0.05  0.95  0.5   0.25;
        0.47  0.81  0.05  0.95  0.5   0.75;
    ]

    result = bdmk(tree, fvals, LaplaceKernel(); eps = 1e-3, grad = grad, hess = hess, targets = targets)
    return tree, result, targets
end

@testset "PotentialField" begin
    tree, result, targets = _field_test_setup()
    field = PotentialField(tree, result)
    interior = 1:4
    boundary = 5:6

    @testset "matches bdmk target evaluation (interior points)" begin
        for itarg in interior
            x = targets[:, itarg]

            pot = evaluate_potential(field, x)
            @test pot isa Vector{Float64}
            @test length(pot) == 1
            @test pot[1] ≈ result.target_pot[1, itarg] rtol = 1e-12

            g = evaluate_gradient(field, x)
            @test size(g) == (1, 3)
            @test vec(g) ≈ result.target_grad[1, :, itarg] rtol = 1e-12

            h = evaluate_hessian(field, x)
            @test size(h) == (1, 6)
            @test vec(h) ≈ result.target_hess[1, :, itarg] rtol = 1e-12
        end
    end

    @testset "box-boundary points agree to solver accuracy" begin
        grad_scale = maximum(abs.(result.target_grad))
        hess_scale = maximum(abs.(result.target_hess))

        for itarg in boundary
            x = targets[:, itarg]
            @test evaluate_potential(field, x)[1] ≈ result.target_pot[1, itarg] rtol = 1e-6
            @test vec(evaluate_gradient(field, x)) ≈ result.target_grad[1, :, itarg] atol = 1e-2 * grad_scale
            @test vec(evaluate_hessian(field, x)) ≈ result.target_hess[1, :, itarg] atol = 1e-2 * hess_scale
        end
    end

    @testset "callable handle" begin
        x = [0.31, 0.64, 0.47]
        @test field(x) isa Float64
        @test field(x) ≈ result.target_pot[1, 1] rtol = 1e-12

        batch = field(targets)
        @test batch isa Vector{Float64}
        @test batch ≈ vec(result.target_pot) rtol = 1e-6
    end

    @testset "batch evaluation is consistent with per-point evaluation" begin
        pots = evaluate_potential(field, targets)
        @test size(pots) == (1, size(targets, 2))

        grads = evaluate_gradient(field, targets)
        @test size(grads) == (1, 3, size(targets, 2))

        hesses = evaluate_hessian(field, targets)
        @test size(hesses) == (1, 6, size(targets, 2))

        for itarg in axes(targets, 2)
            x = targets[:, itarg]
            @test pots[:, itarg] == evaluate_potential(field, x)
            @test grads[:, :, itarg] == evaluate_gradient(field, x)
            @test hesses[:, :, itarg] == evaluate_hessian(field, x)
        end
    end

    @testset "domain boundary and corners" begin
        for x in ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.5, 1.0])
            @test evaluate_potential(field, x) isa Vector{Float64}
        end
    end

    @testset "outside domain throws" begin
        @test_throws ArgumentError field([1.5, 0.5, 0.5])
        @test_throws ArgumentError evaluate_potential(field, [-0.1, 0.5, 0.5])
    end

    @testset "missing grad/hess throws" begin
        tree2, result2, _ = _field_test_setup(grad = false, hess = false)
        field2 = PotentialField(tree2, result2)

        @test evaluate_potential(field2, [0.5, 0.5, 0.5]) isa Vector{Float64}
        @test_throws ArgumentError evaluate_gradient(field2, [0.5, 0.5, 0.5])
        @test_throws ArgumentError evaluate_hessian(field2, [0.5, 0.5, 0.5])
    end

    @testset "interpolation reproduces grid values" begin
        ibox = first(ib for ib in BoxDMK.leaves(tree))
        nodes, _ = BoxDMK.nodes_and_weights(tree.basis, tree.norder)
        halfsize = tree.boxsize[tree.level[ibox] + 1] / 2

        ipoint = 1
        x = [tree.centers[d, ibox] + halfsize * BoxDMK._reference_grid(nodes, 3)[d, ipoint] for d in 1:3]
        @test field(x) ≈ result.pot[1, ipoint, ibox] rtol = 1e-10
    end
end
