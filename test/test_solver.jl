using BoxDMK
using LinearAlgebra
using Test

# u = Σ a exp(-|x-c|²/σ²) and f = -Δu/(4π): the Laplace volume potential of f is u.
struct ManufacturedSource
    centers::Vector{NTuple{3,Float64}}
    sigmas::Vector{Float64}
    amps::Vector{Float64}
end

const _MS = ManufacturedSource([(0.52, 0.47, 0.5), (0.4, 0.58, 0.55)], [0.05, 0.03], [1.0, -0.6])

function _gauss_terms(m::ManufacturedSource, x)
    u = 0.0; f = 0.0
    g = zeros(3)
    for (c, s, a) in zip(m.centers, m.sigmas, m.amps)
        d = (x[1] - c[1], x[2] - c[2], x[3] - c[3])
        r2 = d[1]^2 + d[2]^2 + d[3]^2
        e = a * exp(-r2 / s^2)
        u += e
        f -= (4r2 / s^4 - 6 / s^2) * e / (4pi)
        for k in 1:3
            g[k] -= 2 * d[k] / s^2 * e
        end
    end
    return u, f, g
end

(m::ManufacturedSource)(x) = _gauss_terms(m, x)[2]

function _leaf_points(tree)
    nodes, _ = BoxDMK.nodes_and_weights(tree.basis, tree.norder)
    out = Tuple{Int,Int,NTuple{3,Float64}}[]
    for ibox in 1:size(tree.centers, 2)
        BoxDMK.isleaf(tree, ibox) || continue
        h = tree.boxsize[tree.level[ibox] + 1] / 2
        j = 0
        for k3 in eachindex(nodes), k2 in eachindex(nodes), k1 in eachindex(nodes)
            j += 1
            x = (tree.centers[1, ibox] + h * nodes[k1], tree.centers[2, ibox] + h * nodes[k2],
                 tree.centers[3, ibox] + h * nodes[k3])
            push!(out, (ibox, j, x))
        end
    end
    return out
end

function _manufactured_errors(eps; norder = 8, basis = LegendreBasis(), grad = false)
    tree, fvals = build_tree(_MS, LaplaceKernel(), basis; ndim = 3, norder = norder, eps = eps, boxlen = 1.0)
    res = bdmk(tree, fvals, LaplaceKernel(); eps = eps, grad = grad)
    nu = du = ng = dg = 0.0
    for (ibox, j, x) in _leaf_points(tree)
        u, _, g = _gauss_terms(_MS, x)
        nu += (res.pot[1, j, ibox] - u)^2; du += u^2
        if grad
            for k in 1:3
                ng += (res.grad[1, k, j, ibox] - g[k])^2; dg += g[k]^2
            end
        end
    end
    return sqrt(nu / du), grad ? sqrt(ng / dg) : NaN, tree, fvals, res
end

@testset "Solver" begin
    @testset "Laplace accuracy vs exact solution" begin
        for eps in (1e-3, 1e-6)
            err, gerr, = _manufactured_errors(eps; grad = true)
            @test err < 10 * eps
            @test gerr < 100 * eps
        end
        err, = _manufactured_errors(1e-6; basis = ChebyshevBasis())
        @test err < 1e-5
    end

    @testset "Targets, derivatives and nd" begin
        _, _, tree, fvals, _ = _manufactured_errors(1e-6)
        targets = [0.5 0.31 0.66; 0.45 0.52 0.7; 0.55 0.6 0.41]
        res = bdmk(tree, fvals, LaplaceKernel(); eps = 1e-6, grad = true, hess = true, targets = targets)
        @test size(res.target_pot) == (1, 3)
        @test size(res.target_grad) == (1, 3, 3)
        @test size(res.target_hess) == (1, 6, 3)
        gscale = maximum(abs, res.grad)
        for t in 1:3
            u, _, g = _gauss_terms(_MS, targets[:, t])
            @test abs(res.target_pot[1, t] - u) < 1e-5 * maximum(abs, res.pot)
            @test norm(res.target_grad[1, :, t] - g) < 1e-4 * gscale
        end
        # Hessian trace = Δu = -4π f
        for t in 1:3
            _, f, _ = _gauss_terms(_MS, targets[:, t])
            tr = res.target_hess[1, 1, t] + res.target_hess[1, 2, t] + res.target_hess[1, 3, t]
            @test abs(tr + 4pi * f) < 1e-3 * maximum(abs, fvals) * 4pi
        end
        # agrees with PotentialField evaluation
        field = PotentialField(tree, res)
        @test evaluate_potential(field, targets) ≈ res.target_pot rtol = 1e-12

        # several densities at once give the same results as one at a time
        # (potential-only solves; with derivatives the near field takes
        # Fortran's sparse path, which rounds differently)
        f2 = cat(fvals, 2 .* fvals; dims = 1)
        res1 = bdmk(tree, fvals, LaplaceKernel(); eps = 1e-6)
        res2 = bdmk(tree, f2, LaplaceKernel(); eps = 1e-6)
        @test res2.pot[1:1, :, :] ≈ res1.pot rtol = 1e-14
        @test res2.pot[2:2, :, :] ≈ 2 .* res1.pot rtol = 1e-14
    end

    @testset "Kernels" begin
        f(x) = exp(-200 * ((x[1] - 0.5)^2 + (x[2] - 0.5)^2 + (x[3] - 0.5)^2))
        for kernel in (YukawaKernel(5.0), SqrtLaplaceKernel())
            tree, fvals = build_tree(f, kernel, LegendreBasis(); ndim = 3, norder = 8, eps = 1e-6, boxlen = 1.0)
            r1 = bdmk(tree, fvals, kernel; eps = 1e-3, targets = [0.5, 0.5, 0.5])
            r2 = bdmk(tree, fvals, kernel; eps = 1e-6, targets = [0.5, 0.5, 0.5])
            @test all(isfinite, r2.pot)
            @test abs(r1.target_pot[1] - r2.target_pot[1]) < 1e-2 * abs(r2.target_pot[1])
        end
        # Yukawa with tiny beta approaches Laplace
        tree, fvals = build_tree(f, LaplaceKernel(), LegendreBasis(); ndim = 3, norder = 8, eps = 1e-6, boxlen = 1.0)
        rl = bdmk(tree, fvals, LaplaceKernel(); eps = 1e-6, targets = [0.5, 0.5, 0.5])
        ry = bdmk(tree, fvals, YukawaKernel(1e-3); eps = 1e-6, targets = [0.5, 0.5, 0.5])
        @test isapprox(ry.target_pot[1], rl.target_pot[1]; rtol = 1e-2)
    end

    @testset "Timings" begin
        _, _, tree, fvals, _ = _manufactured_errors(1e-3; norder = 6)
        tm = zeros(10)
        bdmk(tree, fvals, LaplaceKernel(); eps = 1e-3, timings = tm)
        @test all(>=(0), tm) && sum(tm) > 0
    end

    # Parity with the Fortran reference (optional: needs the vendored library).
    if isfile(BoxDMK._resolve_fortran_solve_library_path())
        @testset "Fortran parity" begin
            _, _, tree, fvals, _ = _manufactured_errors(1e-6)
            for (kernel, grad, hess) in ((LaplaceKernel(), true, true), (YukawaKernel(3.0), false, false),
                                         (SqrtLaplaceKernel(), true, false))
                rn = bdmk(tree, fvals, kernel; eps = 1e-6, grad = grad, hess = hess)
                rf = bdmk_fortran(tree, fvals, kernel; eps = 1e-6, grad = grad, hess = hess)
                @test norm(rn.pot - rf.pot) / norm(rf.pot) < 1e-11
                grad && @test norm(rn.grad - rf.grad) / norm(rf.grad) < 1e-11
                hess && @test norm(rn.hess - rf.hess) / norm(rf.hess) < 1e-11
            end
        end
    else
        @info "Fortran reference library not built; skipping parity tests"
    end
end
