using BoxDMK
using Test

@testset "Tree Construction" begin
    f_smooth(x) = [sin(pi * x[1]) * cos(pi * x[2]) * exp(x[3])]
    tree, fvals = build_tree(
        f_smooth,
        LaplaceKernel(),
        LegendreBasis();
        ndim = 3,
        norder = 6,
        eps = 1e-6,
        boxlen = 1.0,
        nd = 1,
    )

    @test tree.nlevels >= 0
    @test size(tree.centers, 2) > 0
    @test all(tree.level .>= 0)
    @test size(fvals) == (1, 6^3, size(tree.centers, 2))

    f_sharp(x) = [exp(-1000 * sum((x .- 0.5) .^ 2))]
    tree2, fvals2 = build_tree(
        f_sharp,
        LaplaceKernel(),
        LegendreBasis();
        ndim = 3,
        norder = 6,
        eps = 1e-6,
        boxlen = 1.0,
        nd = 1,
    )

    @test tree2.nlevels > tree.nlevels
    @test size(fvals2) == (1, 6^3, size(tree2.centers, 2))

    for ibox in 1:size(tree2.centers, 2)
        for jbox in tree2.colleagues[ibox]
            @test abs(tree2.level[ibox] - tree2.level[jbox]) <= 1
        end
    end
end

@testset "Interaction Lists" begin
    f(x) = [exp(-100 * sum((x .- 0.5) .^ 2))]
    tree, _ = build_tree(
        f,
        LaplaceKernel(),
        LegendreBasis();
        ndim = 3,
        norder = 6,
        eps = 1e-6,
        boxlen = 1.0,
        nd = 1,
    )
    lists = BoxDMK.build_interaction_lists(tree)
    nboxes = size(tree.centers, 2)
    @test length(lists.list1) == nboxes
    @test length(lists.listpw) == nboxes

    function touches(tree, ibox, jbox)
        size_i = tree.boxsize[tree.level[ibox] + 1]
        size_j = tree.boxsize[tree.level[jbox] + 1]
        tol = 128 * eps(Float64) * max(size_i, size_j, 1.0)

        for d in 1:tree.ndim
            if abs(tree.centers[d, ibox] - tree.centers[d, jbox]) > (size_i + size_j) / 2 + tol
                return false
            end
        end

        return true
    end

    for ibox in 1:nboxes
        @test length(unique(lists.list1[ibox])) == length(lists.list1[ibox])
        @test length(unique(lists.listpw[ibox])) == length(lists.listpw[ibox])

        if BoxDMK.isleaf(tree, ibox)
            for jbox in lists.list1[ibox]
                @test BoxDMK.isleaf(tree, jbox)
                @test touches(tree, ibox, jbox)
            end
        else
            @test isempty(lists.list1[ibox])
        end

        for jbox in lists.listpw[ibox]
            @test tree.level[ibox] == tree.level[jbox]
            @test jbox != ibox
            @test jbox in tree.colleagues[ibox]
            @test !BoxDMK.isleaf(tree, ibox) || !BoxDMK.isleaf(tree, jbox)
        end
    end
end

# The Fortran tree builder calls the Julia source function from inside an
# OpenMP parallel region, which can deadlock against Julia's GC on many-core
# machines. Opt in with BOXDMK_TEST_FORTRAN_TREE=1 (and OMP_NUM_THREADS=1).
if get(ENV, "BOXDMK_TEST_FORTRAN_TREE", "0") == "1"
@testset "Exact Fortran Tree Parity" begin
    f(x) = [exp(-40 * sum((x .- 0.5) .^ 2))]

    function check_exact_parity(kernel, basis; ndim)
        tree, fvals = build_tree(
            f,
            kernel,
            basis;
            ndim = ndim,
            norder = 4,
            eps = 1e-3,
            boxlen = 1.0,
            nd = 1,
            eta = 1.0,
        )
        ftree = build_tree_fortran(
            f,
            kernel,
            basis;
            ndim = ndim,
            norder = 4,
            eps = 1e-3,
            boxlen = 1.0,
            nd = 1,
            eta = 1.0,
        )

        @test tree.nlevels == ftree.tree.nlevels
        @test tree.level == ftree.tree.level
        @test tree.parent == ftree.tree.parent
        @test tree.children == ftree.tree.children
        @test tree.colleagues == ftree.tree.colleagues
        @test tree.centers ≈ ftree.tree.centers atol = 1e-14
        @test tree.boxsize ≈ ftree.tree.boxsize atol = 1e-14
        @test fvals ≈ ftree.fvals atol = 1e-12
    end

    check_exact_parity(LaplaceKernel(), LegendreBasis(); ndim = 3)
    check_exact_parity(YukawaKernel(1.0), LegendreBasis(); ndim = 3)
    check_exact_parity(SqrtLaplaceKernel(), LegendreBasis(); ndim = 3)
    check_exact_parity(LaplaceKernel(), ChebyshevBasis(); ndim = 3)
    check_exact_parity(LaplaceKernel(), LegendreBasis(); ndim = 1)
    check_exact_parity(LaplaceKernel(), LegendreBasis(); ndim = 2)
end
end

@testset "Tree Matches Fortran Benchmark" begin
    rsig = 1e-4
    rsign = rsig^(3 / 2)

    function analytic_rhs_scalar(x)
        c1 = [0.1, 0.02, 0.04]
        c2 = [0.03, -0.1, 0.05]
        s1 = rsig
        s2 = rsig / 2
        str1 = 1.0 / (π * rsign)
        str2 = -0.5 / (π * rsign)
        rr1 = sum((x .- c1) .^ 2)
        rr2 = sum((x .- c2) .^ 2)
        return str1 * exp(-rr1 / s1) * (-6 + 4 * rr1 / s1) / s1 +
               str2 * exp(-rr2 / s2) * (-6 + 4 * rr2 / s2) / s2
    end

    function benchmark_rhs(boxlen)
        shift = fill(Float64(boxlen) / 2, 3)
        return x -> [analytic_rhs_scalar(x .- shift)]
    end

    tree, fvals = build_tree(
        benchmark_rhs(1.18),
        LaplaceKernel(),
        LegendreBasis();
        ndim = 3,
        norder = 16,
        eps = 5e-4,
        boxlen = 1.18,
        nd = 1,
        eta = 0.0,
    )

    @test tree.nlevels == 6
    @test size(tree.centers, 2) == 1129
    @test size(fvals) == (1, 16^3, 1129)
    @test all(0.0 .<= tree.centers .<= 1.18)
end
