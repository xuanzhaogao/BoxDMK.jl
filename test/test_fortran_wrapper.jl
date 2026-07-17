using BoxDMK
using Test

_fortran_wrapper_source(x) = [exp(-40 * sum((x .- 0.5) .^ 2))]

@testset "Fortran Wrapper" begin
    @test :build_tree_fortran in names(BoxDMK)
    @test :bdmk_fortran in names(BoxDMK)
    @test :FortranTreeData in names(BoxDMK)

    kernel = LaplaceKernel()
    basis = LegendreBasis()
    eps_val = 1e-2

    # Julia tree build + Fortran solve (the default supported path).
    tree, fvals = build_tree(
        _fortran_wrapper_source,
        kernel,
        basis;
        ndim = 3,
        norder = 4,
        eps = eps_val,
        boxlen = 1.0,
        nd = 1,
        eta = 1.0,
    )

    @test all(0.0 .<= tree.centers .<= 1.0)

    result = bdmk_fortran(tree, fvals, kernel; eps = eps_val, targets = [0.5, 0.5, 0.5])
    @test result isa BoxDMK.SolverResult
    @test size(result.pot) == size(fvals)
    @test result.target_pot !== nothing
    @test size(result.target_pot) == (1, 1)
    @test maximum(abs.(result.pot)) > 0
end

# The Fortran tree builder calls the Julia source function from inside an
# OpenMP parallel region, which can deadlock against Julia's GC on many-core
# machines. Opt in with BOXDMK_TEST_FORTRAN_TREE=1 (and OMP_NUM_THREADS=1).
if get(ENV, "BOXDMK_TEST_FORTRAN_TREE", "0") == "1"
@testset "Fortran Tree Builder Wrapper" begin
    kernel = LaplaceKernel()
    basis = LegendreBasis()
    eps_val = 1e-2

    ftree = build_tree_fortran(
        _fortran_wrapper_source,
        kernel,
        basis;
        ndim = 3,
        norder = 4,
        eps = eps_val,
        boxlen = 1.0,
        nd = 1,
        eta = 1.0,
    )

    @test ftree isa BoxDMK.FortranTreeData
    @test ftree.tree isa BoxDMK.BoxTree
    @test size(ftree.fvals) == (1, 4^3, BoxDMK.nboxes(ftree.tree))
    @test all(0.0 .<= ftree.tree.centers .<= 1.0)

    tree, fvals = ftree
    @test tree === ftree.tree
    @test fvals === ftree.fvals

    result = bdmk_fortran(ftree, kernel; eps = eps_val, targets = [0.5, 0.5, 0.5])
    @test result isa BoxDMK.SolverResult
    @test size(result.pot) == size(ftree.fvals)
    @test result.target_pot !== nothing
    @test size(result.target_pot) == (1, 1)
    @test maximum(abs.(result.pot)) > 0

    result2 = bdmk_fortran(tree, fvals, kernel; eps = eps_val)
    @test result2.pot ≈ result.pot
end
end
