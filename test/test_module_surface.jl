using BoxDMK
using Test

@testset "Module Surface" begin
    for name in (:LaplaceKernel, :YukawaKernel, :SqrtLaplaceKernel, :LegendreBasis, :ChebyshevBasis,
                 :build_tree, :bdmk, :bdmk_native, :PotentialField, :evaluate_potential,
                 :evaluate_gradient, :evaluate_hessian, :build_tree_fortran, :bdmk_fortran)
        @test name in names(BoxDMK)
    end
    @test isdefined(BoxDMK, :SolverResult)
    @test isdefined(BoxDMK, :FortranTreeData)
    # the solver must not need the Fortran library
    @test BoxDMK.bdmk === bdmk
end
