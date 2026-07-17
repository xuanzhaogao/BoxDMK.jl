using BoxDMK
using Test
using LinearAlgebra

@testset "BoxDMK.jl" begin
    @info "Running tests for BoxDMK.jl"
    include("test_module_surface.jl")
    @info "Finished testing module surface"
    include("test_types.jl")
    @info "Finished testing types"
    include("test_basis.jl")
    @info "Finished testing basis"
    include("test_tensor.jl")
    @info "Finished testing tensor"
    include("test_tree.jl")
    @info "Finished testing tree"
    include("test_sog.jl")
    @info "Finished testing SOG"
    include("test_kernels.jl")
    @info "Finished testing kernels"
    include("test_proxy.jl")
    @info "Finished testing proxy"
    include("test_tree_data.jl")
    @info "Finished testing tree data"
    include("test_local.jl")
    @info "Finished testing local"
    include("test_planewave.jl")
    @info "Finished testing planewave"
    include("test_passes.jl")
    @info "Finished testing passes"
    include("test_boxfgt.jl")
    @info "Finished testing boxfgt"
    include("test_derivatives.jl")
    @info "Finished testing derivatives"
    include("test_solver.jl")
    @info "Finished testing solver"
    include("test_field.jl")
    @info "Finished testing field"
    include("test_cross_validation.jl")
    @info "Finished testing cross validation"
    include("test_fortran_wrapper.jl")
    @info "Finished testing Fortran wrapper"
    include("test_hybrid_parity.jl")
    @info "Finished testing hybrid parity"
end
