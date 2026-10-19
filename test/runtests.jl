using BoxDMK
using Test
using LinearAlgebra

@testset "BoxDMK.jl" begin
    include("test_module_surface.jl")
    include("test_types.jl")
    include("test_basis.jl")
    include("test_tensor.jl")
    include("test_tree.jl")
    include("test_solver.jl")
    include("test_field.jl")
    include("test_periodic.jl")
    include("test_q2d.jl")
    if isfile(BoxDMK._resolve_fortran_library_path())
        include("test_fortran_wrapper.jl")
    else
        @info "Fortran reference library not built; skipping wrapper tests"
    end
end
