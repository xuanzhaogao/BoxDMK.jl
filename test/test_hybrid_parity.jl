using BoxDMK
using Test

_hybrid_debug_source(x) = [exp(-40 * sum((x .- 0.5) .^ 2))]

const _HYBRID_REFERENCE_TARGETS_PHYSICAL = [
    -0.32  -0.18   0.00   0.12   0.24  -0.28   0.30   0.08;
    -0.10   0.14  -0.22   0.20  -0.24   0.32   0.06  -0.30;
     0.18  -0.26   0.28  -0.12   0.04   0.10  -0.16   0.22;
]

const _HYBRID_REFERENCE_RSIG = 1e-4

function _hybrid_reference_analytic_rhs_scalar(x::AbstractVector{<:Real})
    centers = (
        (0.1, 0.02, 0.04),
        (0.03, -0.1, 0.05),
    )
    sigmas = (_HYBRID_REFERENCE_RSIG, _HYBRID_REFERENCE_RSIG / 2)
    strengths = (
        1.0 / (π * (_HYBRID_REFERENCE_RSIG * 1.0)^(3 / 2)),
        -0.5 / (π * (_HYBRID_REFERENCE_RSIG * 1.0)^(3 / 2)),
    )

    value = 0.0
    @inbounds for k in eachindex(centers)
        rr = 0.0
        for d in 1:3
            delta = Float64(x[d]) - centers[k][d]
            rr += delta * delta
        end
        sigma = sigmas[k]
        value += strengths[k] * exp(-rr / sigma) * (-6.0 + 4.0 * rr / sigma) / sigma
    end

    return value
end

function _hybrid_reference_shifted_rhs(boxlen::Real)
    shift = fill(Float64(boxlen) / 2, 3)
    return x -> [_hybrid_reference_analytic_rhs_scalar(x .- shift)]
end

function _hybrid_reference_case_inputs()
    boxlen = 1.18
    tree, fvals = build_tree(
        _hybrid_reference_shifted_rhs(boxlen),
        LaplaceKernel(),
        LegendreBasis();
        ndim = 3,
        norder = 16,
        eps = 5e-4,
        boxlen = boxlen,
        nd = 1,
        eta = 0.0,
    )
    targets = _HYBRID_REFERENCE_TARGETS_PHYSICAL .+ fill(boxlen / 2, 3)
    return tree, fvals, targets
end

@testset "Hybrid Parity Surface" begin
    @test isdefined(BoxDMK, :_resolve_fortran_library_path)
    @test isdefined(BoxDMK, :_vendored_fortran_root)
    @test isdefined(BoxDMK, :_fortran_library_candidates)
    @test isdefined(BoxDMK, :_resolve_fortran_solve_library_path)
    @test isdefined(BoxDMK, :_fortran_solve_library_candidates)

    vendored_root = BoxDMK._vendored_fortran_root()
    @test isdir(vendored_root)
    @test isfile(joinpath(vendored_root, "README.md"))
    @test isfile(joinpath(vendored_root, "src", "bdmk", "bdmk4.f"))
    @test isfile(joinpath(vendored_root, "src", "bdmk", "bdmk_c_api.f90"))

    build_script = normpath(joinpath(@__DIR__, "..", "deps", "build_fortran_ref.jl"))
    @test isfile(build_script)

    candidates = BoxDMK._fortran_library_candidates()
    @test !isempty(candidates)
    @test occursin("deps", first(candidates))
    solve_candidates = BoxDMK._fortran_solve_library_candidates()
    @test !isempty(solve_candidates)
    @test occursin("libboxdmk_hot.so", first(solve_candidates))

    benchmark_driver = normpath(joinpath(@__DIR__, "..", "benchmark", "hybrid_parity.jl"))
    @test isfile(benchmark_driver)

    if isfile(benchmark_driver)
        include(benchmark_driver)
        @test isdefined(Main, :run_hybrid_parity_reference)
        report = run_hybrid_parity_reference(; execute = false)
        @test report.case.kernel == :laplace
        @test report.case.ndim == 3
        @test report.case.norder == 16
        @test report.case.eps == 1e-6
        @test report.library_path == BoxDMK._resolve_fortran_library_path()
        @test report.vendored_root == vendored_root
        @test report.execute == false
        @test length(report.step_names) == 9
    end
end

# The two testsets below use the Fortran tree builder, which calls the Julia
# source function from inside an OpenMP parallel region and can deadlock
# against Julia's GC on many-core machines. Opt in with
# BOXDMK_TEST_FORTRAN_TREE=1 (and OMP_NUM_THREADS=1). The debug snapshot
# additionally requires the debug variant of the vendored solve library.
if get(ENV, "BOXDMK_TEST_FORTRAN_TREE", "0") == "1"
@testset "Fortran Debug Snapshot Surface" begin
    @test isdefined(BoxDMK, :reset_fortran_debug!)
    @test isdefined(BoxDMK, :get_fortran_debug_snapshot)

    kernel = LaplaceKernel()
    basis = LegendreBasis()
    eps_val = 1e-2

    ftree = build_tree_fortran(
        _hybrid_debug_source,
        kernel,
        basis;
        ndim = 3,
        norder = 4,
        eps = eps_val,
        boxlen = 1.0,
        nd = 1,
        eta = 1.0,
    )

    BoxDMK.reset_fortran_debug!()
    result = bdmk_fortran(ftree, kernel; eps = eps_val)
    snapshot = BoxDMK.get_fortran_debug_snapshot()

    @test snapshot.nd == 1
    @test snapshot.npbox == 4^3
    @test snapshot.nboxes == BoxDMK.nboxes(ftree.tree)
    @test snapshot.step2_pot !== nothing
    @test snapshot.step3_proxycharge !== nothing
    @test snapshot.step6_proxypotential !== nothing
    @test snapshot.step7_pot !== nothing
    @test snapshot.step8_pot !== nothing
    @test snapshot.step9_pot !== nothing
    @test size(snapshot.step9_pot) == size(result.pot)
    @test snapshot.step9_pot ≈ result.pot atol = 1e-12 rtol = 1e-12
end

@testset "Fortran Tree Packing Surface" begin
    @test isdefined(BoxDMK, :pack_tree_fortran)

    kernel = LaplaceKernel()
    basis = LegendreBasis()
    eps_val = 1e-2

    original = build_tree_fortran(
        _hybrid_debug_source,
        kernel,
        basis;
        ndim = 3,
        norder = 4,
        eps = eps_val,
        boxlen = 1.0,
        nd = 1,
        eta = 1.0,
    )

    packed = BoxDMK.pack_tree_fortran(original.tree, original.fvals)
    direct = bdmk_fortran(original, kernel; eps = eps_val)
    roundtrip = bdmk_fortran(packed, kernel; eps = eps_val)

    @test packed.tree === original.tree
    @test size(packed.fvals) == size(original.fvals)
    @test packed.boxsize == original.boxsize
    @test roundtrip.pot ≈ direct.pot atol = 1e-12 rtol = 1e-12
end
end

@testset "Hybrid Direct Solve Surface" begin
    kernel = LaplaceKernel()
    basis = LegendreBasis()
    eps_tree = 1e-2
    eps_solve = 1e-2

    tree, fvals = build_tree(
        _hybrid_debug_source,
        kernel,
        basis;
        ndim = 3,
        norder = 4,
        eps = eps_tree,
        boxlen = 1.0,
        nd = 1,
        eta = 1.0,
    )

    packed = BoxDMK.pack_tree_fortran(tree, fvals)
    via_explicit_pack = bdmk_fortran(packed, kernel; eps = eps_solve)
    tree2, fvals2 = build_tree(
        _hybrid_debug_source,
        kernel,
        basis;
        ndim = 3,
        norder = 4,
        eps = eps_tree,
        boxlen = 1.0,
        nd = 1,
        eta = 1.0,
    )
    via_direct_hybrid = bdmk_fortran(tree2, fvals2, kernel; eps = eps_solve)

    @test via_direct_hybrid.pot ≈ via_explicit_pack.pot atol = 1e-12 rtol = 1e-12
end

@testset "Public Solver Hybrid Dispatch" begin
    tree, fvals, targets = _hybrid_reference_case_inputs()
    kernel = LaplaceKernel()

    packed_order = BoxDMK._fortran_level_order(tree)
    reference = bdmk_fortran(tree, fvals, kernel; eps = 1e-6, targets = targets)
    reference_pot = similar(reference.pot)
    reference_pot[:, :, packed_order] = reference.pot
    public = bdmk(tree, fvals, kernel; eps = 1e-6, targets = targets)

    @test public.pot ≈ reference_pot atol = 1e-11 rtol = 1e-11
    @test public.target_pot ≈ reference.target_pot atol = 1e-11 rtol = 1e-11
    @test public.grad === nothing
    @test public.hess === nothing
    @test public.target_grad === nothing
    @test public.target_hess === nothing
end

@testset "Public Solver Uses Hybrid Outside Reference Slice" begin
    kernel = LaplaceKernel()
    basis = LegendreBasis()
    eps_tree = 1e-2
    eps_solve = 1e-2
    targets = [
        0.20  0.35  0.65;
        0.25  0.55  0.70;
        0.30  0.60  0.80;
    ]

    tree, fvals = build_tree(
        _hybrid_debug_source,
        kernel,
        basis;
        ndim = 3,
        norder = 4,
        eps = eps_tree,
        boxlen = 1.0,
        nd = 1,
        eta = 1.0,
    )

    packed_order = BoxDMK._fortran_level_order(tree)
    reference = bdmk_fortran(tree, fvals, kernel; eps = eps_solve, grad = true, hess = true, targets = targets)
    public = bdmk(tree, fvals, kernel; eps = eps_solve, grad = true, hess = true, targets = targets)

    reorder3(a) = begin
        out = similar(a)
        out[:, :, packed_order] = a
        out
    end
    reorder4(a) = begin
        out = similar(a)
        out[:, :, :, packed_order] = a
        out
    end
    reorder5(a) = begin
        out = similar(a)
        out[:, :, :, :, packed_order] = a
        out
    end

    @test public.pot ≈ reorder3(reference.pot) atol = 1e-11 rtol = 1e-11
    @test public.grad ≈ reorder4(reference.grad) atol = 1e-11 rtol = 1e-11
    @test public.hess ≈ reorder4(reference.hess) atol = 1e-11 rtol = 1e-11
    @test public.target_pot ≈ reference.target_pot atol = 1e-11 rtol = 1e-11
    @test public.target_grad ≈ reference.target_grad atol = 1e-11 rtol = 1e-11
    @test public.target_hess ≈ reference.target_hess atol = 1e-11 rtol = 1e-11
end
