using BoxDMK
using Test
using Random

include("periodic_oracle.jl")

const _PXS = [(0.20, 0.25, 0.40), (0.72, 0.65, 0.55), (0.95, 0.30, 0.80), (0.45, 0.85, 0.10)]

# Pseudo-random integers in 1:m, the same on every Julia version (the streams
# of `rand(rng, range)` are not).
mutable struct _LCG
    s::UInt64
end
_draw!(r::_LCG, m::Int) = (r.s = r.s * 0x5851f42d4c957f2d + 0x14057b7ef767814f; Int((r.s >> 33) % UInt64(m)) + 1)

# Errors of the solution at `nsamp` pseudo-random in-cell leaf nodes against
# Ewald: relative L2 over the samples, and max error relative to max|u| over
# all computed leaf nodes (printed with BOXDMK_TEST_VERBOSE=1).
function _periodic_errors(res, g; nsamp = parse(Int, get(ENV, "BOXDMK_TEST_NSAMP", "300")))
    ft, n, G = res.ft, res.norder, res.grid
    x = BoxDMK.ortho1d(LegendreBasis(), n).x
    leaves = [i for i in 1:ft.nboxes if ft.nchild[i] == 0 && res.istarget[i]]
    rmax = maximum(i -> maximum(abs, view(res.pot, :, :, :, i)), leaves)
    rng = _LCG(7)
    e2 = r2 = emax = 0.0
    k = 0
    while k < nsamp
        ib = leaves[_draw!(rng, length(leaves))]
        a, b, c = _draw!(rng, n), _draw!(rng, n), _draw!(rng, n)
        h = ft.boxsize[ft.level[ib] + 1] / 2
        y = [ft.centers[1, ib] + h * x[a], ft.centers[2, ib] + h * x[b], ft.centers[3, ib] + h * x[c]]
        all(0 .<= G.Uinv * y .< 1) || continue          # node in the cell
        k += 1
        uref = ewald_potential(g, G.Q * y)
        u = res.pot[a, b, c, ib]
        e2 += (u - uref)^2; r2 += uref^2
        emax = max(emax, abs(u - uref))
    end
    get(ENV, "BOXDMK_TEST_VERBOSE", "0") == "1" && println("    rel L2 ", sqrt(e2 / r2), "  max ", emax / rmax)
    return sqrt(e2 / r2), emax / rmax
end

@testset "Periodic" begin
    @testset "u-series SOG" begin
        for (nd, nterms) in ((3, 47), (6, 135), (9, 249))
            r0, ws, ts = BoxDMK.useries_sog(nd, 1.0)
            @test length(ts) == nterms
            @test r0 == 2.0^-16
            for r in (1e-3, 0.1, 1.0, 10.0)
                @test abs(1 - r * sum(ws .* exp.(-r^2 ./ ts))) < 10.0^-nd
            end
        end
    end

    @testset "spherical Bessel" begin
        j = zeros(40)
        BoxDMK.sphbesselj!(j, 39, 1e-3)                # small x: leading series terms
        @test j[2] ≈ 1e-3 / 3 - 1e-9 / 30 rtol = 1e-12
        @test j[3] ≈ 1e-6 / 15 - 1e-12 / 210 rtol = 1e-12
        for x in (1e-3, 0.5, 3.0, 20.0)
            BoxDMK.sphbesselj!(j, 39, x)
            @test j[1] ≈ sin(x) / x rtol = 1e-13
            if x >= 0.5
                @test j[2] ≈ sin(x) / x^2 - cos(x) / x rtol = 1e-10
                @test j[3] ≈ (3 / x^2 - 1) * sin(x) / x - 3cos(x) / x^2 rtol = 1e-8
            end
            for l in 2:38                      # three-term recurrence
                @test j[l + 1] + j[l - 1] ≈ (2l - 1) / x * j[l] rtol = 1e-10 atol = 1e-300
            end
        end
    end

    @testset "forest" begin
        g = neutral_mixture((1.0, 1.0, 1.0), _PXS, [20000.0, 200, 300, 400], [1.0, 0.7, -0.5, 0.9])
        grid = BoxDMK.PeriodicGrid((1.0, 1.0, 1.0), 0.25, 1)
        ft, fvals, ist = BoxDMK.build_periodic_forest(g, grid; norder = 16, eps = 1e-3)
        @test length(BoxDMK.levelboxes(ft, 0)) == 6^3
        @test count(ist[BoxDMK.levelboxes(ft, 0)]) == 4^3
        @test ft.nlevels >= 2
        # 2:1 balance: no leaf touches a box two levels finer
        for b in 1:ft.nboxes, c in 1:ft.nboxes
            (ft.nchild[b] == 0 && ft.level[c] >= ft.level[b] + 2) || continue
            d = 0.5 * (ft.boxsize[ft.level[b] + 1] + ft.boxsize[ft.level[c] + 1])
            @test any(k -> abs(ft.centers[k, b] - ft.centers[k, c]) > d * (1 + 1e-12), 1:3)
        end
        @test fvals[3, 4, 5, 1] ≈ g([ft.centers[1, 1] + 0.125 * BoxDMK.ortho1d(LegendreBasis(), 16).x[3],
                                     ft.centers[2, 1] + 0.125 * BoxDMK.ortho1d(LegendreBasis(), 16).x[4],
                                     ft.centers[3, 1] + 0.125 * BoxDMK.ortho1d(LegendreBasis(), 16).x[5]])
    end

    @testset "cube vs Ewald, $(dist), eps=1e-$nd" for (dist, alpha) in (("smooth", [60.0, 120, 180, 240]),
                                                                        ("multiscale", [20000.0, 200, 300, 400])),
                                                      nd in (3, 6)
        eps = 10.0^-nd
        g = neutral_mixture((1.0, 1.0, 1.0), _PXS, alpha, [1.0, 0.7, -0.5, 0.9])
        res = bdmk_periodic(g, 1.0; eps = eps)
        l2, mx = _periodic_errors(res, g)
        @test l2 < 5eps
        @test mx < 2eps
    end

    @testset "orthorhombic cell vs Ewald" begin
        L = (1.0, 1.5, 2.0)
        xs = [(0.20, 0.35, 0.80), (0.72, 0.95, 1.1), (0.95, 0.45, 1.6), (0.45, 1.3, 0.2)]
        g = neutral_mixture(L, xs, [2000.0, 200, 300, 400], [1.0, 0.7, -0.5, 0.9])
        res = bdmk_periodic(g, L; eps = 1e-6)
        @test res.grid.mgrid == (4, 6, 8)
        l2, mx = _periodic_errors(res, g)
        @test l2 < 5e-6
        @test mx < 2e-6
        # evaluation anywhere, periodically
        p = [0.3, 0.7, 1.9]
        u = evaluate_periodic(res, p)
        @test evaluate_periodic(res, p .+ [1.0, -1.5, 4.0]) ≈ u rtol = 1e-12
        @test u ≈ ewald_potential(g, p) rtol = 1e-5 atol = 1e-6 * maximum(abs, res.pot)
    end

    @testset "$name cell, $(neutral ? "neutral" : "charged") $dist, eps=1e-$nd" for (name, A, dist, neutral, nd) in (
            ("sheared", [1.0 0.35 0.15; 0 1 -0.2; 0 0 1], "multiscale", false, 6),
            ("sheared", [1.0 0.35 0.15; 0 1 -0.2; 0 0 1], "smooth", true, 3),
            ("hexagonal", [1.0 0.5 0; 0 0.8660254037844386 0; 0 0 1.2], "smooth", false, 6),
            ("1:√2:√3", [1.0 0 0; 0 sqrt(2) 0; 0 0 sqrt(3)], "multiscale", true, 6),
            ("cubic", [1.0 0 0; 0 1 0; 0 0 1], "multiscale", false, 6))
        alpha = dist == "smooth" ? [60.0, 120, 180, 240] : [20000.0, 200, 300, 400]
        xs = [Tuple(A * collect(t)) for t in _PXS]
        c = [1.0, 0.7, -0.5, 0.9]
        g = neutral ? neutral_mixture(A, xs, alpha, c) : GaussMixture(A, xs, alpha, c)
        res = bdmk_periodic(g, A; eps = 10.0^-nd)
        @test res.grid.tiled == (name == "cubic")
        l2, mx = _periodic_errors(res, g)
        @test l2 < 5 * 10.0^-nd
        @test mx < 2 * 10.0^-nd
        p = [0.3, 0.7, 0.9]
        @test evaluate_periodic(res, p .+ A * [1.0, -2.0, 1.0]) ≈ evaluate_periodic(res, p) rtol = 1e-12
    end

    # A spike far narrower than the coarse cells (the C++ known failure): the
    # fractional source tree must follow the main forest's refinement.
    @testset "lone spike, sheared cell, eps=1e-9" begin
        A = [1.0 0.35 0.15; 0 1 -0.2; 0 0 1]
        tf = [(0.52, 0.48, 0.51), _PXS[1], _PXS[2], _PXS[3]]
        g = neutral_mixture(A, [Tuple(A * collect(t)) for t in tf], [5e5, 200, 300, 400], [1.0, 0.7, -0.5, 0.9])
        res = bdmk_periodic(g, A; eps = 1e-9)
        l2, mx = _periodic_errors(res, g; nsamp = 200)
        @test mx < 2e-9
    end

    @testset "plan reuse and targets" begin
        A = [1.0 0.35 0.15; 0 1 -0.2; 0 0 1]
        xs = [Tuple(A * collect(t)) for t in _PXS]
        g = neutral_mixture(A, xs, [2000.0, 200, 300, 400], [1.0, 0.7, -0.5, 0.9])
        r1 = bdmk_periodic(g, A; eps = 1e-6, near32 = false)     # exactly linear in f
        p1 = copy(r1.pot)
        plan = periodic_plan(g, A; eps = 1e-6, near32 = false)
        @test periodic_solve!(plan, g).pot == p1                     # fresh samples
        p2 = copy(periodic_solve!(plan, g).pot)                       # resampled
        m = [i for i in 1:plan.ft.nboxes if plan.ft.nchild[i] == 0 && plan.istarget[i]]
        um = maximum(abs, p1[:, :, :, m])
        @test maximum(abs, p2[:, :, :, m] .- p1[:, :, :, m]) < 1e-8 * um
        p3 = periodic_solve!(plan, x -> -3 * g(x)).pot
        @test maximum(abs, p3[:, :, :, m] .+ 3 .* p1[:, :, :, m]) < 1e-8 * um
        T = A * rand(MersenneTwister(5), 3, 20) .+ A * [0.0, 2, -1]
        rt = bdmk_periodic(g, A; eps = 1e-6, targets = T, near32 = false)
        @test count(rt.istarget) < count(r1.istarget)
        @test rt.target_pot ≈ [evaluate_periodic(r1, T[:, j]) for j in 1:20] rtol = 1e-12
        @test_throws ArgumentError evaluate_periodic(rt, A * [0.5, 0.01, 0.99] .+ 1e-3 .* (1:3))
    end

    @testset "sample once" begin
        A = [1.0 0.35 0.15; 0 1 -0.2; 0 0 1]
        xs = [Tuple(A * collect(t)) for t in _PXS]
        g = GaussMixture(A, xs, [2000.0, 200, 300, 400], [1.0, 0.7, -0.5, 0.9])
        ncall = Threads.Atomic{Int}(0)
        gc(x) = (Threads.atomic_add!(ncall, 1); g(x))
        r0 = bdmk_periodic(gc, A; eps = 1e-6, sample_once = false)
        n0 = ncall[]; ncall[] = 0
        plan = periodic_plan(gc, A; eps = 1e-6)
        r1 = periodic_solve!(plan, gc)
        @test ncall[] < n0 / 2                      # f only in the adaptive refinement
        lv = [i for i in 1:r1.ft.nboxes if r1.ft.nchild[i] == 0 && r1.istarget[i]]
        u = maximum(abs, r0.pot)
        p1 = copy(r1.pot)
        @test r1.ft.nboxes == r0.ft.nboxes
        @test maximum(i -> maximum(abs, view(p1, :, :, :, i) .- view(r0.pot, :, :, :, i)), lv) < 1e-6 * u
        r2 = periodic_solve!(plan, gc)               # resampled, then derived as when built
        @test maximum(i -> maximum(abs, view(p1, :, :, :, i) .- view(r2.pot, :, :, :, i)), lv) < 1e-13 * u
    end

    @testset "batch density" begin
        g = neutral_mixture((1.0, 1.0, 1.0), _PXS, [60.0, 120, 180, 240], [1.0, 0.7, -0.5, 0.9])
        gb!(v, pts) = (for i in axes(pts, 2); v[1, i] = g(view(pts, :, i)); end)
        r1 = bdmk_periodic(g, 1.0; eps = 1e-3)
        r2 = bdmk_periodic(nothing, 1.0; eps = 1e-3, f_batch = gb!)
        @test r1.pot == r2.pot
    end

    @testset "argument checks" begin
        @test_throws ArgumentError bdmk_periodic(x -> 0.0, [1.0 2; 2 4; 0 0][:, [1, 2, 2]]; eps = 1e-3)   # degenerate
        @test_throws ArgumentError bdmk_periodic(x -> 0.0, 1.0; eps = 1e-3, nhalo = 0)
    end
end
