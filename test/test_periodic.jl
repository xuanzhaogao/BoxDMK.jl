using BoxDMK
using Test
using Random

include("periodic_oracle.jl")

const _PXS = [(0.20, 0.25, 0.40), (0.72, 0.65, 0.55), (0.95, 0.30, 0.80), (0.45, 0.85, 0.10)]

# Errors of the solution at `nsamp` random in-cell leaf nodes against Ewald.
function _periodic_errors(res, g; nsamp = 300)
    ft, n = res.ft, res.norder
    x = BoxDMK.ortho1d(LegendreBasis(), n).x
    leaves = [i for i in 1:ft.nboxes if ft.nchild[i] == 0 && res.istarget[i]]
    rng = MersenneTwister(7)
    e2 = r2 = emax = rmax = 0.0
    for _ in 1:nsamp
        ib = rand(rng, leaves)
        a, b, c = rand(rng, 1:n, 3)
        h = ft.boxsize[ft.level[ib] + 1] / 2
        p = [ft.centers[1, ib] + h * x[a], ft.centers[2, ib] + h * x[b], ft.centers[3, ib] + h * x[c]]
        uref = ewald_potential(g, p; mmax = 16)
        u = res.pot[a, b, c, ib]
        e2 += (u - uref)^2; r2 += uref^2
        emax = max(emax, abs(u - uref)); rmax = max(rmax, abs(uref))
    end
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

    @testset "batch density" begin
        g = neutral_mixture((1.0, 1.0, 1.0), _PXS, [60.0, 120, 180, 240], [1.0, 0.7, -0.5, 0.9])
        gb!(v, pts) = (for i in axes(pts, 2); v[1, i] = g(view(pts, :, i)); end)
        r1 = bdmk_periodic(g, 1.0; eps = 1e-3)
        r2 = bdmk_periodic(nothing, 1.0; eps = 1e-3, f_batch = gb!)
        @test r1.pot == r2.pot
    end

    @testset "argument checks" begin
        @test_throws ArgumentError bdmk_periodic(x -> 0.0, (1.0, 1.3, 1.0); eps = 1e-3)  # 1.3 not a multiple of rc
        @test_throws ArgumentError bdmk_periodic(x -> 0.0, 1.0; eps = 1e-3, nhalo = 0)
    end
end
