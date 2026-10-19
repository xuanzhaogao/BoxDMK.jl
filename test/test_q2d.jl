using BoxDMK
using Test

include("q2d_oracle.jl")

# Errors at `nsamp` pseudo-random in-slab leaf nodes against the 2-periodic
# Ewald sum: relative L2 over the samples, max error relative to max|u|.
function _q2d_errors(res, g, L, zl; nsamp = 200)
    ft, n = res.ft, res.norder
    x = BoxDMK.ortho1d(LegendreBasis(), n).x
    leaves = [i for i in 1:ft.nboxes if ft.nchild[i] == 0 && res.istarget[i]]
    umax = maximum(i -> maximum(abs, view(res.pot, :, :, :, i)), leaves)
    draw(s, m) = (s = s * 0x5851f42d4c957f2d + 0x14057b7ef767814f; (s, Int((s >> 33) % UInt64(m)) + 1))
    s = UInt64(7)
    emax = e2 = r2 = 0.0
    k = 0
    while k < nsamp
        s, ib = draw(s, length(leaves)); s, a = draw(s, n); s, b = draw(s, n); s, c = draw(s, n)
        ib = leaves[ib]
        h = ft.boxsize[ft.level[ib] + 1] / 2
        y = (ft.centers[1, ib] + h * x[a], ft.centers[2, ib] + h * x[b], ft.centers[3, ib] + h * x[c])
        (0 <= y[1] < L[1] && 0 <= y[2] < L[2] && 0 <= y[3] <= zl[2] - zl[1]) || continue
        k += 1
        uref = q2d_potential(g, [y[1], y[2], y[3] + zl[1]])
        u = res.pot[a, b, c, ib]
        emax = max(emax, abs(u - uref)); e2 += (u - uref)^2; r2 += uref^2
    end
    return sqrt(e2 / r2), emax / umax
end

@testset "Quasi-2D" begin
    @testset "Ewald 2P reference" begin
        g = neutral_q2d_mixture((2.0, 3.0), [(0.3, 0.5, -0.05), (1.4, 2.1, 0.1)], [800.0, 400], [1.0, 1.0])
        p = [0.6, 2.3, 0.07]
        @test q2d_potential(g, p; xi = 2.0) ≈ q2d_potential(g, p; xi = 4.5) rtol = 1e-13
    end

    @testset "$name, eps=1e-$nd" for (name, L, zl, xs, al) in (
            ("square cell", (1.0, 1.0), (0.0, 1.0),
             [(0.2, 0.3, 0.45), (0.7, 0.6, 0.55), (0.9, 0.2, 0.5), (0.4, 0.8, 0.6)], [2000.0, 300, 400, 500]),
            ("2 × 3 cell", (2.0, 3.0), (-0.5, 0.5),
             [(0.3, 0.5, -0.05), (1.4, 2.1, 0.1), (1.8, 0.4, 0.0), (0.7, 2.7, -0.1)], [800.0, 400, 600, 500]),
            ("thin slab 4 × 4 × 0.5", (4.0, 4.0), (0.0, 0.5),
             [(0.5, 0.7, 0.25), (2.9, 1.3, 0.22), (1.7, 3.4, 0.28), (3.6, 2.2, 0.25)], [1500.0, 1000, 1200, 2000])),
                                                                    nd in (3, 6)
        eps = 10.0^-nd
        g = neutral_q2d_mixture(L, xs, al, [1.0, 0.7, -0.5, 0.9])
        res = bdmk_q2d(g, L, zl; eps = eps)
        l2, mx = _q2d_errors(res, g, L, zl)
        @test l2 < 3eps
        @test mx < eps
    end

    @testset "evaluation, plan, checks" begin
        L = (2.0, 3.0); zl = (-0.5, 0.5)
        g = neutral_q2d_mixture(L, [(0.3, 0.5, -0.05), (1.4, 2.1, 0.1), (1.8, 0.4, 0.0)], [800.0, 400, 600], [1.0, 0.7, -0.5])
        plan = q2d_plan(g, L, zl; eps = 1e-6, near32 = false)
        r1 = periodic_solve!(plan, g)
        p = [0.7, 1.9, 0.12]
        u = evaluate_periodic(r1, p)
        @test u ≈ q2d_potential(g, p) rtol = 1e-5
        @test evaluate_periodic(r1, p .+ [2.0, -6.0, 0.0]) ≈ u rtol = 1e-12        # periodic in x, y
        @test_throws ArgumentError evaluate_periodic(r1, [0.7, 1.9, 0.6])         # outside the slab
        u1 = copy(r1.pot)
        r2 = periodic_solve!(plan, x -> 2 * g(x))                                 # resampled, linear in f
        lv = [i for i in 1:r2.ft.nboxes if r2.ft.nchild[i] == 0 && r2.istarget[i]]
        @test maximum(i -> maximum(abs, view(r2.pot, :, :, :, i) .- 2 .* view(u1, :, :, :, i)), lv) < 1e-12 * maximum(abs, u1)
        gb!(v, P) = (for j in axes(P, 2); v[1, j] = g(view(P, :, j)); end; v)
        r3 = bdmk_q2d(nothing, L, zl; eps = 1e-6, near32 = false, f_batch = gb!)
        @test maximum(i -> maximum(abs, view(r3.pot, :, :, :, i) .- view(u1, :, :, :, i)), lv) < 1e-12 * maximum(abs, u1)
        # a charged slab is rejected; sides must be multiples of rc
        gq = neutral_q2d_mixture(L, [(0.3, 0.5, 0.0), (1.4, 2.1, 0.1)], [800.0, 400], [1.0, 0.7])
        @test_throws ArgumentError bdmk_q2d(x -> gq(x) + 0.3 * exp(-500 * sum(abs2, x .- [1.0, 1.0, 0.0])), L, zl; eps = 1e-3)
        @test_throws ArgumentError bdmk_q2d(g, (1.0, sqrt(2.0)), zl; eps = 1e-3)
    end
end
