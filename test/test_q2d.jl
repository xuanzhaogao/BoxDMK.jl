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
        t = g.Ainv * [y[1], y[2]]
        (all(0 .<= t .< 1) && 0 <= y[3] <= zl[2] - zl[1]) || continue
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
        # oblique lattice, charged: still independent of ξ; far field -2πQ|z|/A
        A = [1.5 0.75; 0.0 1.5 * sqrt(3) / 2]
        gq = q2d_mixture(A, [(0.3, 0.5, -0.05), (1.4, 0.9, 0.1)], [800.0, 400], [1.0, 0.3]; neutral = false)
        @test q2d_potential(gq, p; xi = 2.5) ≈ q2d_potential(gq, p; xi = 5.0) rtol = 1e-13
        uf = -2pi / det(A) * sum(gq.c[j] * (pi / gq.alpha[j])^1.5 * (9.0 - gq.x[j][3]) for j in 1:2)
        @test q2d_potential(gq, [0.2, 0.1, 9.0]) ≈ uf rtol = 1e-10
    end

    @testset "$name, eps=1e-$nd" for (name, L, zl, xs, al) in (
            ("square cell", (1.0, 1.0), (0.0, 1.0),
             [(0.2, 0.3, 0.45), (0.7, 0.6, 0.55), (0.9, 0.2, 0.5), (0.4, 0.8, 0.6)], [2000.0, 300, 400, 500]),
            ("2 × 3 cell", (2.0, 3.0), (-0.5, 0.5),
             [(0.3, 0.5, -0.05), (1.4, 2.1, 0.1), (1.8, 0.4, 0.0), (0.7, 2.7, -0.1)], [800.0, 400, 600, 500]),
            ("thin slab 4 × 4 × 0.5", (4.0, 4.0), (0.0, 0.5),
             [(0.5, 0.7, 0.25), (2.9, 1.3, 0.22), (1.7, 3.4, 0.28), (3.6, 2.2, 0.25)], [1500.0, 1000, 1200, 2000]),
            ("hexagonal 2 × 1", [2.0 0.5; 0.0 sqrt(3) / 2], (0.0, 0.8),
             [(0.3, 0.2, 0.4), (1.6, 0.6, 0.35), (1.0, 0.4, 0.45), (2.1, 0.1, 0.4)], [900.0, 500, 700, 600]),
            ("rectangle 1 × √2", (1.0, sqrt(2.0)), (0.0, 0.7),
             [(0.2, 0.3, 0.35), (0.7, 1.1, 0.3), (0.5, 0.6, 0.4), (0.9, 0.2, 0.35)], [1000.0, 600, 800, 500])),
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
        # a charged slab is rejected unless `charged`; rc at most min(spacing, height)
        gq = neutral_q2d_mixture(L, [(0.3, 0.5, 0.0), (1.4, 2.1, 0.1)], [800.0, 400], [1.0, 0.7])
        @test_throws ArgumentError bdmk_q2d(x -> gq(x) + 0.3 * exp(-500 * sum(abs2, x .- [1.0, 1.0, 0.0])), L, zl; eps = 1e-3)
        @test_throws ArgumentError bdmk_q2d(g, L, zl; eps = 1e-3, rc = 1.5)
    end

    @testset "given values, period quadrature ($name)" for (name, L, zl) in (("2 × 3 cell", (2.0, 3.0), (-0.5, 0.5)),
                                                                          ("hexagonal 2 × 1", [2.0 0.5; 0.0 sqrt(3) / 2], (-0.4, 0.4)))
        g = q2d_mixture(L, [(0.3, 0.5, -0.05), (1.4, 0.6, 0.1), (1.0, 0.2, 0.0)], [800.0, 400, 600], [1.0, 0.7, -0.2];
                        neutral = false)
        plan = q2d_plan(g, L, zl; eps = 1e-8, charged = true)
        Q = period_quadrature(plan)
        @test sum(Q.w) ≈ abs(det(g.A)) * (zl[2] - zl[1]) rtol = 1e-12
        fq = [g(view(Q.x, :, j)) for j in axes(Q.x, 2)]
        @test sum(Q.w .* fq) ≈ total_charge(g) rtol = 1e-10
        r1 = periodic_solve!(plan, g)
        u1 = zeros(size(Q.x, 2))
        period_values!(u1, Q, r1)
        e1 = sum(Q.w .* fq .* u1)
        P = sample_points(plan)
        r2 = periodic_solve!(plan, [g(view(P, :, j)) for j in axes(P, 2)])
        u2 = zeros(size(Q.x, 2))
        period_values!(u2, Q, r2)
        @test maximum(abs, u2 .- u1) < 1e-13 * maximum(abs, u1)
        # interpolated potential at the nodes and the energy against the Ewald sum
        js = 1:max(1, size(Q.x, 2) ÷ 97):size(Q.x, 2)
        uref = [q2d_potential(g, view(Q.x, :, j)) for j in js]
        @test maximum(abs, u1[js] .- uref) < 1e-8 * maximum(abs, uref)
        fv = zeros(size(Q.x, 2))
        period_values!(fv, Q, r1; field = r1.fvals)
        @test maximum(abs, fv .- fq) < 1e-7 * maximum(abs, fq)         # forest: L2 tail test at eps
    end

    @testset "oblique NUFFT back-transform, eps=1e-$nd" for nd in (6, 9)
        eps = 10.0^-nd
        # identical modes: NUFFT at the cube nodes vs dense exact projection
        A = [8.0 4.0 0; 0 4sqrt(3) 0; 0 0 1.0]
        gr = BoxDMK.PeriodicGrid(A, 1.0, 1; slab = true)
        r0, ws, ts = BoxDMK.useries_sog(nd, 1.0)
        iw = [sqrt(t * log(1 / eps)) > 1.0 for t in ts]
        pf, pe = BoxDMK.wide_orders(eps, BoxDMK.select_porder(eps), 16, 1.0)
        Sd = BoxDMK.SlabWide(gr, eps, pf, pe, ts[iw], ws[iw]; nufft = false)
        Sn = BoxDMK.SlabWide(gr, eps, pf, pe, ts[iw], ws[iw]; nufft = true)
        M = [sin(0.37 * i + 0.11 * j * j - 0.7 * k) for i in 1:(pf * Sd.mt[1]), j in 1:(pf * Sd.mt[2]), k in 1:(pf * Sd.mt[3])]
        ar = BoxDMK.Arena()
        Out = copy(BoxDMK._slab_modes_z(Sd, M, ar))
        Cd = BoxDMK._slab_back_oblique(Sd, Out, ar); Cn = BoxDMK._slab_back_oblique(Sn, Out, ar)
        blk(C, i, j) = view(C, :, pe * j .+ (1:pe), pe * i .+ (1:pe))
        cmax = maximum(c -> maximum(abs, blk(Cd, c...)), Sn.nu.cubes)
        @test maximum(c -> maximum(abs, blk(Cn, c...) .- blk(Cd, c...)), Sn.nu.cubes) < 0.5eps * cmax
        BoxDMK.release!(ar)
        # potentials after the downward pass, and against the Ewald sum (thin
        # hexagonal slab with level-0 target leaves), neutral and charged
        L = [6.0 3.0; 0.0 3sqrt(3)]; zl = (0.0, 0.5)
        for charged in (false, true)
            g = q2d_mixture(L, [(0.5, 0.7, 0.25), (4.9, 1.3, 0.22), (3.7, 3.4, 0.28), (6.6, 2.2, 0.25), (2.0, 4.5, 0.3)],
                            [1500.0, 1000, 1200, 2000, 800], [1.0, 0.7, -0.5, 0.9, -0.3]; neutral = !charged)
            rd = bdmk_q2d(g, L, zl; eps, charged, wide_nufft = false)
            rn = bdmk_q2d(g, L, zl; eps, charged, wide_nufft = true)
            ft = rn.ft
            lv = [b for b in 1:ft.nboxes if ft.nchild[b] == 0 && rn.istarget[b]]
            @test any(b -> ft.level[b] == 0, lv)
            umax = maximum(b -> maximum(abs, view(rd.pot, :, :, :, b)), lv)
            @test maximum(b -> maximum(abs, view(rn.pot, :, :, :, b) .- view(rd.pot, :, :, :, b)), lv) < 0.05eps * umax
            l2, mx = _q2d_errors(rn, g, L, zl)
            @test l2 < 3eps
            @test mx < eps
        end
    end

    @testset "charged, $name, eps=1e-$nd" for (name, L, zl) in (("2 × 3 cell", (2.0, 3.0), (-0.5, 0.5)),
                                                               ("hexagonal 2 × 1", [2.0 0.5; 0.0 sqrt(3) / 2], (-0.4, 0.4))),
                                               nd in (3, 6)
        eps = 10.0^-nd
        g = q2d_mixture(L, [(0.3, 0.5, -0.05), (1.4, 0.6, 0.1), (1.0, 0.2, 0.0)], [800.0, 400, 600], [1.0, 0.7, -0.2];
                        neutral = false)
        res = bdmk_q2d(g, L, zl; eps = eps, charged = true)
        l2, mx = _q2d_errors(res, g, L, zl)
        @test l2 < 3eps
        @test mx < eps
    end
end
