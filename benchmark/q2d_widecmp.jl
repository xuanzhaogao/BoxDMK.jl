# Wide-part transforms alone (random level-0 moments -> coefficients) for the
# quasi-2D slab and the fully periodic solver on the same grids.
#   julia --project -t 48 benchmark/q2d_widecmp.jl <digits>
using BoxDMK, Printf, LinearAlgebra
function setup(A, rc, nd, slab)
    eps = 10.0^-nd
    g = BoxDMK.PeriodicGrid(A, rc, 1; slab)
    r0, ws, ts = BoxDMK.useries_sog(nd, rc)
    iw = [sqrt(t * log(1 / eps)) > rc for t in ts]
    pf, pe = BoxDMK.wide_orders(eps, BoxDMK.select_porder(eps), 16, rc)
    t = @elapsed P = (slab ? BoxDMK.SlabWide : BoxDMK.WidePlan)(g, eps, pf, pe, ts[iw], ws[iw])
    return g, P, t
end
function run(name, A, rc, nd, slab)
    g, P, tp = setup(A, rc, nd, slab)
    M = randn(P.pf * g.mgrid[1], P.pf * g.mgrid[2], P.pf * g.mgrid[3])
    best = Inf
    for r in 1:3
        ar = BoxDMK.Arena()
        t = @elapsed (slab ? BoxDMK.slab_coefficients(P, M, ar) : BoxDMK.wide_coefficients(P, BoxDMK.wide_modes(P, M, ar), ar))
        BoxDMK.release!(ar); best = min(best, t)
    end
    modes = slab ? (P.Kh[1] + 1) * (2P.Kh[2] + 1) : prod(size(P.W))
    @printf("%-34s eps=1e-%d T%-2d | level-0 cells %-14s modes %9d | wide transforms %8.1f ms  (plan %.2f s)\n",
            name, nd, Threads.nthreads(), string(g.mgrid), modes, 1e3best, tp)
end
nd = parse(Int, ARGS[1])
run("3D periodic cube 1x1x1, rc=0.25", Matrix(1.0I, 3, 3), 0.25, nd, false)
run("3D periodic 100x100x4, rc=1", Matrix(Diagonal([100.0, 100, 4])), 1.0, nd, false)
run("q2d 100x100x1, rc=1", Matrix(Diagonal([100.0, 100, 1])), 1.0, nd, true)
run("q2d 100x100x1, rc=0.5", Matrix(Diagonal([100.0, 100, 1])), 0.5, nd, true)
