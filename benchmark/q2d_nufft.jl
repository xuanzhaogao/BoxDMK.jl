# Oblique slab cells: dense vs NUFFT back-transform of the wide part (modes ->
# Legendre coefficients of the level-0 cubes), on identical modes `Out`.
# Hexagonal cells L × L (60°), slab height H, level-0 side rc.
#
#   julia --project -t 48 benchmark/q2d_nufft.jl [digits] [H] [rc] [L list]
#
# Reports per size: level-0 grid, in-plane modes K, needed cubes C, NUFFT
# targets C q², fine grid, NUFFT plan time, best-of-3 times of the dense and
# NUFFT back-transforms and of the mode/z step that precedes both, their
# arena memory (MB) and the max coefficient difference relative to max|C|.

using BoxDMK, LinearAlgebra, Printf, Random
const BD = BoxDMK

nd = parse(Int, get(ARGS, 1, "6"))
H = parse(Float64, get(ARGS, 2, "1.0"))
rc = parse(Float64, get(ARGS, 3, "1.0"))
Ls = length(ARGS) >= 4 ? parse.(Float64, split(ARGS[4], ",")) : [8.0, 16, 32, 64, 128]
eps = 10.0^-nd

function besttime(f, reps = 3)
    t = Inf; mem = 0
    for _ in 1:reps
        ar = BD.Arena()
        t = min(t, @elapsed f(ar))
        mem = sum(last, ar.ptrs; init = 0) * 8
        BD.release!(ar)
    end
    return t, mem
end

@printf("# eps=1e-%d H=%.2f rc=%.2f threads=%d\n", nd, H, rc, Threads.nthreads())
@printf("#  L   mgrid          K(modes)  cubes   targets    fine grid    w  q | plan(s) | modes+z(ms) | dense(ms) MB | nufft(ms) MB | rel diff\n")
for L in Ls
    A = [L L / 2 0; 0 L * sqrt(3) / 2 0; 0 0 H]
    g = BD.PeriodicGrid(A, rc, 1; slab = true)
    r0, ws, ts = BD.useries_sog(nd, rc)
    iw = [sqrt(t * log(1 / eps)) > rc for t in ts]
    pf, pe = BD.wide_orders(eps, BD.select_porder(eps), 16, rc)
    Sd = BD.SlabWide(g, eps, pf, pe, ts[iw], ws[iw]; nufft = false)
    tp = @elapsed Sn = BD.SlabWide(g, eps, pf, pe, ts[iw], ws[iw]; nufft = true)
    tp -= @elapsed BD.SlabWide(g, eps, pf, pe, ts[iw], ws[iw]; nufft = false)   # NUFFT part of the setup
    Random.seed!(1)
    M = randn(pf * Sd.mt[1], pf * Sd.mt[2], pf * Sd.mt[3])
    ar0 = BD.Arena()
    Out = copy(BD._slab_modes_z(Sd, M, ar0))
    BD.release!(ar0)
    tz, _ = besttime(ar -> BD._slab_modes_z(Sd, M, ar))
    td, md = besttime(ar -> BD._slab_back_oblique(Sd, Out, ar))
    tn, mn = besttime(ar -> BD._slab_back_oblique(Sn, Out, ar))
    # difference on the needed cubes (the NUFFT writes only those)
    P = Sn.nu
    ar1 = BD.Arena(); Cd = BD._slab_back_oblique(Sd, Out, ar1); Cn = BD._slab_back_oblique(Sn, Out, ar1)
    pe = Sd.pe
    blk(C, i, j) = view(C, :, pe * j .+ (1:pe), pe * i .+ (1:pe))
    dmax = maximum(c -> maximum(abs, blk(Cn, c...) .- blk(Cd, c...)), P.cubes)
    cmax = maximum(c -> maximum(abs, blk(Cd, c...)), P.cubes)
    BD.release!(ar1)
    @printf("%5.0f %-14s %9d %6d %10d  %-12s %2d %2d | %7.3f | %10.1f | %8.1f %5.0f | %8.1f %5.0f | %.2e\n",
            L, string(g.mgrid), (Sd.Kh[1] + 1) * (2Sd.Kh[2] + 1), length(P.cubes), length(P.cubes) * P.q^2,
            string(P.n), P.w, P.q, max(tp, 0.0), 1e3tz, 1e3td, md / 2^20, 1e3tn, mn / 2^20, dmax / cmax)
    flush(stdout)
end
