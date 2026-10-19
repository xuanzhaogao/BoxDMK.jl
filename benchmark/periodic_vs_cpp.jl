# Periodic solver timing on the C++ `bench_steps cubic <dist> <n>` problem
# (PeriodicDMK_private): unit cube, the same Gaussian-mixture density with the
# same image loop. Optionally compares the potential box by box with a C++ run
# dumped by `BDMK_DUMP_POT=<pot> BDMK_DUMP_TREE=<tree> bench_steps ...`.
#
#   julia --project -t 48 benchmark/periodic_vs_cpp.jl <n> [multi|smooth] [reps] [cpp_pot cpp_tree]

using BoxDMK, Printf

const XS = [(0.20, 0.25, 0.40), (0.72, 0.65, 0.55), (0.95, 0.30, 0.80), (0.45, 0.85, 0.10)]
const CS = [1.0, 0.7, -0.5, 0.9]

# C++ `lattice_density` on the unit cube: per Gaussian, the images around the
# nearest one within sqrt(46/α).
struct CppDensity
    alpha::Vector{Float64}
    mrange::Vector{Int}
end
CppDensity(alpha) = CppDensity(alpha, [max(1, ceil(Int, sqrt(46 / a) - 0.5)) for a in alpha])

function (g::CppDensity)(p::AbstractVector)
    s = 0.0
    for j in eachindex(g.alpha)
        a = g.alpha[j]; m = g.mrange[j]
        d1 = p[1] - XS[j][1]; d2 = p[2] - XS[j][2]; d3 = p[3] - XS[j][3]
        c1 = round(d1); c2 = round(d2); c3 = round(d3)
        for i3 in -m:m, i2 in -m:m, i1 in -m:m
            e1 = d1 - (c1 + i1); e2 = d2 - (c2 + i2); e3 = d3 - (c3 + i3)
            s += CS[j] * exp(-a * (e1 * e1 + e2 * e2 + e3 * e3))
        end
    end
    return s
end

ndig = parse(Int, get(ARGS, 1, "6"))
dist = get(ARGS, 2, "multi")
reps = parse(Int, get(ARGS, 3, "3"))
f = CppDensity(dist == "multi" ? [20000.0, 200, 300, 400] : [60.0, 120, 180, 240])
eps = 10.0^-ndig

tm = zeros(6)
res = bdmk_periodic(f, 1.0; eps = eps, timings = tm)        # compile + warm up
best = Inf; tbest = zeros(6)
for r in 1:reps
    t = @elapsed global res = bdmk_periodic(f, 1.0; eps = eps, timings = tm)
    t < best && (global best = t; tbest .= tm)
    BoxDMK.release_memory!()
end
ft = res.ft
nleaf = count(i -> ft.nchild[i] == 0 && res.istarget[i], 1:ft.nboxes)
@printf("[JULIA-PERIODIC] threads=%d cell=cubic dist=%s n=%d | nboxes=%d depth=%d in-cell leaves=%d | total=%.0f ms : forest=%.0f setup=%.0f local+up=%.0f wide=%.0f pw=%.0f near=%.0f | %.2e pts/s\n",
        Threads.nthreads(), dist, ndig, ft.nboxes, ft.nlevels, nleaf, 1e3best, (1e3 .* tbest)...,
        nleaf * res.norder^3 / best)

function parity(res, potfile, treefile)
    ft = res.ft
    nleaf = count(i -> ft.nchild[i] == 0 && res.istarget[i], 1:ft.nboxes)
    n = res.norder; np = n^3
    cpot = reinterpret(Float64, read(potfile))
    ctree = reshape(reinterpret(Float64, read(treefile)), 5, :)
    key(c, h) = (round(Int, c[1] / h * 2), round(Int, c[2] / h * 2), round(Int, c[3] / h * 2), round(Int, 1 / h))
    cidx = Dict(key(ctree[1:3, b], ctree[4, b]) => b for b in axes(ctree, 2) if ctree[5, b] == 1)
    err2 = 0.0; ref2 = 0.0; errmax = 0.0; refmax = 0.0; nmatch = 0
    for i in 1:ft.nboxes
        (ft.nchild[i] == 0 && res.istarget[i]) || continue
        b = get(cidx, key(ft.centers[:, i], ft.boxsize[ft.level[i] + 1]), 0)
        b == 0 && continue
        nmatch += 1
        for q in 1:np
            u = res.pot[(i - 1) * np + q]; v = cpot[(b - 1) * np + q]
            err2 += (u - v)^2; ref2 += v^2
            errmax = max(errmax, abs(u - v)); refmax = max(refmax, abs(v))
        end
    end
    @printf("[PARITY] matched %d / %d in-cell leaves | rel L2 diff %.3e  max diff / max|u| %.3e\n",
            nmatch, nleaf, sqrt(err2 / ref2), errmax / refmax)
end
length(ARGS) >= 5 && parity(res, ARGS[4], ARGS[5])
