# Quick native-only timing of one case: julia -t T benchmark/scaling_probe.jl norder eps grad hess
using BoxDMK, Printf
include(joinpath(@__DIR__, "native_vs_fortran.jl"))
function probe()
n = parse(Int, ARGS[1]); e = parse(Float64, ARGS[2]); g = ARGS[3] == "1"; h = ARGS[4] == "1"
tree, fvals = build_tree(SRC, LaplaceKernel(), LegendreBasis(); ndim = 3, norder = n, eps = e, boxlen = 1.18, nd = 1, eta = 0.0)
tm = zeros(10); best = Inf; bt = zeros(10)
for _ in 1:4
    t = @elapsed bdmk(tree, fvals, LaplaceKernel(); eps = e, grad = g, hess = h, timings = tm)
    t < best && (best = t; bt .= tm)
end
@printf("T=%3d n=%d eps=%.0e grad=%d hess=%d: %.3f s  (pre %.3f s2 %.3f s3 %.3f pw %.3f [form %.3f m2l %.3f eval %.3f down %.3f] s7 %.3f out %.3f)\n",
        Threads.nthreads(), n, e, g, h, best, bt[1], bt[2], bt[3], bt[4], bt[5], bt[6], bt[8], bt[9], bt[7], bt[10])
end
probe()
