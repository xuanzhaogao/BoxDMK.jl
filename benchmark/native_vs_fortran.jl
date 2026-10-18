# Pure-Julia solver vs the Fortran reference on the same trees.
#
#   julia --project -t <T> benchmark/native_vs_fortran.jl [out.md]
#
# Fortran runs with OMP_NUM_THREADS from the environment (build the reference
# with `julia --project deps/build_fortran_ref.jl`). The density is
# f = -Δu/(4π) for a sum of Gaussians u, so the exact Laplace potential is u.

using BoxDMK, LinearAlgebra, Printf

struct Manufactured
    centers::Vector{NTuple{3,Float64}}
    sigmas::Vector{Float64}
    amps::Vector{Float64}
    shift::Float64          # domain is [0, L]^3, Gaussians placed around L/2
end

function exact_u(m::Manufactured, x)
    u = 0.0
    for (c, s, a) in zip(m.centers, m.sigmas, m.amps)
        r2 = (x[1] - m.shift - c[1])^2 + (x[2] - m.shift - c[2])^2 + (x[3] - m.shift - c[3])^2
        u += a * exp(-r2 / s^2)
    end
    return u
end

function (m::Manufactured)(x)
    f = 0.0
    for (c, s, a) in zip(m.centers, m.sigmas, m.amps)
        r2 = (x[1] - m.shift - c[1])^2 + (x[2] - m.shift - c[2])^2 + (x[3] - m.shift - c[3])^2
        f -= a * (4r2 / s^4 - 6 / s^2) * exp(-r2 / s^2) / (4pi)
    end
    return f
end

function grid_points(tree, ibox)
    nodes, _ = BoxDMK.nodes_and_weights(tree.basis, tree.norder)
    h = tree.boxsize[tree.level[ibox] + 1] / 2
    c = tree.centers[:, ibox]
    return [(c[1] + h * nodes[i], c[2] + h * nodes[j], c[3] + h * nodes[k])
            for i in eachindex(nodes), j in eachindex(nodes), k in eachindex(nodes)]
end

function exact_error(tree, pot, m)
    num = 0.0; den = 0.0
    for ibox in 1:size(pot, 3)
        all(iszero, @view tree.children[:, ibox]) || continue
        for (j, x) in enumerate(vec(grid_points(tree, ibox)))
            u = exact_u(m, x)
            num += (pot[1, j, ibox] - u)^2
            den += u^2
        end
    end
    return sqrt(num / den)
end

const SRC = Manufactured([(0.1, 0.02, 0.04), (0.03, -0.1, 0.05), (-0.12, 0.08, -0.06)],
                         [0.02, 0.01, 0.03], [1.0, -0.5, 0.7], 0.59)

function run_case(io; norder, eps, grad = false, hess = false, nrep = 3, fortran = true)
    kernel = LaplaceKernel()
    tree, fvals = build_tree(SRC, kernel, LegendreBasis(); ndim = 3, norder = norder, eps = eps,
                             boxlen = 1.18, nd = 1, eta = 0.0)
    tf = Inf; rf = nothing
    if fortran
        for _ in 1:nrep
            t = @elapsed rf = bdmk_fortran(tree, fvals, kernel; eps = eps, grad = grad, hess = hess)
            tf = min(tf, t)
        end
    end
    tn = Inf; rn = nothing; tm = zeros(10); best = zeros(10)
    for _ in 1:nrep
        t = @elapsed rn = bdmk(tree, fvals, kernel; eps = eps, grad = grad, hess = hess, timings = tm)
        t < tn && (tn = t; best .= tm)
    end
    parity = fortran ? norm(rn.pot .- rf.pot) / norm(rf.pot) : NaN
    err = exact_error(tree, rn.pot, SRC)
    errf = fortran ? exact_error(tree, rf.pot, SRC) : NaN
    what = hess ? "pot+grad+hess" : (grad ? "pot+grad" : "pot")
    @printf(io, "| %2d | %.0e | %-13s | %5d | %2d | %8.3f | %8.3f | %5.2fx | %.1e | %.1e | %.1e |\n",
            norder, eps, what, size(fvals, 3), tree.nlevels, tf, tn, tf / tn, parity, err, errf)
    flush(io)
    @printf("   native steps: pre %.3f  s2 %.3f  s3 %.3f  pw %.3f  s7 %.3f  s8 %.3f  s9 %.3f  out %.3f\n",
            best[1], best[2], best[3], best[4], best[7], best[8], best[9], best[10])
    flush(stdout)
end

function main(out = stdout)
    @printf(out, "\nJulia threads: %d, OMP_NUM_THREADS: %s, host: %s\n\n", Threads.nthreads(),
            get(ENV, "OMP_NUM_THREADS", "unset"), gethostname())
    println(out, "| n | eps | outputs | boxes | lev | Fortran (s) | Julia (s) | speedup | parity | err Julia | err Fortran |")
    println(out, "|---|-----|---------|-------|-----|-------------|-----------|---------|--------|-----------|-------------|")
    for (n, e) in ((8, 1e-3), (8, 1e-6), (16, 1e-3), (16, 1e-6), (16, 1e-9), (12, 1e-9))
        run_case(out; norder = n, eps = e)
    end
    run_case(out; norder = 16, eps = 1e-6, grad = true)
    run_case(out; norder = 16, eps = 1e-6, grad = true, hess = true, nrep = 2)
end

if abspath(PROGRAM_FILE) == @__FILE__
    if isempty(ARGS)
        main()
    else
        open(io -> main(io), ARGS[1], "a")
    end
end
