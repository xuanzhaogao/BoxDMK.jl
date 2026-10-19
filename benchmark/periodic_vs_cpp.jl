# Periodic solver timing on the C++ `bench_steps <cell> <dist> <n>` problem
# (PeriodicDMK_private): the same cell, the same Gaussian-mixture density with
# the same image loop. Optionally compares the potential box by box with a C++
# run dumped by `BDMK_DUMP_POT=<pot> BDMK_DUMP_TREE=<tree> bench_steps ...`.
#
#   julia --project -t 48 benchmark/periodic_vs_cpp.jl <cell> <n> [multi|smooth] [reps] [cpp_pot cpp_tree]
#
# DENSITY=simd evaluates the density with `density_batch!` (SIMD.jl, through
# `f_batch`) instead of point by point; the C++ code evaluates it point by point.
# "sampling" is the adaptive refinement that evaluates f, "rest" everything
# else. SAMPLE_ONCE=0 samples f everywhere (the C++ rules: box-by-box parity to
# rounding); by default the potentials agree with C++ to about eps.

using BoxDMK, Printf, LinearAlgebra

const CELLS = Dict(
    "cubic" => [1.0 0 0; 0 1 0; 0 0 1],
    "sheared" => [1.0 0.35 0.15; 0 1 -0.2; 0 0 1],
    "hex" => [1.0 0.5 0; 0 0.8660254037844386 0; 0 0 1.2],
    "tri-aniso4" => [1.0 0.5 0.4; 0 1 -0.6; 0 0 4],
    "rect-sqrt" => [1.0 0 0; 0 1.4142135623730951 0; 0 0 1.7320508075688772],
)
const TFRAC = [(0.20, 0.25, 0.40), (0.72, 0.65, 0.55), (0.95, 0.30, 0.80), (0.45, 0.85, 0.10)]
const CS = [1.0, 0.7, -0.5, 0.9]

# C++ `BdmkLattice` orders the lattice vectors by decreasing norm (stable);
# with the columns in that order both codes use the same rotated frame.
function cpp_sorted(A)
    ord = sortperm([norm(A[:, i]) for i in 1:3]; rev = true, alg = MergeSort)
    return A[:, ord]
end

# C++ `lattice_density`, in the rotated frame y = Qᵀ x of the sorted cell: per
# Gaussian (center U t_j), the images around the nearest one within
# sqrt(46/α), skipping exp below 1e-20.
struct CppDensity
    Q::Matrix{Float64}
    U::Matrix{Float64}
    Uinv::Matrix{Float64}
    xc::Vector{NTuple{3,Float64}}
    alpha::Vector{Float64}
    mrange::Vector{Int}
end

function CppDensity(A, alpha)
    g = BoxDMK.PeriodicGrid(A, 0.1, 1)
    xc = [Tuple(g.U * collect(t)) for t in TFRAC]
    CppDensity(g.Q, g.U, g.Uinv, xc, alpha, [max(1, ceil(Int, sqrt(46 / a) / g.dmin - 0.5)) for a in alpha])
end

function (g::CppDensity)(p::AbstractVector)
    Q, U, Ui = g.Q, g.U, g.Uinv
    y1 = Q[1, 1] * p[1] + Q[2, 1] * p[2] + Q[3, 1] * p[3]
    y2 = Q[1, 2] * p[1] + Q[2, 2] * p[2] + Q[3, 2] * p[3]
    y3 = Q[1, 3] * p[1] + Q[2, 3] * p[2] + Q[3, 3] * p[3]
    s = 0.0
    for j in eachindex(g.alpha)
        a = g.alpha[j]; m = g.mrange[j]
        d1 = y1 - g.xc[j][1]; d2 = y2 - g.xc[j][2]; d3 = y3 - g.xc[j][3]
        c1 = round(Int, Ui[1, 1] * d1 + Ui[1, 2] * d2 + Ui[1, 3] * d3)
        c2 = round(Int, Ui[2, 2] * d2 + Ui[2, 3] * d3)
        c3 = round(Int, Ui[3, 3] * d3)
        for n3 in (c3 - m):(c3 + m), n2 in (c2 - m):(c2 + m), n1 in (c1 - m):(c1 + m)
            e1 = d1 - (U[1, 1] * n1 + U[1, 2] * n2 + U[1, 3] * n3)
            e2 = d2 - (U[2, 2] * n2 + U[2, 3] * n3)
            e3 = d3 - U[3, 3] * n3
            r2 = e1 * e1 + e2 * e2 + e3 * e3
            r2 * a < 46 && (s += CS[j] * exp(-a * r2))      # exp(-46) ~ 1e-20
        end
    end
    return s
end

# The same density for 8 points at a time with SIMD.jl (`DENSITY=simd`, passed
# as `f_batch`): every lane runs the image loop around its own nearest image,
# and exp is a vectorized 2^k p(r) (|r| ≤ ln2/2, Taylor degree 12) evaluated
# only when some lane is inside the cutoff.
using SIMD
const V8 = Vec{8,Float64}

@inline function vexp(x::V8)
    k = round(x * (1 / log(2.0)))
    r = muladd(k, V8(-6.93147180369123816490e-01), x)      # Cody-Waite ln 2 = hi + lo
    r = muladd(k, V8(-1.90821492927058770002e-10), r)
    p = V8(1 / 479001600)
    for c in (1 / 39916800, 1 / 3628800, 1 / 362880, 1 / 40320, 1 / 5040, 1 / 720, 1 / 120, 1 / 24, 1 / 6, 0.5, 1.0, 1.0)
        p = muladd(p, r, V8(c))
    end
    ki = convert(Vec{8,Int64}, k)
    scale = reinterpret(V8, (ki + 1023) << 52)
    return vifelse(x < V8(-700.0), zero(V8), p * scale)
end

function density_batch!(vals::AbstractMatrix, pts::AbstractMatrix, g::CppDensity)
    Q, U, Ui = g.Q, g.U, g.Uinv
    np = size(pts, 2)
    @inbounds for i in 1:8:np
        nl = min(8, np - i + 1)
        lane(d) = V8(ntuple(k -> pts[d, i + min(k, nl) - 1], 8))
        p1, p2, p3 = lane(1), lane(2), lane(3)
        y1 = Q[1, 1] * p1 + Q[2, 1] * p2 + Q[3, 1] * p3
        y2 = Q[1, 2] * p1 + Q[2, 2] * p2 + Q[3, 2] * p3
        y3 = Q[1, 3] * p1 + Q[2, 3] * p2 + Q[3, 3] * p3
        s = zero(V8)
        for j in eachindex(g.alpha)
            a = g.alpha[j]; m = g.mrange[j]; cj = CS[j]
            d1 = y1 - g.xc[j][1]; d2 = y2 - g.xc[j][2]; d3 = y3 - g.xc[j][3]
            c1 = round(Ui[1, 1] * d1 + Ui[1, 2] * d2 + Ui[1, 3] * d3)
            c2 = round(Ui[2, 2] * d2 + Ui[2, 3] * d3)
            c3 = round(Ui[3, 3] * d3)
            e1c = d1 - (U[1, 1] * c1 + U[1, 2] * c2 + U[1, 3] * c3)    # offset of the nearest image
            e2c = d2 - (U[2, 2] * c2 + U[2, 3] * c3)
            e3c = d3 - U[3, 3] * c3
            for o3 in -m:m, o2 in -m:m, o1 in -m:m
                e1 = e1c - (U[1, 1] * o1 + U[1, 2] * o2 + U[1, 3] * o3)
                e2 = e2c - (U[2, 2] * o2 + U[2, 3] * o3)
                e3 = e3c - U[3, 3] * o3
                r2 = e1 * e1 + e2 * e2 + e3 * e3
                msk = r2 * a < V8(46.0)
                any(msk) && (s += vifelse(msk, cj * vexp(-a * r2), zero(V8)))
            end
        end
        for k in 1:nl
            vals[1, i + k - 1] = s[k]
        end
    end
    return vals
end

cell = get(ARGS, 1, "cubic")
ndig = parse(Int, get(ARGS, 2, "6"))
dist = get(ARGS, 3, "multi")
reps = parse(Int, get(ARGS, 4, "3"))
A = cpp_sorted(CELLS[cell])
f = CppDensity(A, dist == "multi" ? [20000.0, 200, 300, 400] : [60.0, 120, 180, 240])
eps = 10.0^-ndig

simd = get(ENV, "DENSITY", "scalar") == "simd"
sample_once = get(ENV, "SAMPLE_ONCE", "1") == "1"
solve() = simd ? bdmk_periodic(nothing, A; eps, sample_once, timings = tm, f_batch = (v, p) -> density_batch!(v, p, f)) :
                 bdmk_periodic(f, A; eps, sample_once, timings = tm)
tm = zeros(10)
# compile + warm up; like the C++ solver, later solves reuse the buffers
res = solve()
best = Inf; tbest = zeros(10)
for r in 1:reps
    t = @elapsed global res = solve()
    t < best && (global best = t; tbest .= tm)
end
ft = res.ft
nleaf = count(i -> ft.nchild[i] == 0 && res.istarget[i], 1:ft.nboxes)
@printf("[JULIA-PERIODIC%s%s] threads=%d cell=%s dist=%s n=%d | nboxes=%d depth=%d target leaves=%d | total=%.0f ms = sampling %.0f + rest %.0f : forest+setup+src=%.0f local+up=%.0f wide=%.0f pw=%.0f near=%.0f\n",
        simd ? " simd-density" : "", sample_once ? "" : " sample-all", Threads.nthreads(), cell, dist, ndig, ft.nboxes, ft.nlevels, nleaf,
        1e3best, 1e3tbest[1], 1e3 * (best - tbest[1]), (1e3 .* tbest[2:6])...)

function parity(res, potfile, treefile)
    ft = res.ft
    n = res.norder; np = n^3
    cpot = reinterpret(Float64, read(potfile))
    ctree = reshape(reinterpret(Float64, read(treefile)), 5, :)
    key(c, h) = (round(Int, c[1] / h * 2), round(Int, c[2] / h * 2), round(Int, c[3] / h * 2), round(Int, 1 / h))
    cidx = Dict(key(ctree[1:3, b], ctree[4, b]) => b for b in axes(ctree, 2) if ctree[5, b] == 1)
    err2 = 0.0; ref2 = 0.0; errmax = 0.0; refmax = 0.0; nmatch = 0; nleaf = 0
    for i in 1:ft.nboxes
        (ft.nchild[i] == 0 && res.istarget[i]) || continue
        nleaf += 1
        b = get(cidx, key(ft.centers[:, i], ft.boxsize[ft.level[i] + 1]), 0)
        b == 0 && continue
        nmatch += 1
        for q in 1:np
            u = res.pot[(i - 1) * np + q]; v = cpot[(b - 1) * np + q]
            err2 += (u - v)^2; ref2 += v^2
            errmax = max(errmax, abs(u - v)); refmax = max(refmax, abs(v))
        end
    end
    @printf("[PARITY] boxes C++ %d / Julia %d | matched %d / %d target leaves | rel L2 diff %.3e  max diff / max|u| %.3e\n",
            size(ctree, 2), ft.nboxes, nmatch, nleaf, sqrt(err2 / ref2), errmax / refmax)
end
length(ARGS) >= 6 && parity(res, ARGS[5], ARGS[6])
