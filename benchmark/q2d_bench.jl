# Quasi-2D benchmark: a 100 × 100 × 1 slab (periodic in x, y) with a neutral
# mixture of K Gaussians (exponents 300–1000, centres in 0.4 ≤ z ≤ 0.6),
# evaluated through an in-plane cell list. Reports the sampling / rest split,
# the number of in-slab points and the error against the 2-periodic Ewald sum.
#
#   julia --project -t 48 benchmark/q2d_bench.jl [eps digits] [K] [reps] [rc]

using BoxDMK, Printf
include(joinpath(@__DIR__, "..", "test", "q2d_oracle.jl"))

const L = (100.0, 100.0)
const ZL = (0.0, 1.0)

struct BinnedMixture
    g::Q2DMixture
    nb::NTuple{2,Int}
    h::NTuple{2,Float64}
    bins::Matrix{Vector{Int}}
end

function BinnedMixture(g::Q2DMixture)
    rcut = sqrt(46 / minimum(g.alpha))
    nb = (max(1, floor(Int, g.L[1] / rcut)), max(1, floor(Int, g.L[2] / rcut)))
    h = (g.L[1] / nb[1], g.L[2] / nb[2])
    bins = [Int[] for _ in 1:nb[1], _ in 1:nb[2]]
    for (j, x) in enumerate(g.x)
        push!(bins[1 + floor(Int, mod(x[1], g.L[1]) / h[1]), 1 + floor(Int, mod(x[2], g.L[2]) / h[2])], j)
    end
    return BinnedMixture(g, nb, h, bins)
end

function (m::BinnedMixture)(p::AbstractVector)
    g = m.g
    i0 = floor(Int, mod(p[1], g.L[1]) / m.h[1]); j0 = floor(Int, mod(p[2], g.L[2]) / m.h[2])
    s = 0.0
    @inbounds for dj in -1:1, di in -1:1
        for j in m.bins[mod(i0 + di, m.nb[1]) + 1, mod(j0 + dj, m.nb[2]) + 1]
            a = g.alpha[j]
            d1 = p[1] - g.x[j][1]; d2 = p[2] - g.x[j][2]; d3 = p[3] - g.x[j][3]
            d1 -= g.L[1] * round(d1 / g.L[1]); d2 -= g.L[2] * round(d2 / g.L[2])
            r2 = d1^2 + d2^2 + d3^2
            r2 * a < 46 && (s += g.c[j] * exp(-a * r2))
        end
    end
    return s
end

nd = parse(Int, get(ARGS, 1, "6"))
K = parse(Int, get(ARGS, 2, "64"))
reps = parse(Int, get(ARGS, 3, "2"))
rc = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : nothing
eps = 10.0^-nd

# deterministic random mixture
seed = Ref(UInt64(12345))
rnd() = (seed[] = seed[] * 0x5851f42d4c957f2d + 0x14057b7ef767814f; (seed[] >> 11) / 2.0^53)
xs = [(L[1] * rnd(), L[2] * rnd(), 0.4 + 0.2 * rnd()) for _ in 1:K]
al = [300.0 * (1000 / 300)^rnd() for _ in 1:K]             # 300 … 1000 (resolved by the default nodes; vanish outside the slab)
cs = [rnd() < 0.5 ? -1.0 : 1.0 for _ in 1:K]
g = neutral_q2d_mixture(L, xs, al, cs)
f = BinnedMixture(g)

tm = zeros(10)
solve() = bdmk_q2d(f, L, ZL; eps, timings = tm, (rc === nothing ? (;) : (; rc))...)
res = solve()
best = Inf; tb = zeros(10)
for _ in 1:reps
    t = @elapsed global res = solve()
    t < best && (global best = t; tb .= tm)
end
ft, n = res.ft, res.norder
leaves = [i for i in 1:ft.nboxes if ft.nchild[i] == 0 && res.istarget[i]]
npts = n^3 * length(leaves)
rest = best - tb[1]
@printf("[Q2D] threads=%d L=%gx%gx%g K=%d eps=1e-%d rc=%g mgrid=%s | nboxes=%d target leaves=%d points=%.1fM | total=%.0f ms = sampling %.0f + rest %.0f : forest+setup=%.0f local+up=%.0f wide=%.0f pw=%.0f near=%.0f | rest %.0f ns/pt, %.1f Mpts/s\n",
        Threads.nthreads(), L..., ZL[2] - ZL[1], K, nd, res.grid.rc, string(res.grid.mgrid), ft.nboxes, length(leaves), npts / 1e6,
        1e3best, 1e3tb[1], 1e3rest, (1e3 .* tb[2:6])..., 1e9rest / npts, npts / rest / 1e6)

# error at the node nearest to some Gaussian centres (and random nodes) against Ewald
x = BoxDMK.ortho1d(LegendreBasis(), n).x
umax = maximum(i -> maximum(abs, view(res.pot, :, :, :, i)), leaves)
emax = 0.0; e2 = 0.0; r2 = 0.0
for t in 1:40
    xc = xs[1 + (t - 1) % K]
    p = [mod(xc[1] + 0.05 * (rnd() - 0.5), L[1]), mod(xc[2] + 0.05 * (rnd() - 0.5), L[2]), xc[3] + 0.05 * (rnd() - 0.5)]
    u = evaluate_periodic(res, p)
    uref = q2d_potential(g, p)
    global emax = max(emax, abs(u - uref)); global e2 += (u - uref)^2; global r2 += uref^2
end
@printf("[Q2D-ERR] 40 points near the charges vs 2P Ewald: rel L2 %.2e, max / max|u| %.2e (eps %.0e)\n", sqrt(e2 / r2), emax / umax, eps)
