# Sum-of-Gaussians kernel approximation (Fortran `get_sognodes`) and the
# near-field Taylor correction coefficients (bdmk STEP 2).

include("tables/sognodes.jl")

function _sog_band(eps::Float64, thresholds)
    for (k, t) in enumerate(thresholds)
        eps >= t && return k
    end
    throw(ArgumentError("eps=$eps is below the smallest supported precision (1e-12)"))
end

function _y3dsognodes(beta::Float64, rmin::Float64, eps::Float64)
    h = (eps > 1e-7 ? 0.7 : 0.65) * 2pi / log(1.2 / eps)
    tlower = -log(2.0 / beta * sqrt(log(1.2 / eps)))
    tupper = log(1.0 / rmin * sqrt(log(1.2 / eps)))
    mmin = floor(Int, tlower / h)
    mmax = ceil(Int, tupper / h)
    c0 = 2.0 / sqrt(pi)
    ts = Float64[]
    ws = Float64[]
    for i in mmin:mmax
        t = exp(-2 * h * i)
        push!(ts, t)
        push!(ws, exp(-beta^2 / 4 * t) / sqrt(t) * h * c0)
    end
    return ws, ts
end

"""
    sog_nodes(kernel, eps, bsize, nlevels, norder) -> (r0, weights, deltas)

Sum-of-Gaussians approximation `K(r) ≈ Σ w_i exp(-r²/δ_i)` for `r ≥ r0` of the
3D kernel on a root box of size `bsize`.
"""
function sog_nodes(::LaplaceKernel, eps::Float64, bsize::Float64, nlevels::Int, norder::Int)
    ts, ws = SOG_TABLES[:l3dsognodes][_sog_band(eps, (0.9e-3, 0.9e-6, 0.9e-9, 0.9e-12))]
    return bsize / 2^16, ws ./ bsize, ts .* bsize^2
end

function sog_nodes(::SqrtLaplaceKernel, eps::Float64, bsize::Float64, nlevels::Int, norder::Int)
    ts, ws = SOG_TABLES[:sl3dsognodes][_sog_band(eps, (0.9e-3, 0.9e-6, 0.9e-9, 0.9e-12))]
    return bsize / 2^16, ws ./ bsize^2, ts .* bsize^2
end

function sog_nodes(kernel::YukawaKernel, eps::Float64, bsize::Float64, nlevels::Int, norder::Int)
    rmin = bsize / 2^nlevels / norder^2
    ws, ts = _y3dsognodes(Float64(kernel.beta), rmin, eps)
    return rmin, ws, ts
end

# fout[k+1] = ∫_0^a exp(-t²/δ) t^(2k+2) dt, k = 0:ntaylor (Fortran `faddeevan_3d`).
function faddeevan_3d(ntaylor::Int, a::Float64, delta::Float64)
    fout = zeros(ntaylor + 1)
    x = a / sqrt(delta)
    x2 = x * x
    c = zeros(ntaylor + 1)
    c[1] = delta^1.5
    for i in 1:ntaylor
        c[i + 1] = c[i] * delta
    end
    if x < 1.0
        fac = 1.0
        d = x^3
        sgn = 1.0
        terms = zeros(21)
        for i in 0:20
            i > 0 && (fac *= i; d *= x2)
            terms[i + 1] = sgn * d / fac
            sgn = -sgn
        end
        for k in 0:ntaylor
            s = 0.0
            for i in 0:20
                s += terms[i + 1] * x2^k / (2i + 3 + 2k)
            end
            fout[k + 1] = s * c[k + 1]
        end
    else
        sqpi = sqrt(pi)
        erfx = _erf(x)
        expx2 = exp(-x2)
        x4 = x2 * x2
        x6 = x4 * x2
        fout[1] = c[1] * (erfx * sqpi / 2 - x * expx2) / 2
        ntaylor >= 1 && (fout[2] = c[2] * (3 * sqpi * erfx / 8 - x * expx2 * (x2 / 2 + 0.75)))
        ntaylor >= 2 && (fout[3] = c[3] * (15 * sqpi * erfx / 16 - x * expx2 * (15 / 8 + 5 * x2 / 4 + x4 / 2)))
        ntaylor >= 3 && (fout[4] = c[4] * (105 * sqpi * erfx / 32 - x * expx2 * (105 / 16 + 35 * x2 / 8 + 7 * x4 / 4 + x6 / 2)))
    end
    return fout
end

# c(k) = ∫_0^r0 K(r) r^(2k+2) dr for k = 0, 1, 2.
function _kernel_moments(kernel::YukawaKernel, r0::Float64)
    beta = Float64(kernel.beta)
    br = beta * r0
    br2 = br * br; br3 = br2 * br; br4 = br3 * br; br5 = br4 * br
    br6 = br5 * br; br7 = br6 * br; br8 = br7 * br; br9 = br8 * br
    b2 = beta * beta; b4 = b2 * b2; b6 = b2 * b4
    ebr = exp(-br)
    if br > 1e-3
        return [1 / b2 - ebr * (br + 1) / b2,
                6 / b4 - ebr * (br3 + 3br2 + 6br + 6) / b4,
                120 / b6 - ebr * (br5 + 5br4 + 20br3 + 60br2 + 120br + 120) / b6]
    else
        return [(br2 / 2 - br3 / 3 + br4 / 8 - br5 / 30 + br6 / 144) / b2,
                (br4 / 4 - br5 / 5 + br6 / 12 - br7 / 42) / b4,
                (br6 / 6 - br7 / 7 + br8 / 16 - br9 / 54) / b6]
    end
end
_kernel_moments(::LaplaceKernel, r0::Float64) = [r0^(2k + 2) / (2k + 2) for k in 0:2]
_kernel_moments(::SqrtLaplaceKernel, r0::Float64) = [r0^(2k + 1) / (2k + 1) for k in 0:2]

"""
    taylor_coefficients(kernel, r0, weights, deltas) -> (c0, c1, c2)

Coefficients of the local correction `c0 f + c1 Δf + c2 Δ²f` accounting for
the difference between the kernel and its SOG approximation on `[0, r0]`.
"""
function taylor_coefficients(kernel::AbstractKernel, r0::Float64, ws::Vector{Float64}, ts::Vector{Float64})
    c = _kernel_moments(kernel, r0)
    for i in eachindex(ts)
        fv = faddeevan_3d(2, r0, ts[i])
        c .-= ws[i] .* fv
    end
    c[1] *= 4pi
    c[2] *= 4pi / 2 / 3
    c[3] *= 4pi / 24 / 5
    return c[1], c[2], c[3]
end
