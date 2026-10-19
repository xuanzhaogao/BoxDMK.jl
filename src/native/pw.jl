# Plane-wave expansion parameters (Fortran `bdmk_pwterms`, `get_pwnodes_md`,
# `find_npwlevel`) and proxy order selection.

include("tables/pwnodes.jl")

# Error function from openlibm (shipped with Julia).
_erf(x::Float64) = ccall((:erf, Base.Math.libm), Float64, (Float64,), x)

const _POW10 = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12)

function _pw_digits(eps::Float64)
    for d in 2:11
        _POW10[d] < eps <= _POW10[d - 1] && return d
    end
    throw(ArgumentError("eps=$eps outside the supported range (1e-12, 1e-2]"))
end

"""Number of plane-wave terms per dimension for Gaussians handled at level `ilev`."""
function pw_terms(eps::Float64, ilev::Int)
    ilev < -10 && return 2 * ceil(Int, 2 * log(10 / eps) / pi)
    npw0 = PW_NPW0[_pw_digits(eps)]
    return 2 * npw0[min(ilev, 0) + 11]
end

"""Plane-wave nodes `ts` and weights `ws` (length `npw`) at level `ilev`."""
function pw_nodes(eps::Float64, ilev::Int, npw::Int, bs0::Float64)
    ts = zeros(npw)
    ws = zeros(npw)
    npw2 = npw ÷ 2
    if ilev < -10
        deltamax = 4.0^(-ilev) / log(10 / eps)
        h = 2pi / sqrt(deltamax * log(10 / eps))
        for i in 1:npw2
            ts[i + npw2] = h * (i - 0.5)
            ws[i + npw2] = h / 2 / sqrt(pi)
        end
    else
        t0, w0 = PW_NODES[(_pw_digits(eps), min(ilev, 0))]
        sc = ilev > 0 ? 2.0^ilev : 1.0
        for i in 1:npw2
            ts[i + npw2] = t0[i] * sc
            ws[i + npw2] = w0[i] * sc / 2 / sqrt(pi)
        end
    end
    for i in 1:npw2
        ts[npw2 - i + 1] = -ts[i + npw2]
        ws[npw2 - i + 1] = ws[i + npw2]
    end
    ts ./= bs0
    ws ./= bs0
    return ts, ws
end

const _NLEVSTART = -100

"""Level at which a Gaussian of variance `delta` is handled by plane waves."""
function find_npwlevel(eps::Float64, nlevels::Int, boxsize::Vector{Float64}, delta::Float64)
    dcutoff = sqrt(delta * log(1.0 / eps))
    bs(ilev) = ilev >= 0 ? boxsize[ilev + 1] : boxsize[1] * 2.0^(-ilev)
    npwlevel = nlevels + 1
    for i in nlevels:-1:_NLEVSTART
        if bs(i) >= dcutoff
            npwlevel = i
            break
        end
    end
    boxsize[nlevels + 1] > dcutoff && (npwlevel = nlevels)
    bs(_NLEVSTART) < dcutoff && throw(ArgumentError("Gaussian too wide: npwlevel < $_NLEVSTART"))
    return npwlevel
end

"""Proxy polynomial order for precision `eps`."""
function select_porder(eps::Real)
    eps >= 0.8e-3 && return 16
    eps >= 0.8e-4 && return 22
    eps >= 0.8e-5 && return 26
    eps >= 0.8e-6 && return 30
    eps >= 0.8e-7 && return 36
    eps >= 0.8e-8 && return 42
    eps >= 0.8e-9 && return 46
    eps >= 0.8e-10 && return 50
    eps >= 0.8e-11 && return 56
    eps >= 0.8e-12 && return 62
    throw(ArgumentError("eps=$eps is below the smallest supported precision"))
end
