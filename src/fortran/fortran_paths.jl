const _EXTERNAL_FORTRAN_LIBBOXDMK_PATH = normpath("/mnt/home/xgao1/codes/boxdmk/build/libboxdmk.so")
const _BUILD_FORTRAN_REF_CMD = "julia --project deps/build_fortran_ref.jl"

_vendored_fortran_root() = normpath(joinpath(_PACKAGE_ROOT, "deps", "boxdmk_fortran"))
_vendored_fortran_build_root() = normpath(joinpath(_PACKAGE_ROOT, "deps", "usr"))
# BOXDMK_FORTRAN_LIBDIR overrides the location of the reference libraries.
_vendored_fortran_libdir() = get(ENV, "BOXDMK_FORTRAN_LIBDIR", joinpath(_vendored_fortran_build_root(), "lib"))

function _fortran_library_candidates(; debug::Bool = false)
    basename = debug ? "libboxdmk_debug.so" : "libboxdmk.so"

    candidates = String[
        normpath(joinpath(_vendored_fortran_libdir(), basename)),
    ]

    if !debug
        push!(candidates, _EXTERNAL_FORTRAN_LIBBOXDMK_PATH)
    end

    return candidates
end

function _fortran_solve_library_candidates(; debug::Bool = false)
    debug && return _fortran_library_candidates(; debug = true)

    return String[
        normpath(joinpath(_vendored_fortran_libdir(), "libboxdmk_hot.so")),
        _EXTERNAL_FORTRAN_LIBBOXDMK_PATH,
        normpath(joinpath(_vendored_fortran_libdir(), "libboxdmk.so")),
    ]
end

function _resolve_fortran_library_path(; debug::Bool = false, must_exist::Bool = false)
    candidates = _fortran_library_candidates(; debug = debug)

    for candidate in candidates
        isfile(candidate) && return candidate
    end

    must_exist && throw(ArgumentError("Fortran library not found. Checked: $(join(candidates, ", "))"))
    return first(candidates)
end

function _resolve_fortran_solve_library_path(; debug::Bool = false, must_exist::Bool = false)
    candidates = _fortran_solve_library_candidates(; debug = debug)

    for candidate in candidates
        isfile(candidate) && return candidate
    end

    must_exist && throw(ArgumentError("Fortran solve library not found. Checked: $(join(candidates, ", "))"))
    return first(candidates)
end

function _require_fortran_solve_library!(; path::AbstractString = _resolve_fortran_solve_library_path())
    isfile(path) && return path

    candidates = _fortran_solve_library_candidates()
    throw(ArgumentError(
        "Required Fortran solve library not found at $(path). Checked: $(join(candidates, ", ")). Build it with `$_BUILD_FORTRAN_REF_CMD`.",
    ))
end
