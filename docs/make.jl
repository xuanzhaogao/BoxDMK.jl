using BoxDMK
using Documenter

DocMeta.setdocmeta!(BoxDMK, :DocTestSetup, :(using BoxDMK); recursive = true)

makedocs(;
    modules = [BoxDMK],
    authors = "Xuanzhao Gao <xgao@flatironinstitute.org> and contributors",
    sitename = "BoxDMK.jl",
    format = Documenter.HTML(;
        canonical = "https://xuanzhaogao.github.io/BoxDMK.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Free-space potentials" => "freespace.md",
        "Periodic potentials" => "periodic.md",
        "Performance" => "performance.md",
        "Implementation notes" => "implementation.md",
        "API reference" => "api.md",
    ],
    checkdocs = :exports,
)

deploydocs(;
    repo = "github.com/xuanzhaogao/BoxDMK.jl",
    devbranch = "main",
)
