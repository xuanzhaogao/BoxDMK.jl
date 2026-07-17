using BoxDMK
using Documenter

DocMeta.setdocmeta!(BoxDMK, :DocTestSetup, :(using BoxDMK); recursive=true)

makedocs(;
    modules=[BoxDMK],
    authors="Xuanzhao Gao <xgao@flatironinstitute.org> and contributors",
    sitename="BoxDMK.jl",
    format=Documenter.HTML(;
        canonical="https://xuanzhaogao.github.io/BoxDMK.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/xuanzhaogao/BoxDMK.jl",
    devbranch="main",
)
