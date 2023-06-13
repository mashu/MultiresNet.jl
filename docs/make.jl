using MultiresNet
using Documenter

DocMeta.setdocmeta!(MultiresNet, :DocTestSetup, :(using MultiresNet); recursive=true)

makedocs(;
    modules=[MultiresNet],
    authors="Mateusz Kaduk <mateusz.kaduk@gmail.com> and contributors",
    repo="https://gitlab.com/mateusz-kaduk/MultiresNet.jl/blob/{commit}{path}#{line}",
    sitename="MultiresNet.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://mateusz-kaduk.gitlab.io/MultiresNet.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
