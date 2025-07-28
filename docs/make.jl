using PortHamiltonianModelReduction
using Documenter, DocumenterCitations

DocMeta.setdocmeta!(PortHamiltonianModelReduction, :DocTestSetup, :(using PortHamiltonianModelReduction); recursive=true)

bib = CitationBibliography(joinpath(@__DIR__, "..", "CITATION.bib"))

makedocs(;
    modules=[PortHamiltonianModelReduction],
    authors="Jonas Nicodemus <jonas.nicodemus@icloud.com> and contributors",
    repo="https://github.com/Jonas-Nicodemus/PortHamiltonianModelReduction.jl/blob/{commit}{path}#{line}",
    sitename="PortHamiltonianModelReduction.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Jonas-Nicodemus.github.io/PortHamiltonianModelReduction.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API" => "API.md",
    ],
    plugins=[bib],
)

deploydocs(;
    repo="github.com/Jonas-Nicodemus/PortHamiltonianModelReduction.jl",
    devbranch="main",
)
