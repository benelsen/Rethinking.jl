using Documenter, Rethinking

makedocs(
    modules = [Rethinking],
    format = :html,
    sitename = "Rethinking.jl",
    pages = Any["index.md"]
)

deploydocs(
    repo = "github.com/benelsen/Rethinking.jl.git",
    target = "build",
    julia = "1.0",
    deps = nothing,
    make = nothing,
)
