using Documenter
using TensorOperator

makedocs(
    sitename = "TensorOperator",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    modules = [TensorOperator],
    pages = [
        "Manual"=>"index.md",
        "API"=>"api.md"
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/jmwjc/TensorOperator.jl.git",  
)
