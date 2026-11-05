# Manuscript figure data

[`results/`](results/) contains the data used to plot the manuscript figures,
including the full 555,800-point grid.

## Regenerate the data

Clone this repository and enter the checkout.

```sh
git clone https://github.com/WooJoongKim0107/EpiDist_public.jl.git
cd EpiDist_public.jl
```

The reproduction code is written in Julia. If Julia is not installed, install
[Juliaup](https://julialang.org/install/) and run:

```sh
juliaup add 1.11.7
```

From the checkout, start Julia with the repository's project environment:

```sh
julia +1.11.7 --project=.
```

If you installed Julia 1.11.7 without Juliaup, use `julia --project=.`.

In Julia, install the dependencies and run the calls for the figures you need.

```julia
using Pkg
Pkg.instantiate()
include("publication/results.jl")

write_figure1("figure1.csv")
write_figure4("figure4.csv")
write_figure5("figure5.csv")
write_grid_figures(".", 1.1:0.1:15.0, 1:100)
```

The last call writes the CSVs for Figs. 2, 3, 6, and 7 on a coarser
14,000-point grid.
