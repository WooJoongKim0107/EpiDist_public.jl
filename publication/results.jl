include("equilibria.jl")
include("../src/ao_lf/channels_grid.jl")
include("timeseries.jl")
include("deviators.jl")

csvrow(io, values) = println(io, join((ismissing(x) ? "" : x for x in values), ','))

function write_grid_figures(directory, r0s, costs)
    mkpath(directory)
    terms = (
        "ir_i",
        "ir_p",
        "futility_i",
        "futility_p",
        "post_futility_i",
        "post_futility_p",
        "free_riding",
        "direct_substitution",
    )
    paths = [joinpath(directory, "figure$n.csv") for n in (2, 3, 6, 7)]
    fig2, fig3, fig6, fig7 = outputs = (IOBuffer(), IOBuffer(), IOBuffer(), IOBuffer())
    csvrow(fig2, ("r0", "c", "prepeak_private_share", "late_public_share"))
    csvrow(fig3, ("r0", "c", "ao_minus_lf_private", "lf_minus_co_investment"))
    csvrow(
        fig6,
        (
            "r0",
            "c",
            "delta_private",
            "dominant",
            "dominant_split",
            ("coarse_" .* terms)...,
            ("fine_" .* terms)...,
        ),
    )
    csvrow(fig7, ("r0", "c", "lf_cost_over_co", "ao_cost_over_co", "ao_public"))
    for c in costs, r0 in r0s
        lf, co, ao = equilibria(r0, c)
        timing = Adpt.Temporal.measure(ao)
        channels = channel_point(ao, c * lf.j1)
        csvrow(fig2, (r0, c, timing.prepeak_private_share, timing.late_public_realized))
        csvrow(fig3, (r0, c, ao.j1v - lf.j1, lf.j1 - co.j1))
        csvrow(
            fig6,
            (
                r0,
                c,
                channels.delta,
                channels.full.channel,
                channels.split.channel,
                channels.coarse...,
                channels.fine...,
            ),
        )
        csvrow(fig7, (r0, c, lf.obj / co.obj, ao.obj / co.obj, ao.j1p))
    end
    for (path, output) in zip(paths, outputs)
        write(path, take!(output))
    end
    return paths
end
