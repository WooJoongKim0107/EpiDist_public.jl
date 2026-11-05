function write_figure1(path)
    open(path, "w") do io
        csvrow(
            io,
            (
                "r0",
                "c",
                "policy",
                "t",
                "infected_fraction",
                "public_rate",
                "private_rate",
                "public_cumulative",
                "private_cumulative",
            ),
        )
        for (r0, c) in ((1.5, 40), (5.0, 60))
            lf, _, ao = equilibria(r0, c)
            ao = Forward.lreproduce(ao)
            lf = Nash.lreproduce(lf)
            null = EpiDist.Null.shoot(r0; tmax=100)
            ao_private = ao.varphi .* ao.S ./ ao.I ./ r0
            lf_private = lf.varphi .* lf.S ./ lf.I ./ r0
            for i in eachindex(ao.t)
                csvrow(
                    io,
                    (
                        r0,
                        c,
                        "AO",
                        ao.t[i],
                        ao.I[i] / r0,
                        ao.p[i],
                        ao.v[i],
                        ao.X[6, i] / r0,
                        ao_private[i] - first(ao_private),
                    ),
                )
            end
            for i in eachindex(lf.t)
                csvrow(
                    io,
                    (
                        r0,
                        c,
                        "LF",
                        lf.t[i],
                        lf.I[i] / r0,
                        0,
                        lf.action[i],
                        0,
                        lf_private[i] - first(lf_private),
                    ),
                )
            end
            for i in eachindex(null.t)
                csvrow(io, (r0, c, "Null", null.t[i], null[2, i] / r0, 0, 0, 0, 0))
            end
        end
    end
    return path
end
