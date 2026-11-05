function deviators(background, r0, c, costs)
    model = background === Ctrl ? EpiDist.Expt : EpiDist.Expt3
    guess = 0.0
    map(costs) do cpop
        rec =
            cpop <= background.min_c(r0) ? background.Record(r0, cpop) :
            background.aimed_shot(r0, cpop)
        dc, dvalue, dinvestment = rec.next
        motivation = background === Ctrl ? rec.λ0 : rec.q0
        result = model.aimed_shot(
            r0,
            c,
            cpop,
            motivation,
            guess;
            branch=rec.branch,
            dcpop=dc,
            dλ0=dvalue,
            dJ1pop=dinvestment,
        )
        guess = result.q0
        result
    end
end

function write_figure4(path; costs=vcat(1:200, 205:5:1000))
    open(path, "w") do io
        csvrow(
            io,
            (
                "r0",
                "c_dev",
                "background",
                "c_pop",
                "a_pop",
                "a_dev",
                "interruption",
                "free_riding",
                "futility",
                "futility_tail",
                "total",
            ),
        )
        for (r0, c) in ((9.0, 10), (3.0, 50))
            for rec in deviators(Ctrl, r0, c, costs)
                abstol = r0 == 9.0 ? 1e-15 : 1e-13
                partial = EpiDist.Expt.PartialA(rec; intkwargs=(; abstol, reltol=1e-12))
                csvrow(
                    io,
                    (
                        r0,
                        c,
                        "CO",
                        rec.cpop,
                        rec.J1pop,
                        rec.J1,
                        partial.sensitivity_futility,
                        partial.sensitivity_adjust,
                        partial.sensitivity_revaluation,
                        partial.sensitivity_revaluation_tail,
                        partial.sensitivity,
                    ),
                )
            end
            for rec in deviators(Nash, r0, c, first(costs, min(length(costs), 300)))
                csvrow(
                    io,
                    (
                        r0,
                        c,
                        "LF",
                        rec.cpop,
                        rec.J1pop,
                        rec.J1,
                        missing,
                        missing,
                        missing,
                        missing,
                        missing,
                    ),
                )
            end
        end
    end
    return path
end

function write_figure5(path)
    open(path, "w") do io
        csvrow(io, ("r0", "c_pop", "background", "t", "distancing_end", "dI_dc", "dS_dc"))
        for (background, label, model) in ((Ctrl, "CO", EpiDist.Expt), (Nash, "LF", EpiDist.Expt3))
            for rec in deviators(background, 3.0, 1, [50, 500])
                partial = model.PartialX(rec)
                for (t, dI, dS) in zip(partial.t, partial.∂Ipt, partial.∂Spt)
                    csvrow(io, (3.0, rec.cpop, label, t, rec.tfpop, dI, dS))
                end
            end
        end
    end
    return path
end
