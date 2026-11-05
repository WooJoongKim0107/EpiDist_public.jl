function public_events(src)
    !src.public_seen && return Float64[]
    b = EpiDist.Adpt.Forward.shoot(src.rec; tol=1e-13, save_everystep=true)
    shift = src.total_end - last(b.sol.t)
    ts = sort(unique(b.sol.t))
    windows = Tuple{Float64,Float64}[]
    for (l, r) in zip(ts[1:end-1], ts[2:end])
        EpiDist.Adpt.Forward.action(b.sol((l + r) / 2))[1] > 0 || continue
        if !isempty(windows) && abs(last(windows)[2] - (l + shift)) < 1e-9
            windows[end] = (last(windows)[1], r + shift)
        else
            push!(windows, (l + shift, r + shift))
        end
    end
    sort(unique(vcat([collect(range(l, r; length=17)) for (l, r) in windows]...)))
end

function public_cuts(out, k)
    src = out.source
    lo = k == 1 ? 0.0 : src.split
    L = k == 1 ? src.split : src.T - src.split
    extra = [
        begin
            x = clamp((t - lo) / L, 0.0, 1.0)
            x < 1e-12 ? 0.0 : x > 1 - 1e-12 ? 1.0 : x
        end for t in out.public_nodes if lo < t < lo + L
    ]
    sort(unique(vcat(PD.active_cuts(out, k), extra)))
end

function response(src, lam; initial=nothing, step=0.12, events=public_events(src))
    nmin = max(600, ceil(Int, 72 / step))
    # Bracket internal on/off events from both sides; phase-boundary jumps
    # already use the solver's separate one-sided phases.
    events = copy(events)
    for e in copy(events)[vcat(collect(1:17:length(events)), collect(17:17:length(events)))]
        minimum(abs.([e, e - src.split, e - src.T])) > 1e-7 || continue
        append!(events, [max(0.0, e - 1e-9), min(src.T, e + 1e-9)])
    end
    sort!(unique!(events))
    out = if !isnothing(initial) && hasproperty(initial, :public_nodes)
        PD.refresh_solution(
            merge(initial, (; eta=Float64(lam))),
            PD.solve_mesh(src, lam, initial, initial.sol.x),
        )
    else
        EA.eta_response(src, lam; initial, max_step=step, intervals=nmin, tol=1e-11)
    end
    L = max(src.split, src.T - src.split)
    n = max(nmin, ceil(Int, L / step))
    uniform = collect(range(0, 1; length=n + 1))
    fixed = Float64[]
    for k = 1:2
        lo = k == 1 ? 0.0 : src.split
        len = k == 1 ? src.split : src.T - src.split
        append!(fixed, [clamp((t - lo) / len, 0.0, 1.0) for t in events if lo < t < lo + len])
    end
    for iteration = 1:10
        roots = [PD.active_cuts(out, k)[2:end-1] for k = 1:2]
        added = sort(unique(vcat(fixed, roots...)))
        keep = [x for x in uniform if x == 0 || x == 1 || all(abs(x - r) > 0.08 / n for r in added)]
        xs = sort(unique(vcat(0.0, keep, added, 1.0)))
        # Merge only roundoff-scale near duplicates.
        x = Float64[first(xs)]
        for t in xs[2:end]
            t - last(x) > 1e-13 && push!(x, t)
        end
        x[end] = 1.0
        for j = length(x)-1:-1:1
            (x[j+1] - x[j]) * L > step && insert!(x, j + 1, (x[j] + x[j+1]) / 2)
        end
        x == out.sol.x || (out = PD.refresh_solution(out, PD.solve_mesh(src, lam, out, x)))
        updated = [PD.active_cuts(out, k)[2:end-1] for k = 1:2]
        if all(length(roots[k]) == length(updated[k]) for k = 1:2)
            e = maximum(
                [
                    0.0
                    [
                        abs(a - b) * (k == 1 ? src.split : src.T - src.split) for k = 1:2 for
                        (a, b) in zip(roots[k], updated[k])
                    ]
                ],
            )
            if e < 1e-7
                out = merge(
                    out,
                    (; public_nodes=events, alignment_error=e, alignment_iterations=iteration),
                )
                return merge(
                    out,
                    (; metrics=PD.aligned_metrics(out, [public_cuts(out, k) for k = 1:2])),
                )
            end
        end
    end
    error("Public-event mesh alignment failed")
end

function state_slope(out)
    src = out.source
    dy = sensitivity(out)
    value = 0.0
    for k = 1:2
        value += first(quadgk(public_cuts(out, k)...; rtol=1e-8, atol=1e-11) do x
            a = PD.at(out, x, k)
            y = dy(x)
            dI = a.I * y[3k-1]
            dq = src.c * y[3k]
            dp = EA.policy(src, 1.0, a.t, k)
            dv = a.v > 0 ? a.R / 2 * (a.q * dI + a.I * dq) - dp : 0.0
            a.L * a.S * (dv + a.v * y[3k-2])
        end)
    end
    value
end
