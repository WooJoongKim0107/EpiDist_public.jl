include("channels.jl")

function groups(v; split=false)
    ir = v[1] + v[2]
    fu = v[3] + v[4]
    post = v[5] + v[6]
    split ? [ir, v[7], v[8], fu - post, post] : [ir, v[7], v[8], fu]
end

function winner(v; direction=1)
    score = direction .* v
    m = maximum(score)
    m <= TOL ? 0 : count(==(m), score) > 1 ? -1 : argmax(score)
end

function labels(a, b, regime; split=false, aligned=false)
    aligned && regime == 0 && return 0
    d = aligned ? regime : 1
    x = winner(groups(a; split); direction=d)
    y = winner(groups(b; split); direction=d)
    x == y ? x : -1
end

function estimate(rec, N, step)
    src = PostChannels.PD.path_source(rec)
    events = PostChannels.public_events(src)
    base = nothing
    vals = zeros(8)
    for j = N:-1:0
        x = j / N
        eta = x^4
        # Start both resolutions independently at eta=1; only continuation within
        # the same resolution is used. No point-specific numerical refinement.
        base = try
            PostChannels.response(src, eta; initial=base, step, events)
        catch
            isnothing(base) && rethrow()
            previous = base
            for u in range(base.eta, eta; length=9)[2:end]
                previous = PostChannels.response(src, u; initial=previous, step, events)
            end
            previous
        end
        z = PostChannels.channels(base)
        w = (j == 0 || j == N ? 1 : isodd(j) ? 4 : 2) / (3N)
        vals .+= (w * 4x^3) .* z.input
    end
    vals
end

const TOL = 1e-6
const CHANNELS = (:interruption, :free_riding, :direct_substitution, :futility)
const SPLIT_CHANNELS =
    (:interruption, :free_riding, :direct_substitution, :futility_before_end, :futility_post)

function channel_point(rec, lf)
    delta = rec.c * rec.j1v - lf
    regime = abs(delta) <= TOL ? 0 : Int(sign(delta))
    src = PostChannels.PD.path_source(rec)
    S0, I0 = EpiDist.Commons.initial_susceptible, EpiDist.Commons.initial_infected
    peak = I0 + S0 - 1 / rec.r0 - log(rec.r0 * S0) / rec.r0
    zero_channels =
        !src.public_seen || (rec.r0 * rec.c * peak < 1 - 1e-8 && max(lf, rec.j1v * rec.c) < TOL)
    coarse, fine = if zero_channels
        (zeros(8), zeros(8))
    else
        try
            (estimate(rec, 16, 0.12), estimate(rec, 32, 0.06))
        catch err
            err isa InterruptException && rethrow()
            @warn "Channel calculation failed" r0 = rec.r0 c = rec.c exception = err
            (fill(NaN, 8), fill(NaN, 8))
        end
    end
    function classify(names; split=false)
        regime == 0 && return (; channel=:neutral)
        all(isfinite, coarse) && all(isfinite, fine) || return (; channel=:unclassified)
        i = labels(coarse, fine, regime; split, aligned=true)
        (; channel=i > 0 ? names[i] : :unclassified)
    end
    (;
        coarse,
        fine,
        delta,
        regime,
        full=classify(CHANNELS),
        split=classify(SPLIT_CHANNELS; split=true),
    )
end
