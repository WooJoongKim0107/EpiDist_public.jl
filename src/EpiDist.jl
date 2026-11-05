module EpiDist
export Commons, Nash, Ctrl, Adpt

include("commons.jl")
using .Commons


module Nash
    using ..Commons
    include("nash.jl")
    using PropertyArrays: pload
    datapath = Dict(
        :fine     => "../rsrc/nash_fine.jld2",
        :wider    => "../rsrc/nash_wsparse.jld2",
        :widerpp  => "../rsrc/nash_wide.jld2",
        :wide     => "../rsrc/nash_wide.jld2",
        :wsparse  => "../rsrc/nash_wsparse.jld2",
    )
    function load(name::Symbol=:main)
        haskey(datapath, name) && return pload(Record, datapath[name])
        name === :expt && return pload(Record, datapath[:wsparse])[vcat(1, 2:3:end), :]
        throw(KeyError(name))
    end
end

module Ctrl
    using ..Commons
    include("controlled.jl")
    using PropertyArrays: pload
    datapath = Dict(
        :fine     => "../rsrc/ctrl_fine.jld2",
        :wider    => "../rsrc/ctrl_wsparse.jld2",
        :widerpp  => "../rsrc/ctrl_wide.jld2",
        :wide     => "../rsrc/ctrl_wide.jld2",
        :wsparse  => "../rsrc/ctrl_wsparse.jld2",
    )
    function load(name::Symbol=:main)
        haskey(datapath, name) && return pload(Record, datapath[name])
        name === :expt && return pload(Record, datapath[:wsparse])[vcat(1, 2:3:end), :]
        throw(KeyError(name))
    end
end

module Adpt
    using ..Commons
    include("adaptive.jl")
    using PropertyArrays: pload
    using JLD2: load_object
    datapath = Dict(
        :fine         => "../rsrc/adpt_fine.jld2",
        :most_public  => "../rsrc/data/adpt_manifold_most_public.jld2",
        :least_public => "../rsrc/data/adpt_codex_i0_least_public.jld2",
        :snapping     => "../rsrc/data/adpt_codex_i0_snapping.jld2",
        :greedy       => "../rsrc/data/adpt_codex_i0_greedy.jld2",
    )
    load(name::Symbol=:main) = pload(Record, datapath[name])
    cherrypick(field::Symbol, version::Symbol=:main) = load_object("../rsrc/cherry/$(field)_$(version).jld2")
    include("adaptive_forward.jl")
    include("adaptive_temporal.jl")
    include("adaptive_eta.jl")
    using .Eta: eta_source, eta_response, eta_sweep, eta_timeseries
end

using .Adpt: Forward
export Forward

import .Adpt.Eta: eta_sweep, eta_timeseries
export eta_sweep, eta_timeseries

"""Solve one eta value for an existing AO Record and return its physical-time series."""
function eta_timeseries(rec::Adpt.Record; eta=1., n=1201, kwargs...)
    result=only(eta_sweep(rec; etas=[eta], kwargs...))
    eta_timeseries(result; n)
end


module Null
    using ..Commons
    include("null.jl")
end

module Expt
    using ..Commons
    include("exploit.jl")
end

module Expt3
    using ..Commons
    include("exploit3.jl")
end

end
