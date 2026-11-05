# Investment-presence evidence is recorded only when an actual mode interval
# ends (switch/deactivation/solve completion), never during RHS evaluations.
struct InvestmentWitness
    mode::Symbol
    start_time::Float64
    stop_time::Float64
    start_motivation::Float64
    stop_motivation::Float64
    start_growth::Float64
    reason::Symbol
end

mutable struct InvestmentPresence
    mode::Symbol
    start_time::Float64
    start_motivation::Float64
    start_growth::Float64
    any_p::Bool
    any_v::Bool
    public_witness::Union{Nothing,InvestmentWitness}
    private_witness::Union{Nothing,InvestmentWitness}
end

_presence_motivation(mode, u) =
    mode === :public ? Float64(u[4]) : mode === :private ? Float64(u[3]) : 0.0

function _presence_start!(flight, u, t)
    level = _presence_motivation(flight.mode, u)
    growth = Float64(u[1] + u[2] - 1)
    if isnothing(flight.presence)
        flight.presence = InvestmentPresence(
            flight.mode,
            Float64(t),
            level,
            growth,
            false,
            false,
            nothing,
            nothing,
        )
    else
        p = flight.presence
        p.mode = flight.mode
        p.start_time = Float64(t)
        p.start_motivation = level
        p.start_growth = growth
    end
    nothing
end

function _presence_close!(flight, u, t)
    p = flight.presence
    isnothing(p) && return nothing
    # Equal-time callbacks and a zero-duration solve cannot certify investment.
    t > p.start_time || return nothing
    p.mode in (:public, :private) || return nothing
    stop_level = _presence_motivation(p.mode, u)
    reason =
        _level(p.start_motivation) > 0 ? :positive_at_start :
        _level(stop_level) > 0 ? :positive_at_stop :
        p.start_motivation == 1 && p.start_growth > 0 ? :positive_outgoing_contact : :none
    reason === :none && return nothing
    # Continuity gives a positive-length interval if either endpoint has
    # positive control. At an exact motivation=1 entry, the selected control
    # is initially zero but its outgoing motivation derivative is S+I-1;
    # a positive derivative certifies positive control immediately afterward.
    witness = InvestmentWitness(
        p.mode,
        p.start_time,
        Float64(t),
        p.start_motivation,
        stop_level,
        p.start_growth,
        reason,
    )
    if p.mode === :public
        p.any_p = true
        isnothing(p.public_witness) && (p.public_witness = witness)
    else
        p.any_v = true
        isnothing(p.private_witness) && (p.private_witness = witness)
    end
    nothing
end

function _investment_presence(b)
    !b.active && return (any_p=false, any_v=false, public_witness=nothing, private_witness=nothing)
    flight = b.sol.prob.p
    if !(flight isa Flight) || isnothing(flight.presence)
        b.status === :no_distancing &&
            return (any_p=false, any_v=false, public_witness=nothing, private_witness=nothing)
        throw(
            ArgumentError(
                "Investment-presence metadata is unavailable; reproduce this Bullet with Forward.shoot",
            ),
        )
    end
    p = flight.presence
    (
        any_p=p.any_p,
        any_v=p.any_v,
        public_witness=p.public_witness,
        private_witness=p.private_witness,
    )
end
