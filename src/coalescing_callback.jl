mutable struct CoalescedVectorCondition{F}
    raw_condition::F
    held_signs::Vector{Int8}
    coalesce_abstol::Float64
    release_abstol::Float64
end

function (condition::CoalescedVectorCondition)(out, u, t, integrator)
    condition.raw_condition(out, u, t, integrator)

    @inbounds for index in eachindex(condition.held_signs)
        held_sign = condition.held_signs[index]
        iszero(held_sign) && continue

        if (t == integrator.tprev) && (held_sign * out[index] > condition.release_abstol)
            condition.held_signs[index] = 0
        else
            out[index] = held_sign * condition.release_abstol
        end
    end
    return nothing
end

struct CoalescedVectorAffect{C,FP,FN}
    condition::C
    affect!::FP
    affect_neg!::FN
    direction::Int8
end

function _approaches_coalesced_event(previous::Real, current::Real, tolerance::Real)
    abs(current) <= tolerance || return false
    return abs(current) <= abs(previous) || abs(previous) <= tolerance || previous * current <= 0
end

function _coalesced_crossing_direction(previous::Real, current::Real)
    change = current - previous
    scale = max(1.0, abs(previous), abs(current))
    change_tolerance = 16eps(Float64) * scale

    change > change_tolerance && return Int8(1)
    change < -change_tolerance && return Int8(-1)
    current > 0 && return Int8(1)
    current < 0 && return Int8(-1)
    return Int8(1)
end

_call_coalesced_affect!(::Nothing, integrator, index) = nothing

function _call_coalesced_affect!(affect!, integrator, index)
    affect!(integrator, index)
    return nothing
end

function (affect::CoalescedVectorAffect)(integrator, triggered_index)
    condition_state = affect.condition
    current = zeros(Float64, length(condition_state.held_signs))
    previous = similar(current)
    condition_state.raw_condition(current, integrator.u, integrator.t, integrator)
    condition_state.raw_condition(previous, integrator.uprev, integrator.tprev, integrator)

    for index in eachindex(current)
        direction = if index == triggered_index
            affect.direction
        elseif _approaches_coalesced_event(previous[index], current[index], condition_state.coalesce_abstol)
            _coalesced_crossing_direction(previous[index], current[index])
        else
            continue
        end

        if direction > 0
            _call_coalesced_affect!(affect.affect!, integrator, index)
        else
            _call_coalesced_affect!(affect.affect_neg!, integrator, index)
        end
        condition_state.held_signs[index] = direction
    end
    return nothing
end

function coalesced_vector_callback(
    raw_condition,
    affect!,
    affect_neg!,
    event_count::Integer;
    coalesce_abstol::Real=1e-12,
    release_abstol::Real=coalesce_abstol,
    callback_abstol::Real=coalesce_abstol,
    repeat_nudge=1 // 100,
    interp_points::Int=10,
    kwargs...,
)
    @assert event_count > 0 "event_count must be positive"
    @assert coalesce_abstol > 0 "coalesce_abstol must be positive"
    @assert release_abstol > 0 "release_abstol must be positive"
    @assert callback_abstol >= 0 "callback_abstol must be nonnegative"

    condition_state = CoalescedVectorCondition(
        raw_condition,
        zeros(Int8, event_count),
        Float64(coalesce_abstol),
        Float64(release_abstol),
    )
    initialize = function (callback, u, t, integrator)
        fill!(condition_state.held_signs, 0)
        u_modified!(integrator, false)
    end

    VectorContinuousCallback(
        condition_state,
        CoalescedVectorAffect(condition_state, affect!, affect_neg!, Int8(1)),
        CoalescedVectorAffect(condition_state, affect!, affect_neg!, Int8(-1)),
        event_count;
        initialize=initialize,
        abstol=Float64(callback_abstol),
        repeat_nudge=repeat_nudge,
        interp_points=interp_points,
        kwargs...,
    )
end

