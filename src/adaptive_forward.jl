"""
Forward shooting for the normal, free-terminal-time adaptive problem.

`shoot(r0, q0, mu0)` starts at the first distancing trigger with k=0.
`aim` checks both the Nash and adjoint endpoint conditions. `aimed_shot`
and `hunt` return the existing `Adpt.Record` type. The backward solver is
unchanged; use `Adpt.Forward.reproduce(record)` for forward reproduction.

This is a necessary-condition solver, not a global optimality certificate.
"""
module Forward

using ...Commons
using ...Nash
using ...Ctrl
using OrdinaryDiffEq:
    ODEProblem, solve, Vern9, ContinuousCallback, VectorContinuousCallback, terminate!
using SciMLBase:
    successful_retcode,
    u_modified!,
    AbstractDiffEqInterpolation,
    build_solution,
    remake,
    DiffEqArray,
    LinearInterpolation,
    ReturnCode
using Roots: find_zero, Bisection, ConvergenceFailed
using StaticArrays: SVector
using PropertyArrays: PArray, PObject, @register, @register_fn
using Distributed: pmap

const Adpt = parentmodule(@__MODULE__)
const Record = Adpt.Record
export Bullet,
    Record,
    shoot,
    aim,
    aimed_shot,
    hunt,
    hunt_active,
    hunt_doormat,
    hunt_inactive,
    reproduce,
    lshoot,
    lreproduce,
    action,
    objectives,
    infer_c,
    diagnostics,
    main,
    trace_branches,
    archive_seeds,
    investment_presence

include("adaptive_forward_presence.jl")

mutable struct Flight
    mode::Symbol
    public_only::Bool
    status::Symbol
    switches::Int
    max_switches::Int
    sliding_tol::Float64
    extend::Bool
    presence::Union{Nothing,InvestmentPresence}
end

# Preserve the seven-argument constructor used by existing callers and tests.
Flight(
    mode::Symbol,
    public_only::Bool,
    status::Symbol,
    switches::Integer,
    max_switches::Integer,
    sliding_tol::Real,
    extend::Bool,
) = Flight(
    mode,
    public_only,
    status,
    Int(switches),
    Int(max_switches),
    Float64(sliding_tol),
    extend,
    nothing,
)

struct Bullet{T} <: PObject
    r0::Float64
    q0::Float64
    mu0::Float64
    sol::T
    status::Symbol
    public_only::Bool
    switches::Int
    active::Bool
end

"""Presence flags and first positive-interval witnesses from the actual forward control modes. Witness times use the underlying trigger-start integration clock."""
investment_presence(b::Bullet) = _investment_presence(b)

function Base.show(io::IO, b::Bullet)
    print(io, "Adpt.Forward.Bullet(r0=$(b.r0), q0=$(b.q0), mu0=$(b.mu0), status=:$(b.status))")
end

_level(x) = x > 1 ? sqrt(x) - 1 : zero(x)

function _threshold(phi, k)
    if phi > 1 && k < 0
        b = sqrt(max(0.0, k * (inv(sqrt(phi)) - 1)))
        return phi * (1 + b)^2
    end
    max(phi, 1.0)
end

"""Pointwise controls, using the private-side forward continuation at k=0."""
function action(u)
    phi, F, k = u[3], u[4], u[5]
    if max(phi, F) <= 1
        return (0.0, 0.0)
    elseif k >= 0
        return F >= phi ? (_level(F), 0.0) : (0.0, _level(phi))
    elseif F > _threshold(phi, k)
        return (_level(F), 0.0)
    end
    (0.0, _level(phi))
end

function _controls(u, flight::Flight)
    flight.mode === :public && return (_level(u[4]), 0.0)
    flight.mode === :private && return (0.0, _level(u[3]))
    (0.0, 0.0)
end

function _dynamics!(du, u, flight, t)
    S, I, phi, F, k, _ = u
    p, v = _controls(u, flight)
    a = p + v
    sig = inv(1 + a)
    growth = sig * (S + I) - 1
    du[1] = -sig * S * I
    du[2] = sig * S * I - I
    du[3] = growth * phi + v * I
    du[4] = growth * F + (p + (1 + k) * v) * (S + I)
    du[5] = 0.0
    du[6] = p * S
    # At the initial phi=1 contact, use the outgoing active-side derivative.
    # There is no epsilon perturbation and no k=0 clamping callback.
    if flight.mode === :private && phi >= 1
        h = 1 + k - F / phi
        du[4] -= 0.5 * sqrt(phi) * h * (S + I)
        du[5] = -0.5 * I / sqrt(phi) * h
    end
    nothing
end

function _switch_slope(u, mode)
    f = Flight(mode, false, :flying, 0, 1000, 1e-8, false)
    du = similar(u)
    _dynamics!(du, u, f, 0.0)
    phi, k = u[3], u[5]
    if phi > 1 && k < 0
        rootphi = sqrt(phi)
        # inv(sqrt(phi))-1 without cancellation near the terminal phi=1.
        w = -(phi - 1) / (rootphi * (rootphi + 1))
        b = sqrt(k * w)
        if b > 0
            dthreshold_dphi = (1 + b)^2 - k * (1 + b) / (2b * rootphi)
            dthreshold_dk = phi * (1 + b) * w / b
            return du[4] - dthreshold_dphi * du[3] - dthreshold_dk * du[5]
        end
    end
    # At a nonsmooth contact keep the existing guard. In the smooth mixed
    # sector above, an Euler probe could cross phi=1 and reverse the apparent
    # slope even when the actual incoming/outgoing derivatives are positive.
    h = 1e-6 / max(1.0, maximum(abs, du[3:5]))
    z = u[4] - _threshold(u[3], u[5])
    trial = u .+ h .* du
    (trial[4] - _threshold(trial[3], trial[5]) - z) / h
end

function _switch!(intg, mode, direction)
    f = intg.p
    f.mode === :idle && return
    if max(intg.u[3], intg.u[4]) <= 1 + 1e-10 && intg.u[1] + intg.u[2] < 1
        _deactivate!(intg)
        return
    end
    f.public_only && return
    f.mode === mode && return
    slope = _switch_slope(intg.u, mode)
    if direction * slope < -f.sliding_tol
        f.status = :unresolved_sliding
        terminate!(intg)
        return
    end
    _presence_close!(f, intg.u, intg.t)
    f.mode = mode
    _presence_start!(f, intg.u, intg.t)
    f.switches += 1
    if f.switches > f.max_switches
        f.status = :too_many_switches
        terminate!(intg)
    end
    u_modified!(intg, true)
end

function _deactivate!(intg)
    _presence_close!(intg.p, intg.u, intg.t)
    intg.p.mode = :idle
    _presence_start!(intg.p, intg.u, intg.t)
    intg.p.status = intg.p.extend ? :coasting : :terminated
    intg.p.extend || terminate!(intg)
    u_modified!(intg, true)
end

function _conditions!(out, u, t, intg)
    out[1] = intg.p.public_only || intg.p.mode === :idle ? -1.0 : u[4] - _threshold(u[3], u[5])
    out[2] = max(u[3], u[4]) - 1
    p, v = _controls(u, intg.p)
    out[3] = u[1] / (1 + p + v) - 1
    out[4] = u[2] - infection_floor
end

function _positive!(intg, i)
    i == 1 && _switch!(intg, :public, 1)
end

function _negative!(intg, i)
    if i == 1
        _switch!(intg, :private, -1)
    elseif i == 2 && intg.u[1] + intg.u[2] < 1 + 1e-9
        _deactivate!(intg)
    elseif i == 4 && intg.p.extend && intg.p.mode === :idle
        intg.p.status = :terminated
        terminate!(intg)
    end
end

function _trigger(r0, q0, mu0)
    S0, I0 = r0 * initial_susceptible, r0 * initial_infected
    m = -S0 * exp(-S0 - I0)
    motivation = max(q0, mu0)
    if motivation * I0 >= 1
        return [S0, I0, q0 * I0, mu0 * I0, 0.0, 0.0], 0.0, true
    elseif r0 <= 1 || motivation <= min_adj0(r0)
        return [S0, I0, q0 * I0, mu0 * I0, 0.0, 0.0], 0.0, false
    end
    S, I = given_slope(inv(S0 * motivation), m, -1)
    factor = S0 * I / S
    u = [S, I, q0 * factor, mu0 * factor, 0.0, 0.0]
    # Make the analytically known trigger exact, not an arbitrary k perturbation.
    mu0 >= q0 ? (u[4] = 1.0) : (u[3] = 1.0)
    u, preceding_t(r0, S), true
end

"""
    shoot(r0, q0, mu0; save_everystep=false, tol=1e-12, tmax=1000)

Forward IVP from the first distancing trigger. q0 and mu0 are physical
initial values q(0) and F(0)/I(0); k(0)=0 is exact. Internal S,I,phi,F
use the same R0 scaling as Adpt. Stops at permanent deactivation.
"""
function shoot(
    r0::Real,
    q0::Real,
    mu0::Real;
    save_everystep::Bool=false,
    tol::Real=1e-12,
    tmax::Real=1000.0,
    maxiters::Int=500000,
    max_switches::Int=1000,
    sliding_tol::Real=1e-8,
    extend::Bool=false,
    alg=Vern9(lazy=false),
    kwargs...,
)
    r0 > 0 || throw(ArgumentError("r0 must be positive"))
    q0 > 0 && isfinite(q0) || throw(ArgumentError("q0 must be finite and positive"))
    isfinite(mu0) || throw(ArgumentError("mu0 must be finite"))
    u, t0, active = _trigger(Float64(r0), Float64(q0), Float64(mu0))
    public_only = mu0 >= q0
    f = Flight(
        active ? (public_only ? :public : :private) : :idle,
        public_only,
        active ? :flying : (extend ? :coasting : :no_distancing),
        0,
        max_switches,
        sliding_tol,
        extend,
    )
    active && _presence_start!(f, u, t0)
    cb = VectorContinuousCallback(
        _conditions!,
        _positive!,
        _negative!,
        4;
        abstol=1e-12,
        reltol=0.0,
        save_positions=(true, true),
    )
    span = active || extend ? (t0, t0 + tmax) : (0.0, 0.0)
    prob = ODEProblem(_dynamics!, u, span, f)
    sol = solve(
        prob,
        alg;
        abstol=tol,
        reltol=tol,
        callback=cb,
        save_everystep=save_everystep,
        maxiters=maxiters,
        kwargs...,
    )
    _presence_close!(f, sol.u[end], sol.t[end])
    status = f.status === :flying ? :horizon_reached : f.status
    Bullet(Float64(r0), Float64(q0), Float64(mu0), sol, status, public_only, f.switches, active)
end

function _final_size(S, I)
    x = -S * exp(-S - I)
    floor = Commons.IE
    if x < floor
        x >= floor - 32eps(Float64) || throw(DomainError(x, "final-size Lambert W domain"))
        x = floor
    end
    Float64(-lambertw(x, 0))
end

function diagnostics(b::Bullet)
    u = b.sol.u[end]
    S, I, phi, F, k, P = u
    si = _final_size(S, I)
    Q = phi * S / I
    V = Q - b.q0 * b.r0 * initial_susceptible
    (
        status=b.status,
        retcode=b.sol.retcode,
        switches=b.switches,
        Sinf=si,
        qinf=Q / si,
        muinf=F * S / (I * si),
        kinf=k,
        public_spending=P / b.r0,
        private_spending=V / b.r0,
        inactive=max(phi, F) <= 1 + 1e-7,
        k_nonpositive=maximum(b.sol[5, :]) <= 1e-9,
    )
end

infer_c(b::Bullet) = diagnostics(b).qinf

"""Two dimensionless endpoint residuals: Nash first, leader transversality second."""
function aim(b::Bullet, c::Real)
    c > 0 || throw(ArgumentError("c must be positive"))
    b.status in (:terminated, :no_distancing) ||
        throw(ErrorException("Forward flight ended with $(b.status)"))
    successful_retcode(b.sol.retcode) ||
        throw(ErrorException("Unsuccessful forward solve: $(b.sol.retcode)"))
    S, I, phi, F, k, _ = b.sol.u[end]
    si = _final_size(S, I)
    qinf = phi * S / (I * si)
    SVector(qinf / c - 1, F * S / I * (1 - si) / (c * si) - (1 + k))
end

aim(r0::Real, c::Real, q0::Real, mu0::Real; kwargs...) = aim(shoot(r0, q0, mu0; kwargs...), c)


@register function μ0(b::Bullet)
    getfield(b, :mu0)
end
@register function t(b::Bullet)
    b.sol.t
end
@register function X(b::Bullet)
    Array(b.sol)
end

function action(b::Bullet)
    stack(action.(b.sol.u))
end
@register_fn action Bullet
@register function p(b::Bullet)
    action(b)[1, :]
end
@register function v(b::Bullet)
    action(b)[2, :]
end
@register function j1p(b::Bullet)
    objectives(b)[1]
end
@register function j1v(b::Bullet)
    objectives(b)[2]
end
@register function j1(b::Bullet)
    sum(objectives(b)[1:2])
end
@register function obj(b::Bullet)
    sum(objectives(b))
end

function objectives(b::Bullet, c::Real=infer_c(b))
    d = diagnostics(b)
    (max(0.0, d.public_spending) / c, max(0.0, d.private_spending) / c, 1 - d.Sinf / b.r0)
end

function Adpt.Record(b::Bullet, c::Real, branch::Int=2)
    b.status in (:terminated, :no_distancing) ||
        throw(ArgumentError("Cannot materialize an unfinished forward flight: $(b.status)"))
    successful_retcode(b.sol.retcode) ||
        throw(ArgumentError("Cannot materialize a failed ODE solve"))
    d = diagnostics(b)
    X = Array(b.sol)
    t = Float64.(b.sol.t)
    Ip, pidx = findmax(X[2, :])
    # If the epidemic peak precedes investment onset, retain the null peak convention.
    if !b.active || X[1, 1] < 1
        Ip, pidx = null_Ip(b.r0), 0
    end
    t, X, pidx = Adpt.fewer(t, X, pidx)
    jp, jv, j2 = objectives(b, c)
    presence = investment_presence(b)
    Record((;
        branch,
        any_p=presence.any_p,
        any_v=presence.any_v,
        target_r0=b.r0,
        r0err=0.0,
        t=Real[t...],
        X=Matrix{Real}(X),
        Ip,
        pidx,
        j1p=jp,
        j1v=jv,
        j1=jp + jv,
        j2,
        obj=jp + jv + j2,
        r0=b.r0,
        c=Float64(c),
        q0=b.q0 * initial_susceptible,
        μ0=b.mu0 * initial_susceptible,
        k0=0.0,
        Sinf=d.Sinf,
        qinf=d.qinf,
        μinf=d.muinf,
        kinf=d.kinf,
    ))
end

function reproduce(rec::Record; kwargs...)
    b = shoot(rec.r0, rec.q0 / initial_susceptible, rec.μ0 / initial_susceptible; kwargs...)
    Record(b, rec.c, rec.branch)
end

# Keep the trigger-start solve intact: errors accumulated while reproducing the
# long idle prefix must not perturb the sensitive shooting trajectory.
struct FullTrajectoryInterpolation{P,S} <: AbstractDiffEqInterpolation
    prefix::P
    suffix::S
    trigger::Float64
    shift::Float64
end

function (interp::FullTrajectoryInterpolation)(t::Number, idxs, deriv, p, continuity=:left)
    if t < interp.trigger || (t == interp.trigger && deriv != Val{0} && continuity === :left)
        interp.prefix(t, deriv; idxs, continuity)
    else
        time =
            t == interp.trigger ? first(interp.suffix.t) :
            t == last(interp.suffix.t) + interp.shift ? last(interp.suffix.t) : t - interp.shift
        interp.suffix(time, deriv; idxs, continuity)
    end
end

function (interp::FullTrajectoryInterpolation)(ts::AbstractVector, idxs, deriv, p, continuity=:left)
    DiffEqArray([interp(t, idxs, deriv, p, continuity) for t in ts], ts)
end

function (interp::FullTrajectoryInterpolation)(out, t, idxs, deriv, p, continuity=:left)
    if t isa Number
        out .= interp(t, idxs, deriv, p, continuity)
    else
        for (dest, time) in zip(out, t)
            interp(dest, time, idxs, deriv, p, continuity)
        end
        out
    end
end

"""
    lshoot(r0, q0, mu0; save_everystep=true, kwargs...)
    lshoot(rec::Record; kwargs...)

Reproduce all six states from physical time zero, where I/r0 is
`initial_infected`, through the zero-investment prefix, distancing, and final
zero-investment continuation. `shoot` still starts at the first trigger.
The prefix is integrated separately to the trigger state. Its event time sets
the physical origin of the otherwise unchanged trigger-onward solution; this
avoids the near-initial-infection cancellation in `preceding_t`. As before,
`tmax` is the maximum integration duration after the trigger. `saveat` values
use physical time; a scalar `saveat` gives a grid starting at zero. Internal
dense segments support sampling even with `save_everystep=false`; `dense=false`
returns ordinary linear interpolation between the requested saved samples.
"""
function lshoot(
    r0::Real,
    q0::Real,
    mu0::Real;
    save_everystep::Bool=true,
    tol::Real=1e-12,
    tmax::Real=1000.0,
    maxiters::Int=500000,
    max_switches::Int=1000,
    sliding_tol::Real=1e-8,
    alg=Vern9(lazy=false),
    saveat=Float64[],
    dense::Bool=save_everystep && !(saveat isa Number) && isempty(saveat),
    save_start::Bool=true,
    save_end::Bool=true,
    kwargs...,
)
    # Validate arguments using the same path as ordinary shooting.
    r0 > 0 || throw(ArgumentError("r0 must be positive"))
    q0 > 0 && isfinite(q0) || throw(ArgumentError("q0 must be finite and positive"))
    isfinite(mu0) || throw(ArgumentError("mu0 must be finite"))
    trigger_u, trigger_old, _ = _trigger(Float64(r0), Float64(q0), Float64(mu0))
    prefix = if trigger_old > 0
        S0, I0 = Float64(r0 * initial_susceptible), Float64(r0 * initial_infected)
        u0 = [S0, I0, Float64(q0) * I0, Float64(mu0) * I0, 0.0, 0.0]
        idle = Flight(:idle, mu0 >= q0, :coasting, 0, 0, 0.0, true)
        prob = ODEProblem(_dynamics!, u0, (0.0, trigger_old + max(1.0, 0.01trigger_old)), idle)
        condition(u, t, integrator) = u[1] - trigger_u[1]
        cb = ContinuousCallback(
            condition,
            nothing,
            terminate!;
            abstol=1e-14,
            reltol=0.0,
            save_positions=(true, true),
        )
        # The epidemic starts at O(epsilon); the ordinary absolute tolerance
        # would give an appreciable error during this exponential-growth phase.
        pre = solve(
            prob,
            alg;
            abstol=min(tol, 1e-14) * initial_infected,
            reltol=min(tol, 1e-14),
            save_everystep=true,
            maxiters,
            dense=true,
            save_start=true,
            save_end=true,
            callback=cb,
            kwargs...,
        )
        successful_retcode(pre.retcode) && pre.retcode == ReturnCode.Terminated ||
            error("Failed to reach the zero-investment prefix trigger")
        pre
    else
        nothing
    end
    trigger = isnothing(prefix) ? 0.0 : last(prefix.t)
    shift = trigger - trigger_old
    samples = if saveat isa Number
        saveat > 0 || throw(ArgumentError("saveat must be positive"))
        collect(0.0:saveat:(trigger+tmax))
    else
        saveat
    end
    b = shoot(
        r0,
        q0,
        mu0;
        extend=true,
        save_everystep=true,
        tol,
        tmax,
        maxiters,
        max_switches,
        sliding_tol,
        alg,
        dense=true,
        save_start=true,
        save_end=true,
        kwargs...,
    )
    interpolation =
        isnothing(prefix) ? b.sol.interp :
        FullTrajectoryInterpolation(prefix, b.sol, trigger, shift)
    tail_times = b.sol.t .+ shift
    times = if save_everystep
        isnothing(prefix) ? copy(tail_times) : vcat(filter(t -> t < trigger, prefix.t), tail_times)
    else
        # Continuous callbacks save both sides of each contact independently
        # of save_everystep. Keep those samples, including switch discontinuities.
        events = [
            tail_times[i] for
            i in eachindex(tail_times) if (i > 1 && tail_times[i] == tail_times[i-1]) ||
            (i < length(tail_times) && tail_times[i] == tail_times[i+1])
        ]
        sort!(unique(events))
    end
    stop = last(tail_times)
    trigger > 0 && push!(times, trigger)
    append!(times, filter(t -> 0 <= t <= stop, samples))
    save_start && push!(times, 0.0)
    save_end && push!(times, stop)
    !save_start && filter!(!=(0.0), times)
    !save_end && filter!(!=(stop), times)
    sort!(unique!(times))
    # Preserve saved suffix states exactly rather than round-tripping their
    # timestamps through floating-point addition and subtraction.
    saved = Dict(zip(tail_times, b.sol.u))
    states = [
        get(saved, t) do
            interpolation(t, nothing, Val{0}, b.sol.prob.p, :left)
        end for t in times
    ]
    initial = isnothing(prefix) ? b.sol.prob.u0 : prefix.prob.u0
    fullprob = remake(b.sol.prob; u0=initial, tspan=(0.0, last(b.sol.prob.tspan) + shift))
    interp = dense ? interpolation : LinearInterpolation(times, states)
    sol = build_solution(
        fullprob,
        b.sol.alg,
        times,
        states;
        interp,
        dense,
        retcode=b.sol.retcode,
        stats=b.sol.stats,
        calculate_error=false,
    )
    Bullet(b.r0, b.q0, b.mu0, sol, b.status, b.public_only, b.switches, b.active)
end

lshoot(rec::Record; kwargs...) =
    lshoot(rec.r0, rec.q0 / initial_susceptible, rec.μ0 / initial_susceptible; kwargs...)
lreproduce(rec::Record; kwargs...) = Record(lshoot(rec; kwargs...), rec.c, rec.branch)

shoot(rec::Record; kwargs...) =
    shoot(rec.r0, rec.q0 / initial_susceptible, rec.μ0 / initial_susceptible; kwargs...)
aim(rec::Record, c::Real=rec.c; kwargs...) = aim(shoot(rec; kwargs...), c)

function _try_residual(r0, c, z, options)
    all(isfinite, z) && abs(z[1]) < 100 && abs(z[2]) < 1e6 || return nothing
    try
        q = exp(z[1])
        b = shoot(r0, q, q * z[2]; options...)
        r = aim(b, c)
        all(isfinite, r) || return nothing
        return r
    catch err
        (err isa DomainError || err isa ErrorException) || rethrow()
        return nothing
    end
end

function _polish(r0, c, seed; aim_tol, iterations, options, bounds=nothing, residual_probe=nothing)
    function probe(z)
        !isnothing(bounds) && !_inside(z, bounds) && return nothing
        isnothing(residual_probe) ? _try_residual(r0, c, z, options) : residual_probe(z)
    end
    z = [log(seed[1]), seed[2] / seed[1]]
    r = probe(z)
    isnothing(r) && return nothing
    for _ = 1:iterations
        maximum(abs, r) <= aim_tol && return (exp(z[1]), exp(z[1]) * z[2])
        hs = (1e-4, 1e-4 * max(1.0, abs(z[2])))
        columns = Vector{Any}(undef, 2)
        for j = 1:2
            zz = copy(z)
            zz[j] += hs[j]
            rr = probe(zz)
            if isnothing(rr)
                zz[j] = z[j] - hs[j]
                rr = probe(zz)
                isnothing(rr) && return nothing
                columns[j] = (r - rr) / hs[j]
            else
                columns[j] = (rr - r) / hs[j]
            end
        end
        a, cc = columns[1]
        bb, d = columns[2]
        det = a * d - bb * cc
        abs(det) > 1e-14 || return nothing
        step = [(d * r[1] - bb * r[2]) / det, (-cc * r[1] + a * r[2]) / det]
        scale = max(1.0, abs(step[1]), abs(step[2]) / 2)
        step ./= scale
        accepted = false
        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.015625)
            zz = z .- alpha .* step
            rr = probe(zz)
            if !isnothing(rr) && sum(abs2, rr) < sum(abs2, r)
                z, r = zz, rr
                accepted = true
                break
            end
        end
        accepted || return nothing
    end
    maximum(abs, r) <= aim_tol ? (exp(z[1]), exp(z[1]) * z[2]) : nothing
end

function _references(r0, c)
    lf = c <= Nash.min_c(r0) ? Nash.Record(r0, c) : Nash.aimed_shot(r0, c)
    co = c <= Ctrl.min_c(r0) ? Ctrl.Record(r0, c) : Ctrl.aimed_shot(r0, c)
    lf, co
end

# A forward costate can be extremely sensitive near the public/private
# separatrix. Bracket the leader condition at fixed q0 before aiming at c.
# A public overshoot that has not deactivated is used ONLY for a bracket sign;
# it is never accepted as a candidate or materialized as a Record.
function _leader_sign(b, c)
    b.status in (:terminated, :no_distancing, :horizon_reached) || return NaN
    successful_retcode(b.sol.retcode) || return NaN
    u = b.sol.u[end]
    S, I, _, F, k, _ = u
    I > 0 || return F > 0 ? 1e100 : NaN
    si = _final_size(S, I)
    value = F * S / I * (1 - si) / (c * si) - (1 + k)
    isnan(value) ? NaN : clamp(value, -1e100, 1e100)
end

function _inner_shot(r0, c, q, options; aim_tol)
    function residual(beta)
        _leader_sign(shoot(r0, q, q * beta; options...), c)
    end
    lo, hi = -1.0, 1.0
    flo, fhi = residual(lo), residual(hi)
    for _ = 1:8
        isfinite(flo) && flo <= 0 && break
        lo *= 2
        flo = residual(lo)
    end
    isfinite(flo) && isfinite(fhi) && flo * fhi <= 0 || return nothing
    beta = try
        find_zero(residual, (lo, hi), Bisection(); xatol=0.0, xrtol=0.0, atol=0.0, rtol=0.0)
    catch err
        (err isa DomainError || err isa ErrorException || err isa ConvergenceFailed) || rethrow()
        return nothing
    end
    best = nothing
    error = Inf
    for be in (prevfloat(beta), beta, nextfloat(beta))
        b = shoot(r0, q, q * be; options...)
        b.status in (:terminated, :no_distancing) || continue
        rr = aim(b, c)
        if abs(rr[2]) < error
            error = abs(rr[2])
            best = (q, q * be, rr)
        end
    end
    # A loose sign-probe may bracket the outer residual; final candidates
    # still have to satisfy both aim_tol checks at the requested ODE tolerance.
    error <= max(100aim_tol, 1e-4) ? best : nothing
end

# Mixed roots can lie extremely close to the LF value: a wide uniform mesh
# can see an invalid endpoint followed only by positive Nash residuals.
# Discover that boundary layer geometrically rather than assigning the LF
# trajectory to a normal candidate or inserting case-specific seed values.
function _boundary_q_mesh(reference, lower, upper)
    values = Float64[]
    offset = 1.0
    while offset >= 64eps(Float64)
        q = Float64(reference * (1 + offset))
        lower <= q <= upper && push!(values, q)
        offset /= 3
    end
    sort(unique(values))
end

function _nested_roots(r0, c, qlf, qco; options, aim_tol, scan_points, return_report=false)
    cache = Dict{Float64,Any}()
    function probe(q)
        get!(cache, Float64(q)) do
            try
                _inner_shot(r0, c, q, options; aim_tol)
            catch err
                (err isa DomainError || err isa ErrorException) || rethrow()
                nothing
            end
        end
    end
    lower, upper = minmax(qlf, qco)
    boundary_mesh = _boundary_q_mesh(qlf, 0.5lower, 1.5upper)
    mesh = sort(
        unique(
            vcat(
                [0.5lower, qlf, qco, 1.5upper],
                collect(range(lower, upper; length=max(5, scan_points))),
                boundary_mesh,
            ),
        ),
    )
    values = [probe(q) for q in mesh]
    roots = Tuple{Float64,Float64}[]
    brackets = Tuple{Float64,Float64}[]
    for value in values
        !isnothing(value) && maximum(abs, value[3]) <= aim_tol && push!(roots, value[1:2])
    end
    for i = 1:length(mesh)-1
        left, right = values[i], values[i+1]
        if isnothing(left) || isnothing(right)
            continue
        end
        left[3][1] * right[3][1] < 0 || continue
        push!(brackets, (mesh[i], mesh[i+1]))
        function residual(q)
            value = probe(q)
            isnothing(value) && throw(DomainError(q, "Inner forward aim failed"))
            value[3][1]
        end
        try
            q = find_zero(
                residual,
                (mesh[i], mesh[i+1]),
                Bisection();
                xatol=0.0,
                xrtol=0.0,
                atol=0.0,
                rtol=0.0,
            )
            for qq in (prevfloat(q), q, nextfloat(q))
                value = probe(qq)
                !isnothing(value) && maximum(abs, value[3]) <= aim_tol && push!(roots, value[1:2])
            end
        catch err
            (err isa DomainError || err isa ErrorException || err isa ConvergenceFailed) ||
                rethrow()
        end
    end
    roots = unique(roots)
    return_report ?
    (;
        roots,
        q_mesh=mesh,
        boundary_q_mesh=boundary_mesh,
        valid_q_slices=count(!isnothing, values),
        root_brackets=brackets,
        probed_q_slices=length(cache),
    ) : roots
end

include("adaptive_forward_continuation.jl")

# Keep the first candidate on an exact objective tie, as the original greedy rule did.
_minimum_objective_candidate(candidates) = candidates[argmin([x.obj for x in candidates])]

"""
    aimed_shot(r0, c, guess=nothing; aim_tol=1e-7, ...)

Solve the two terminal residuals for (q0,mu0). An explicit guess is a pair
of physical initial values. Default multistart seeds use forward Nash/Ctrl
solutions only. Ordered old branch projections may be supplied through
`archive`; they are discovery seeds, never accepted backward solutions.
Hybrid search also traces the leader-residual manifold even when Newton
already found a candidate. Set `return_report=true` to inspect all retained
candidates, traces and stopping reasons. No backward integration is used.
Unresolved sliding, failed endpoint matches, and timeouts are not successes.
"""
function aimed_shot(
    r0::Real,
    c::Real,
    guess=nothing;
    aim_tol::Real=1e-7,
    tol::Real=1e-12,
    final_tol::Real=1e-13,
    iterations::Int=30,
    scan_points::Int=7,
    prefer=:greedy,
    branch::Symbol=:all,
    strategy::Symbol=:hybrid,
    archive=nothing,
    q_bounds=nothing,
    beta_bounds=nothing,
    continuation_step::Real=0.04,
    continuation_min_step::Real=1e-5,
    continuation_max_step::Real=0.12,
    continuation_steps::Int=120,
    return_report::Bool=false,
    save_everystep::Bool=true,
    kwargs...,
)
    c > 0 || throw(ArgumentError("c must be positive"))
    prefer === :greedy ||
        throw(ArgumentError("Forward supports only prefer=:greedy; received $(repr(prefer))"))
    branch in (:all, :active, :public, :inactive) ||
        throw(ArgumentError("Unsupported branch=$branch"))
    strategy in (:hybrid, :newton, :nested, :continuation) ||
        throw(ArgumentError("Unsupported strategy=$strategy"))
    lf, co = _references(r0, c)
    opts = (; tol, save_everystep=false, kwargs...)
    seeds = Tuple{Float64,Float64}[]
    if guess isa Record
        push!(
            seeds,
            (Float64(guess.q0 / initial_susceptible), Float64(guess.μ0 / initial_susceptible)),
        )
    elseif !isnothing(guess)
        length(guess) == 2 || throw(ArgumentError("guess must be (q0,mu0)"))
        push!(seeds, (Float64(guess[1]), Float64(guess[2])))
    end
    qco = c * (1 - co.j2) / initial_susceptible
    # The CO path is a valid public-only AO candidate only when it dominates
    # the private target from the start. The forward solver checks its endpoints.
    if co.λ0 >= qco
        push!(seeds, (qco, Float64(co.λ0)))
    end
    qlf = Float64(lf.q0)
    beta_co = Float64(co.λ0) / max(qco, eps())
    betas = unique(
        vcat(
            [-1.0, 0.0],
            collect(range(0.1, 0.95; length=max(2, scan_points))),
            [min(beta_co, 0.999), 1.0],
        ),
    )
    for q in unique([qlf, qco, sqrt(qlf * qco)])
        for beta in betas
            push!(seeds, (q, beta * q))
        end
    end
    candidates = Record[]
    function retain(root)
        try
            b = shoot(r0, root...; tol=final_tol, save_everystep, kwargs...)
            maximum(abs, aim(b, c)) <= aim_tol || return
            d = diagnostics(b)
            d.inactive && d.k_nonpositive || return
            jp, jv, _ = objectives(b, c)
            kind =
                jp <= 1e-11 && jv <= 1e-11 ? :trivial :
                jp <= 1e-11 ? :inactive : jv <= 1e-11 ? :public : :active
            branch === :all || kind === :trivial || kind === branch || return
            rec = Record(
                b,
                c,
                kind === :trivial ? 0 : kind === :inactive ? 1 : kind === :public ? 3 : 2,
            )
            any(x -> abs(x.q0 - rec.q0) < 1e-7 && abs(x.μ0 - rec.μ0) < 1e-7, candidates) ||
                push!(candidates, rec)
        catch err
            (err isa DomainError || err isa ErrorException) || rethrow()
        end
    end
    if strategy in (:hybrid, :newton)
        for seed in unique(seeds)
            root = _polish(r0, c, seed; aim_tol=aim_tol / 4, iterations, options=opts)
            isnothing(root) || retain(root)
        end
    end
    continuation = nothing
    if strategy in (:hybrid, :continuation)
        # A candidate found by Newton does not suppress branch exploration.
        # Use its location as one discovery seed, then enumerate slice roots
        # and continue each connected leader-residual curve in both directions.
        traced_seeds = copy(seeds)
        append!(
            traced_seeds,
            [(rec.q0 / initial_susceptible, rec.μ0 / initial_susceptible) for rec in candidates],
        )
        continuation = trace_branches(
            r0,
            c;
            seeds=traced_seeds,
            archive,
            q_bounds,
            beta_bounds,
            scan_points=max(3, scan_points),
            continuation_step,
            continuation_min_step,
            continuation_max_step,
            continuation_steps,
            leader_tol=max(aim_tol / 4, 1e-9),
            aim_tol,
            iterations,
            tol=final_tol,
            kwargs...,
        )
        for root in continuation.roots
            retain(root)
        end
    end
    nested = nothing
    if strategy === :nested || (strategy === :hybrid && isempty(candidates))
        nested_options = (; opts..., tol=final_tol)
        nested = _nested_roots(
            r0,
            c,
            qlf,
            qco;
            options=nested_options,
            aim_tol,
            scan_points=max(9, scan_points),
            return_report=true,
        )
        for root in nested.roots
            retain(root)
        end
    end
    isempty(candidates) && error(
        "No forward candidate satisfies both endpoint conditions for r0=$r0,c=$c. Supply (q0,mu0) guesses or increase scan_points; no backward fallback was used.",
    )
    best = _minimum_objective_candidate(candidates)
    best.obj <= lf.obj + 10aim_tol ||
        error("Forward candidates are worse than the feasible LF benchmark; search is incomplete")
    return_report ? (; record=best, candidates, continuation, nested) : best
end

aimed_shot(r0::Real, c::Real, q0::Real, mu0::Real; kwargs...) =
    aimed_shot(r0, c, (q0, mu0); kwargs...)
hunt(r0::Real, c::Real, guess=nothing; kwargs...) = aimed_shot(r0, c, guess; kwargs...)
hunt_active(r0::Real, c::Real; kwargs...) = aimed_shot(r0, c; branch=:active, kwargs...)
hunt_doormat(r0::Real, c::Real; kwargs...) = aimed_shot(r0, c; branch=:public, kwargs...)
hunt_inactive(r0::Real, c::Real; kwargs...) = aimed_shot(r0, c; branch=:inactive, kwargs...)

function main(r0s, cs; parallel::Bool=false, kwargs...)
    jobs = collect(Iterators.product(r0s, cs))
    f = pair -> hunt(pair...; kwargs...)
    values = parallel ? pmap(f, jobs) : map(f, jobs)
    PArray(values)
end

function main(r0::Real, cs; parallel::Bool=false, kwargs...)
    f = c -> hunt(r0, c; kwargs...)
    PArray(parallel ? pmap(f, collect(cs)) : map(f, collect(cs)))
end

function main(r0s, c::Real; parallel::Bool=false, kwargs...)
    f = r0 -> hunt(r0, c; kwargs...)
    PArray(parallel ? pmap(f, collect(r0s)) : map(f, collect(r0s)))
end
main(r0::Real, c::Real; kwargs...) = hunt(r0, c; kwargs...)
main(; kwargs...) = main(Adpt.R0S, Adpt.CS; kwargs...)


end # module Forward
