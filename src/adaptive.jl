using Distributed: pmap
using Optim: optimize
using Roots: find_zero, find_zeros, ConvergenceFailed
using SciMLBase: DEIntegrator, LeftRootFind, RightRootFind, u_modified!
using OrdinaryDiffEq: ODEFunction, ODEProblem, ODESolution, Vern9,
                      VectorContinuousCallback, ContinuousCallback, DiscreteCallback, CallbackSet, terminate!,
                      solve, init, reinit!, step!, solve!, remake
using DiffEqCallbacks: IntegrandValuesSum, IntegratingSumCallback
using Printf
using PyFormattedStrings: @f_str
const zero_tols = pairs((; atol=0.0, rtol=0.0))  # for find_zeros
const zero_tol = pairs((; atol=0.0, rtol=0.0, xatol=0.0, xrtol=0.0))  # for find_zero

# ============================================================================
#   Shooting Interface
# ============================================================================

function zeroing(prob::ODEProblem, alg; tol=1e-13, size=6, kwargs...)
    abstol = fill(tol, size)
    reltol = fill(tol, size)
    intg = init(prob, alg; abstol=abstol, reltol=reltol, kwargs...)
    if prob.tspan[1] == prob.tspan[2]
        return intg
    end
    count = 0
    while (count < 8)
        step!(intg)
        if max(intg.u[3], intg.u[4]) >= 1.0
            return intg
        else
            intg.opts.abstol[3:5] /= 10
            intg.opts.reltol[3:5] /= 10
            reinit!(intg)
            count += 1
        end
    end
    throw(DomainError("Could not find proper zeroing tolerance"))
    #println("Could not find proper zeroing tolerance")
    #return init(remake(prob; tspan=(prob.tspan[1], prob.tspan[1])), alg; abstol=abstol, reltol=reltol, kwargs...)
end

"""
Shoot (solve the IVP) at will.
Adjust (Sinf, qinf=c, μinf) to aim anywhere,
and obtain a *Bullet* — the result of your shot.

Note: An `ErrorException` may rarely occur when root-finding the precise timing of (a) callback event(s).
This appears to happen when the event time cannot be localized beyond the solver’s precision limit,
resulting in the message:
    "ERROR: Callback handling failed. Please file an issue with code to reproduce."
This issue can be solved either by slowing down the dynamics (rescaling dX ./= 10 for example)
or just adjusting tolerance value for the ODE solver.
Note that the default precision is conservative, and this adjustment does not affect
the qualitative results.
"""
function shoot(Sinf::Real, qinf::Real, μinf::Real;
               callback=adaptive_callback(), save_everystep=false,
               tol=1e-13, _retry_depth=0, alg=Vern9(lazy=false), kwargs...)
    try
        intg = initiate(Sinf, qinf, μinf;
            callback=callback, save_everystep=save_everystep,
            abstol=tol, reltol=tol, alg=alg,
            initializer=:zeroing, kwargs...,
        )
        sol = solve!(intg)
        if intg.p[7]
            println("Traversing F=Fthr found:")
            println("  Sinf=$(Sinf); qinf=$(qinf); μinf=$(μinf);")
        end
        Bullet(Sinf, qinf, μinf, intg.p[5], intg.p[6], sol)
    catch e
        if !isa(e, ErrorException) || _retry_depth>7
            println("Sinf=$(Sinf), qinf=$(qinf), μinf=$(μinf)")
            rethrow(e)
        else
            #==#
            println("Lowering Tolerance of: Sinf=$(Sinf), qinf=$(qinf), μinf=$(μinf)")
            tol /= 10
            _retry_depth += 1
            shoot(Sinf, qinf, μinf; callback=callback, save_everystep=save_everystep, tol=tol, _retry_depth=_retry_depth, kwargs...)
        end
    end
end

"""
Refine the shot — search for the IVP that attains the target R0.
May not converge if μinf = c is a poor initial guess,
and it always returns the first solution encountered.
"""
function aimed_shot(r0::Real, c::Real, Sinf::Real, μinfguess=nothing; kwargs...)
    μinfguess = (isnothing(μinfguess) ? c : μinfguess)
    μinf = find_zero(x->aim(shoot(Sinf, c, x; kwargs...), r0), μinfguess; zero_tol...)
    shoot(Sinf, c, μinf; kwargs...)
end

"""
snipes() will automatically decide proper initial guess.
If multiple trajectories meets target R0, it returns the one with the lowest societal cost.
"""
function snipes(r0::Real, c::Real, Sinf::Real; kwargs...)
    Sinfmax = find_Sinf_A(r0, c)
    μinfmax = c/(1-Sinfmax)
    μinfguess = [0.0, μinfmax]
    μinfsols = find_zeros(μinfguess; zero_tols...) do μinf
        if isnan(μinf); return nothing; end  # Don't know why, but find_zeros often do this...
        μinf = clamp(μinf, 0, μinfmax)
        blt = shoot(Sinf, c, μinf; kwargs...)
        return aim(blt, r0)
    end
    blts = shoot.(Sinf, c, μinfsols; kwargs...)
    _, idx = findmin(score, blts)
    blts[idx]
end

function snipes!(answer_sheet::Dict{Real, Real}, r0::Real, c::Real, Sinf::Real, μinfmin::Real, μinfmax::Real; _retry_depth=0, tol=1e-13, kwargs...)
    @assert _retry_depth<4 "Could not find proper sniping tolerance\nr0=$(r0);c=$(c);Sinf=$(Sinf);μinfmin=$(μinfmin);μinfmax=$(μinfmax);"
    μinfsols = find_zeros([μinfmin, μinfmax]; atol=1e-13*r0/(r0-1), rtol=0, xatol=1e-15, xrtol=1e-13) do μinf
        if isnan(μinf); return nothing; end  # Don't know why, but find_zeros often do this...
        μinf = clamp(μinf, μinfmin, μinfmax)
        blt = shoot(Sinf, c, μinf; tol=tol, kwargs...)
        return aim(blt, r0)
    end
    blts = shoot.(Sinf, c, μinfsols; tol=tol, kwargs...)
    qualified = filter(x->(abs(aim(x, r0))<1e-4), blts)
    if isempty(qualified)
        _retry_depth += 1
        tol /= 100
        return snipes!(answer_sheet, r0, c, Sinf, μinfmin, μinfmax; _retry_depth=_retry_depth, tol=tol, kwargs...)
    end
    scr, idx = findmin(score, qualified)
    answer_sheet[qualified[idx].Sinf] = qualified[idx].μinf
    return scr
end

function snipe!(answer_sheet::Dict{Real, Real}, r0::Real, c::Real, Sinf::Real, μinfmin::Real, μinfmax::Real; _retry_depth=0, tol=1e-13, kwargs...)
    @assert _retry_depth<4 "Could not find proper sniping tolerance\nr0=$(r0);c=$(c);Sinf=$(Sinf);μinfmin=$(μinfmin);μinfmax=$(μinfmax);"
    μinf = find_zero([μinfmin, μinfmax]; atol=1e-13*r0/(r0-1), rtol=0, xatol=0, xrtol=0) do μinf
        blt = shoot(Sinf, c, μinf; tol=tol, kwargs...)
        return aim(blt, r0)
    end
    blt = shoot(Sinf, c, μinf; tol=tol, kwargs...)
    if abs(aim(blt, r0)) > 1e-4
        _retry_depth += 1
        tol /= 100
        return snipe!(answer_sheet, r0, c, Sinf, μinfmin, μinfmax; _retry_depth=_retry_depth, tol=tol, kwargs...)
    else
        answer_sheet[Sinf] = μinf
        return score(blt)
    end
end

"""
Snipe repeatedly to find the best shot for (R0, c).
The best result is saved as a *Record*.
Note: the implementation does not call snipes() directly;
 it differs slightly to handle specific BVP details.
"""
function hunt_active(r0::Real, c::Real; kwargs...)
    Sinfmin = find_Sinf_C(r0, c)
    Sinfmax = find_Sinf_A(r0, c)
    μinfmax = c/(1-Sinfmax)
    μinfguess = [0, μinfmax]
    answer_sheet = Dict{Real, Real}()  # maps Sinf -> μinf that hits the target R0
    opt = optimize(Sinfmin, Sinfmax) do (Sinf_target,)
        snipes!(answer_sheet, r0, c, Sinf_target, 0, μinfmax; kwargs...)
    end
    Sinfbest = opt.minimizer
    μinfbest = answer_sheet[Sinfbest]
    foundbest = shoot(Sinfbest, c, μinfbest; kwargs...)
    return Record(foundbest, r0, 2)
end

"""
TODO: Update comments
"""
function hunt_inactive(r0::Real, c::Real; kwargs...)
    Sinf_inactive = find_Sinf_C(r0, c)
    inactive = shoot(Sinf_inactive, c, 0; kwargs...)
    return Record(inactive, r0, 1)
end

function hunt_doormat(r0::Real, c::Real; kwargs...)
    Sinfmin = find_Sinf_A(r0, c)
    μinfmin = c/(1-Sinfmin)
    μinfmax = c/(1-max_Sinf(c))
    Sinfmax = constrained_aimed_shot(r0, c, Sinfmin) do r0, c, Sinf
        μinfmax
    end
    μinfguess = [μinfmin, μinfmax]
    answer_sheet = Dict{Real, Real}()  # maps Sinf -> μinf that hits the target R0
    opt = optimize(Sinfmin, Sinfmax) do (Sinf_target,)
        snipe!(answer_sheet, r0, c, Sinf_target, μinfmin, μinfmax; kwargs...)
    end
    Sinfbest = opt.minimizer
    μinfbest = answer_sheet[Sinfbest]
    foundbest = shoot(Sinfbest, c, μinfbest; kwargs...)
    return Record(foundbest, r0, 3)
end

# ============================================================================
#   Bullet Interface
# ============================================================================

struct Bullet
    Sinf::Real
    qinf::Real
    μinf::Real
    any_p::Bool
    any_v::Bool
    sol::ODESolution
end

infer_r0(blt::Bullet) = infer_r0(blt.sol[:, end])
refer_kinf(blt::Bullet) = refer_kinf(blt.Sinf, blt.qinf, blt.μinf)
score(blt::Bullet) = sum(objectives(blt))
aim(blt::Bullet, r0::Real) = infer_r0(blt)-r0

function objectives(blt::Bullet)
    objectives(blt.sol, blt.Sinf, blt.qinf) .* Bool[blt.any_p, blt.any_v, true]
end

# ============================================================================
#   Record Interface
# ============================================================================

struct Record
    branch::Int
    any_p::Bool
    any_v::Bool
    target_r0::Real
    r0err::Real

    t::Vector{Real}
    X::Matrix{Real}
    Ip::Real
    pidx::Int            # indicates when infections reach its peak level

    j1p::Real
    j1v::Real
    j1::Real
    j2::Real
    obj::Real

    r0::Real
    c::Real
    q0::Real
    μ0::Real
    k0::Real

    Sinf::Real
    qinf::Real
    μinf::Real
    kinf::Real
end

Record(ntpl::NamedTuple) = Record([ntpl[f] for f in fieldnames(Record)]...)
as_ntuple(obj::T) where {T} = NamedTuple{fieldnames(T)}(getfield(obj, f) for f in fieldnames(T))
as_ntuple(ntpl::NamedTuple) = ntpl

"""Promote *Bullet* into *Record*"""
function Record(blt::Bullet, target_r0::Real, branch::Int)
    t, X, pidx, Ip = timeseries_processing(blt.sol, target_r0)
    j1p, j1v, j2 = objectives(blt)
    r0 = infer_r0(blt)
    r0err = r0-target_r0
    q0 = blt.sol[3, end] * blt.sol[1, end] / (blt.sol[2, end] * r0)
    μ0 = blt.sol[4, end] * blt.sol[1, end] / (blt.sol[2, end] * r0)
    k0 = blt.sol[5, end]
    kinf = refer_kinf(blt)
    ntpl = (; branch, blt.any_p, blt.any_v, target_r0, r0err,
            t, X, Ip, pidx,
            j1p, j1v, j1=j1p+j1v, j2, obj=j1p+j1v+j2,
            r0, c=blt.qinf, q0, μ0, k0,
            Sinf=blt.Sinf, qinf=blt.qinf, μinf=blt.μinf, kinf)
    Record(ntpl)
end

"""Default Record for (r0, c)"""
function Record(r0::Real, c::Real)
    t = [0.0]  # One can use `t[1] == 0` as an identifier of the "default" Record.
    X = zeros(6, 1)
    X[1, 1] = r0
    Ip = null_Ip(r0)
    Sinf = null_Sinf(r0)
    qinf = c
    μinf = 0.0  # c*(1+kinf)/(1-Sinf)
    j2 = 1.0 - Sinf/r0
    q0 = qinf*Sinf/r0
    μ0 = μinf*Sinf/r0
    ntpl = (; branch=0, any_p=false, any_v=false, target_r0=r0, r0err=0.0,
            t, X, Ip, pidx=0,
            j1p=0.0, j1v=0.0, j1=0.0, j2, obj=j2,
            r0, c, q0, μ0, k0=0.0,
            Sinf, qinf, μinf, kinf=0.0)
    Record(ntpl)
end

# ============================================================================
#   Processing & Interpreting Results
# ============================================================================

function infer_r0(Xi)
    meps = -Xi[1]*exp(-Xi[1]-Xi[2]) / (1.0-1e-13)
    if (meps > 0) || (meps < Commons.IE)
        throw(DomainError("M/IE-1=$(meps/Commons.IE-1)"))
    end
    -lambertw(meps, -1)
end

function refer_kinf(Sinf::Real, qinf::Real, μinf::Real)
    (μinf*(1-Sinf) / qinf) - 1
end

function objectives(sol::ODESolution, Sinf::Real, qinf::Real)
    c = qinf
    r0 = infer_r0(sol[:, end])
    j1p = -sol[6, end] / (r0*c)
    j1v = let
        sq_start = sol[1, end] * sol[3, end] / sol[2, end]
        sq_stop = sol[1, 1] * sol[3, 1] / sol[2, 1]
        (sq_stop - sq_start) / (c * r0)
    end
    j2 = 1.0 - Sinf/r0
    Real[j1p, j1v, j2]
end

function timeseries_processing(sol::ODESolution, r0::Real)
    Δt = preceding_t(r0, sol[1, end])
    t = reverse(sol.t) .-sol.t[end] .+Δt
    X = reverse(Array(sol), dims=2)
    X[6:end, :] .-= X[6:end, 1]
    if sol[1, end] < 1
        pidx = 0
        Ip = null_Ip(r0)
    else
        Ip, pidx = findmax(X[2, :])
    end
    fewer(t, X, pidx)..., Ip
end

function fewer(t, X, whitelist...)
    len = length(t)
    mask = @views [k == 1 || t[k] != t[k-1] || X[:, k] != X[:, k-1] for k in 1:len]
    valid = collect(filter(!=(0), whitelist))
    mask[valid] .= true
    cs = cumsum(mask)
    remapped = map(w -> w == 0 ? 0 : cs[w], whitelist)
    return t[mask], X[:, mask], remapped...
end

# ============================================================================
#   Shooting Internals
# ============================================================================

function _legacy_null_Sinf_guess(r0::Real)
    -Commons.lambertw(-r0 * exp(-r0), 0)
end

function _default_Sinf_guess(r0::Real, Sinfguess)
    # The shooting tolerances and sample layout were calibrated with the legacy
    # zero-infection separatrix as the starting seed. Use that value only to
    # choose the root-finder basin; the equations below still target the
    # current initial-infection convention.
    isnothing(Sinfguess) ? _legacy_null_Sinf_guess(r0) : Sinfguess
end

"""Previously find_Sinf_inactive"""
function find_Sinf_C(r0::Real, c::Real, Sinfguess=nothing; kwargs...)
    Sinfguess = _default_Sinf_guess(r0, Sinfguess)
    find_zero(Sinfguess; atol=1e-13*r0/(r0-1), rtol=0, xatol=1e-15, xrtol=1e-13, kwargs...) do Sinf
        Sinf = clamp(Sinf, 0, 1-1e-5)
        aim(shoot(Sinf, c, 0), r0)
    end
end

"""TODO add"""
function find_Sinf_Bl(r0::Real, c::Real, Sinfguess=nothing; kwargs...)
    Sinfguess = _default_Sinf_guess(r0, Sinfguess)
    find_zero(Sinfguess; atol=1e-13*r0/(r0-1), rtol=0, xatol=1e-15, xrtol=1e-13, kwargs...) do Sinf
        Sinf = clamp(Sinf, 0, 1-1e-5)
        aim(shoot(Sinf, c, inv(exp(Sinf-1.0)-Sinf)), r0)
    end
end

"""Previously find_Sinfmin"""
function find_Sinf_Bu(r0::Real, c::Real, Sinfguess=nothing; kwargs...)
    Sinfguess = _default_Sinf_guess(r0, Sinfguess)
    find_zero(Sinfguess; atol=1e-13*r0/(r0-1), rtol=0, xatol=1e-15, xrtol=1e-13, kwargs...) do Sinf
        Sinf = clamp(Sinf, 0, 1-1e-5)
        aim(shoot(Sinf, c, c), r0)
    end
end

"""Previously find_Sinfmax"""
function find_Sinf_A(r0::Real, c::Real, Sinfguess=nothing; kwargs...)
    Sinfguess = _default_Sinf_guess(r0, Sinfguess)
    find_zero(Sinfguess; atol=1e-13*r0/(r0-1), rtol=0, xatol=1e-15, xrtol=1e-13, kwargs...) do Sinf
        Sinf = clamp(Sinf, 0, 1-1e-5)
        aim(shoot(Sinf, c, c/(1-Sinf)), r0)
    end
end

function constrained_aimed_shot(f, r0::Real, c::Real, Sinfguess=nothing; kwargs...)
    Sinfguess = _default_Sinf_guess(r0, Sinfguess)
    find_zero(Sinfguess; atol=1e-13*r0/(r0-1), rtol=0, xatol=1e-15, xrtol=1e-13, kwargs...) do Sinf
        Sinf = clamp(Sinf, 0, 1-1e-5)
        aim(shoot(Sinf, c, f(r0, c, Sinf)), r0)
    end
end

function find_μinf_ub(r0::Real, c::Real, μinfguess=nothing; kwargs...)
    Sinf_ub = max_Sinf(c)
    μinfguess = isnothing(μinfguess) ? c/(1-Sinf_ub) : μinfguess
    find_zero(μinfguess; atol=1e-13*r0/(r0-1), rtol=0, xatol=1e-15, xrtol=1e-13, kwargs...) do μinf
        aim(shoot(Sinf_ub, c, μinf), r0)
    end
end

# Dynamical Systems

function dXdt(dX, X, para, t)
    S, I, varphi, F, k, pS = X  # S, I ∈ [0, r0]  while  φ >= 1
    p, v = action(X, para)
    a = p+v
    sig = sigma(a)
    dX[1] = -sig*S*I
    dX[2] = +sig*S*I -I
    dX[3] = +(sig*(S+I)-1.0)*varphi
    dX[4] = +(sig*(S+I)-1.0)*F +p*(S+I)
    dX[5] = 0.0
    dX[6] = p*S
    if !para[2] && para[1]
        temp = (1.0+k - F/varphi)
        dX[3] += v*I
        dX[4] += (v*(1.0+k) -0.5*sqrt(varphi)*temp)*(S+I)
        dX[5] += -0.5*I/sqrt(varphi) * temp
    end
end

function halting_condition(Sinf, qinf, μinf)
    kinf = refer_kinf(Sinf, qinf, μinf)
    if Sinf >= 1.0
        return [Sinf, 0.0, qinf, μinf, kinf, 0.0]
    end
    m = -Sinf * exp(-Sinf)
    if below(Sinf, qinf, μinf)  # if below(...) is true, the system never practice distancing
        S = -m*exp(1)
        I = 1.0 -S
        factor = I*Sinf/S
        [S, I, qinf*factor, μinf*factor, kinf, 0.0]
    elseif qinf == μinf
        slope = 1 / (Sinf * qinf)
        S, I = given_slope(slope, m, 0)
        [S, I, 1.0, 1.0, kinf, 0.0]
    elseif qinf > μinf
        slope = 1 / (Sinf * qinf)
        S, I = given_slope(slope, m, 0)
        factor = I*Sinf/S
        [S, I, 1.0, μinf*factor, kinf, 0.0]
    else
        slope = 1 / (Sinf * μinf)
        S, I = given_slope(slope, m, 0)
        factor = I*Sinf/S
        [S, I, qinf*factor, 1.0, kinf, 0.0]
    end
end

function below(Sinf, qinf, μinf)
    max(qinf, μinf) <= inv(exp(Sinf-1)-Sinf)
end

function set_para(Xf)
    para1 = Xf[4] <= Xf[3]
    para2 = Xf[5] >= 0
    para3 = Xf[3] < 1
    para4 = Xf[4] < 1
    [para1, para2, para3, para4, false, false, false]
end

function action(X, para)
    if para[3] && para[4]
        return (0.0, 0.0)
    elseif para[2]  # k >= 0
        (sqrt(maxtivation(X))-1.0, 0.0)  # maxtivation = X -> max(X[3], X[4])
    elseif para[1]  # right-hand-side of phi-space
        (0.0, sqrt(X[3])-1.0)
    else
        (sqrt(X[4])-1.0, 0.0)
    end
end

# Misc.

function action(X)
    if X[5] >= 0
        (clipped_sqrt(maxtivation(X)), 0.0)
    elseif X[4] > Fthr(X)
        (clipped_sqrt(X[4]), 0.0)
    else
        (0.0, clipped_sqrt(X[3]))
    end
end

action(intg::DEIntegrator) = action(intg.u, intg.p)

function maxtivation(X)
    max(X[3], X[4])
end

function Fthr(X)
    if X[5] >= 0.0
        return X[3]  # possible alternative: motivation(X)
    elseif X[3] >= 1.0
        a = sqrt(X[3])
        b = sqrt(X[5]*(a-X[3]))
        return (a+b)^2
    else
        return 0.0 #1.0 -1e-7  # -1e-7 for some minor numerical details.
    end
end

# ============================================================================
#   Callback Internals
# ============================================================================

"""
  index=1: Crossing ΔH=0 manifold
  index=2: Crossing k=0 manifold
  index=3: Crossing varphi=0
  index=4: Crossing F=0
  index=5: Reaching infectious peak

"""
function _callback_extra_tuple(extra_callbacks)
    isnothing(extra_callbacks) && return ()
    extra_callbacks isa Tuple && return extra_callbacks
    extra_callbacks isa AbstractVector && return tuple(extra_callbacks...)
    return (extra_callbacks,)
end

function _merge_callbacks(callback, extra_callbacks)
    extras = _callback_extra_tuple(extra_callbacks)
    isempty(extras) && return callback
    return CallbackSet(callback, extras...)
end

function _raw_adaptive_callback(; extra_callbacks=(), kwargs...)
    callback = VectorContinuousCallback(
        condition,
        affect!(:silent),
        affect_neg!(:silent),
        5;
        kwargs...,
    )
    base = CallbackSet(t5cb, t6cb, btcb, callback)
    return _merge_callbacks(base, extra_callbacks)
end

"""
    adaptive_callback(; kwargs...)

Build the adaptive-model callback used by `shoot`.

By default, near-simultaneous events are coalesced and the `F=Fthr`
surface is domain-gated. Use `coalesce_events=false` for the legacy raw
`VectorContinuousCallback` behavior. `extra_callbacks` may be a callback, a
tuple, or a vector and is merged into the returned `CallbackSet`.
"""
function adaptive_callback(;
    extra_callbacks=(),
    coalesce_events::Bool=true,
    coalesce_abstol::Real=1e-12,
    release_abstol::Real=coalesce_abstol,
    callback_abstol::Real=coalesce_abstol,
    repeat_nudge=1 // 100,
    interp_points::Int=10,
    kwargs...,
)
    if !coalesce_events
        return _raw_adaptive_callback(; extra_callbacks=extra_callbacks, kwargs...)
    end

    @assert coalesce_abstol > 0 "coalesce_abstol must be positive"
    @assert release_abstol > 0 "release_abstol must be positive"
    @assert callback_abstol >= 0 "callback_abstol must be nonnegative"

    condition_state = CoalescingCondition(
        _domain_condition,
        zeros(Int8, 5),
        Float64(coalesce_abstol),
        Float64(release_abstol),
    )
    initialize = function (callback, u, t, integrator)
        fill!(condition_state.held_signs, 0)
        u_modified!(integrator, false)
    end
    callback = VectorContinuousCallback(
        condition_state,
        CoalescingAffect(condition_state, :silent, Int8(1)),
        CoalescingAffect(condition_state, :silent, Int8(-1)),
        5;
        initialize=initialize,
        abstol=Float64(callback_abstol),
        repeat_nudge=repeat_nudge,
        interp_points=interp_points,
        kwargs...,
    )
    base = CallbackSet(t5cb, t6cb, btcb, callback)
    return _merge_callbacks(base, extra_callbacks)
end

default_callback(; kwargs...) = adaptive_callback(; kwargs...)

function condition(out, u, t, integrator)
    out[1] = if integrator.p[2]
        -1.0  # The crossing could only occur when k(=λq/S) < 0
    elseif (u[5] < 0) && (u[3] <= 1)
        u[4] - u[3]
    else
        u[4] - Fthr(u)
    end
    out[2] = u[5]
    out[3] = u[3] -1
    out[4] = u[4] -1
    p, v = action(u, integrator.p)
    sig = clamp(sigma(p+v), 0, 1)
    out[5] = sig*u[1] -1
end

function _domain_condition(out, u, t, integrator)
    out[1] = if integrator.p[2] || u[5] >= 0 || u[3] < 1 || u[4] < 1
        integrator.p[1] ? -1.0 : 1.0
    else
        u[4] - Fthr(u)
    end
    out[2] = u[5]
    out[3] = u[3] - 1
    out[4] = u[4] - 1
    p, v = action(u, integrator.p)
    sig = clamp(sigma(p + v), 0, 1)
    out[5] = sig * u[1] - 1
    return nothing
end

affect!(mode::Symbol) = (x, y)->affect!(Val(mode), x, y)
affect!(x::DEIntegrator, y::Real) = affect!(Val(:silent), x, y)

"""from 0⁻ to 0⁺"""
function affect!(::Val{:silent}, integrator::DEIntegrator, index::Real)
    if index == 1
        integrator.p[1] = false
        neutralized(integrator) ? neutralize!(integrator) : nothing
        validate_para3!(integrator); validate_para4!(integrator)
        check_traverse_tl(integrator) && (integrator.p[7] = true)
        check_traverse_br(integrator) && (integrator.p[7] = true)
        check_divergence(integrator) && (integrator.p[7] = true)
        no_more_dist(integrator) ? terminate!(integrator) : nothing
    elseif index == 2
        neutralize!(integrator)
        validate_para3!(integrator); validate_para4!(integrator)
        no_more_dist(integrator) ? terminate!(integrator) : nothing
    elseif index == 3
        integrator.p[3] = false
        neutralized(integrator) ? neutralize!(integrator) : nothing
        validate_para4!(integrator)
    elseif index == 4
        integrator.p[4] = false
        neutralized(integrator) ? neutralize!(integrator) : nothing
        validate_para3!(integrator)
    else
        neutralized(integrator) ? neutralize!(integrator) : nothing
        validate_para3!(integrator); validate_para4!(integrator)
        no_more_dist(integrator) ? terminate!(integrator) : nothing
    end
end

affect_neg!(mode::Symbol) = (x, y)->affect_neg!(Val(mode), x, y)
affect_neg!(x::DEIntegrator, y::Real) = affect_neg!(Val(:silent), x, y)

"""from 0⁺ to 0⁻ """
function affect_neg!(::Val{:silent}, integrator::DEIntegrator, index::Real)
    if index == 1
        integrator.p[1] = true
        neutralized(integrator) ? neutralize!(integrator) : nothing
        validate_para3!(integrator); validate_para4!(integrator)
        check_traverse_tl(integrator) && (integrator.p[7] = true)
        check_traverse_br(integrator) && (integrator.p[7] = true)
        check_divergence(integrator) && (integrator.p[7] = true)
        no_more_dist(integrator) ? terminate!(integrator) : nothing
    elseif index == 2
        neutralize!(integrator)
        validate_para3!(integrator); validate_para4!(integrator)
        no_more_dist(integrator) ? terminate!(integrator) : nothing
    elseif index == 3
        integrator.p[3] = true
        neutralized(integrator) ? neutralize!(integrator) : nothing
        validate_para4!(integrator)
        integrator.p[4] ? terminate!(integrator) : nothing
        no_more_dist(integrator) ? terminate!(integrator) : nothing
    elseif index == 4
        integrator.p[4] = true
        neutralized(integrator) ? neutralize!(integrator) : nothing
        validate_para3!(integrator)
        integrator.p[3] ? terminate!(integrator) : nothing
        no_more_dist(integrator) ? terminate!(integrator) : nothing
    else  # index == 4, peak
        neutralized(integrator) ? neutralize!(integrator) : nothing
        validate_para3!(integrator); validate_para4!(integrator)
        no_more_dist(integrator) ? terminate!(integrator) : nothing
    end
end

function below_threshold(u, t, integrator)
    maxtivation(u) < 0.99  # this is only secondary safeguard.
end

btcb = DiscreteCallback(below_threshold, terminate!)

function ever_p(u, t, integrator)
    !integrator.p[5] && (action(u, integrator.p)[1] > 0)
end

toggle_p5!(intg) = (intg.p[5] = true)

t5cb = DiscreteCallback(ever_p, toggle_p5!)

function ever_v(u, t, integrator)
    !integrator.p[6] && (action(u, integrator.p)[2] > 0)
end

toggle_p6!(intg) = (intg.p[6] = true)

t6cb = DiscreteCallback(ever_v, toggle_p6!)

mutable struct CoalescingCondition{F}
    raw_condition::F
    held_signs::Vector{Int8}
    coalesce_abstol::Float64
    release_abstol::Float64
end

function (condition::CoalescingCondition)(out, u, t, integrator)
    condition.raw_condition(out, u, t, integrator)

    @inbounds for index in eachindex(condition.held_signs)
        held_sign = condition.held_signs[index]
        iszero(held_sign) && continue

        if t == integrator.tprev &&
                held_sign * out[index] > condition.release_abstol
            condition.held_signs[index] = 0
        else
            out[index] = held_sign * condition.release_abstol
        end
    end
    return nothing
end

struct CoalescingAffect{C}
    condition::C
    mode::Symbol
    direction::Int8
end

function _approaches_event(previous::Real, current::Real, tolerance::Real)
    abs(current) <= tolerance || return false
    return abs(current) <= abs(previous) ||
           abs(previous) <= tolerance ||
           previous * current <= 0
end

function _crossing_direction(previous::Real, current::Real)
    change = current - previous
    scale = max(1.0, abs(previous), abs(current))
    change_tolerance = 16eps(Float64) * scale

    change > change_tolerance && return Int8(1)
    change < -change_tolerance && return Int8(-1)
    current > 0 && return Int8(1)
    current < 0 && return Int8(-1)
    return Int8(1)
end

function _sync_inactive_mode!(
    integrator,
    indices::Vector{Int},
    directions::Vector{Int8},
)
    integrator.p[2] && return nothing

    varphi_at_least_one = integrator.u[3] >= 1
    F_at_least_one = integrator.u[4] >= 1
    for (index, direction) in zip(indices, directions)
        index == 3 && (varphi_at_least_one = direction > 0)
        index == 4 && (F_at_least_one = direction > 0)
    end

    if !varphi_at_least_one && F_at_least_one
        integrator.p[1] = false
    elseif varphi_at_least_one && !F_at_least_one
        integrator.p[1] = true
    end
    return nothing
end

function (affect::CoalescingAffect)(integrator, triggered_index)
    condition_state = affect.condition
    current = zeros(Float64, length(condition_state.held_signs))
    previous = similar(current)
    condition_state.raw_condition(current, integrator.u, integrator.t, integrator)
    condition_state.raw_condition(
        previous,
        integrator.uprev,
        integrator.tprev,
        integrator,
    )

    indices = Int[]
    directions = Int8[]
    for index in eachindex(current)
        if index == triggered_index ||
                _approaches_event(
                    previous[index],
                    current[index],
                    condition_state.coalesce_abstol,
                )
            push!(indices, index)
            push!(
                directions,
                index == triggered_index ?
                affect.direction :
                _crossing_direction(previous[index], current[index]),
            )
        end
    end

    skip_threshold = 1 in indices && any(
        pair -> first(pair) == 2 || (first(pair) == 4 && last(pair) < 0),
        zip(indices, directions),
    )
    for (index, direction) in zip(indices, directions)
        if skip_threshold && index == 1
            condition_state.held_signs[index] = direction
            continue
        end

        if direction > 0
            affect!(Val(affect.mode), integrator, index)
        else
            affect_neg!(Val(affect.mode), integrator, index)
        end
        condition_state.held_signs[index] = direction
    end
    _sync_inactive_mode!(integrator, indices, directions)
    return nothing
end

# Misc.

function neutralized(intg)
    k = intg.u[5]
    kprev = intg.uprev[5]
    sign(kprev) < 0 && sign(k) > 0  # this is only possible due to numerical precision
end

function neutralize!(intg)
    intg.u[5] = 0.0
    intg.p[2] = true
end

function no_more_dist(intg)
    maxtivation(intg.u) <= 1.0  # maxtivation = X -> max(X[3], X[4])
end

function validate_para3!(intg)
    vpre = sign(intg.uprev[3]-1)
    vcur = sign(intg.u[3]-1)
    ready_terminate = (vpre > vcur)
    if (vpre != vcur) && (intg.p[3] != ready_terminate)
        intg.p[3] = ready_terminate
    end
end

function validate_para4!(intg)
    Fpre = sign(intg.uprev[4]-1)
    Fcur = sign(intg.u[4]-1)
    ready_terminate = (Fpre > Fcur)
    if (Fpre != Fcur) && (intg.p[4] != ready_terminate)
        intg.p[4] = ready_terminate
    end
end

function normal_velocity_Fthr(intg, p)
    if (intg.u[5] >= 0) || (intg.u[3] <= 1) || (intg.u[4] <= 1)
        return nothing
    end
    out = zeros(6)
    dXdt(out, intg.u, p, 0.0)
    y = 1/sqrt(intg.u[3])-1
    z = sqrt(intg.u[5]*y)
    isfinite(z) && !iszero(z) || return nothing
    dFthr = ((1+z)-intg.u[5]*(1+y)/(2*z))*(1+z) * out[3] + intg.u[3]*(1+z)*y/z * out[5]
    dG = out[4] - dFthr
    isfinite(dG) ? dG : nothing
end

function check_traverse_br(intg)
    dG = normal_velocity_Fthr(intg, intg.p)
    !isnothing(dG) && iszero(dG)
end

function check_traverse_tl(intg)
    p_other = [!intg.p[1], intg.p[2:end]...]
    dG = normal_velocity_Fthr(intg, p_other)
    !isnothing(dG) && iszero(dG)
end

function check_divergence(intg)
    dG_current = normal_velocity_Fthr(intg, intg.p)
    isnothing(dG_current) && return false

    p_other = [!intg.p[1], intg.p[2:end]...]
    dG_other = normal_velocity_Fthr(intg, p_other)
    isnothing(dG_other) && return false

    dG_current * dG_other < 0
end

# ============================================================================
#   Misc.
# ============================================================================

function infer_r0(Sinf::Real, c::Real, μinf::Real)
    blt = shoot(Sinf, c, μinf)
    infer_r0(blt)
end

function score(Sinf::Real, c::Real, μinf::Real)
    blt = shoot(Sinf, c, μinf)
    score(blt)
end

function aim_and_score(r0::Real, c::Real, Sinf::Real, μinf::Real)
    blt = shoot(Sinf, c, μinf)
    aim(blt, r0), score(blt)
end

function min_c(r0)
    Sinf = null_Sinf(r0)
    r0*initial_susceptible*min_adj0(r0)/Sinf * (1.0-Sinf)
end

function max_Sinf(c)
    a = 1/(1-c)
    b = c*a
    a - Commons.lambertw(b*exp(b), 0)
end

function _resolve_u0(u0, Sinf, qinf, μinf)
    applicable(u0, Sinf, qinf, μinf) ? u0(Sinf, qinf, μinf) : u0
end

function _resolve_p(p, u)
    applicable(p, u) ? p(u) : p
end

function _default_tspan(Sinf, qinf, μinf)
    below(Sinf, qinf, μinf) ? (1000.0, 1000.0) : (1000.0, 0.0)
end

function _ode_function(f)
    f isa ODEFunction ? f : ODEFunction(f)
end

function initiate(Sinf::Real, qinf::Real, μinf::Real;
    f=dXdt,
    u0=halting_condition,
    p=set_para,
    tspan=nothing,
    initializer::Symbol=:zeroing,
    callback=adaptive_callback(),
    save_everystep=false,
    abstol=1e-13,
    reltol=abstol,
    alg=Vern9(lazy=false),
    kwargs...,
)
    u = _resolve_u0(u0, Sinf, qinf, μinf)
    para = _resolve_p(p, u)
    span = isnothing(tspan) ? _default_tspan(Sinf, qinf, μinf) : tspan
    prob = ODEProblem(_ode_function(f), u, span, para)

    if initializer === :zeroing
        return zeroing(prob, alg;
            tol=abstol,
            size=length(u),
            callback=callback,
            save_everystep=save_everystep,
            kwargs...,
        )
    elseif initializer === :direct
        return init(prob, alg;
            abstol=abstol, reltol=reltol,
            callback=callback,
            save_everystep=save_everystep,
            kwargs...,
        )
    end

    @assert false "initializer must be :zeroing or :direct"
end
