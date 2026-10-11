using Distributed: pmap
using Roots: find_zero, ConvergenceFailed
using SciMLBase: DEIntegrator, RightRootFind
using OrdinaryDiffEq: ODEFunction, ODEProblem, ODESolution, solve, Vern9,
                      DiscreteCallback, CallbackSet, terminate!
using PyFormattedStrings: @f_str
using Optim: optimize

const R0S = 1.1:0.1:15.0
const CS = 1.0:1.0:150.0
const zero_tols = pairs((; atol=0.0, rtol=0.0))  # for find_zeros
const zero_tol = pairs((; atol=0.0, rtol=0.0, xatol=0.0, xrtol=0.0))  # for find_zero

# ============================================================================
#   Shooting Interface
# ============================================================================

function shoot(r0::Real, q0::Real, λ0::Real;
               tmax=1000.0, callback=default_callback, save_everystep=false, kwargs...)
    f = ODEFunction(dXdt)
    X0 = initial_condition(r0, λ0)
    tspan = (0.0, tmax)
    para = [0.0, q0*r0*initial_susceptible]  # flag & value of qS
    prob = ODEProblem(f, X0, tspan, para)
    cb = instantiate_callback(callback)
    sol = solve(prob, Vern9(), abstol=1e-20, reltol=1e-15,
                callback=cb, save_everystep=save_everystep; kwargs...)
    Bullet(r0, q0, λ0, sol)
end

function aim(r0::Real, c::Real, q0::Real, λ0::Real)
    q0 = (isnothing(q0) ? 0.0 : max(q0, 0.0))
    try
        blt = shoot(r0, q0, λ0)
        infer_c(blt) -c
    catch e
        println("R0=$(r0), c=$(c), q0=$(q0), λ0=$(λ0)")
        rethrow(e)
    end
end

function aimed_shot(r0::Real, c::Real, cpop::Real, λ0::Real, q0guess=nothing;
                    branch::Int=-1, dcpop::Real=0, dλ0::Real=0, dJ1pop::Real=0, kwargs...)
    q0guess = (isnothing(q0guess) ? 0.0 : max(q0guess, 0.0))
    q0 = find_zero(x->aim(r0, c, x, λ0), q0guess; zero_tol...)
    blt = shoot(r0, q0, λ0; kwargs...)
    branch = (branch >= 0) ? branch : Int(cpop > min_c(r0))
    dq0, dJ1 = if dcpop != 0
        dq0 = find_zero(x->aim(r0, c, x, λ0+dλ0), q0; zero_tol...) -q0
        dJ1 = objectives(shoot(r0, q0+dq0, λ0+dλ0; kwargs...))[1] -objectives(blt)[1]
        (dq0, dJ1)
    else
        (0, 0)
    end
    Record(blt, branch, c, cpop, dcpop, dλ0, dJ1pop, dq0, dJ1)
end

# ============================================================================
#   Bullet Interface
# ============================================================================

struct Bullet
    r0::Real
    q0::Real
    λ0::Real
    sol::ODESolution
end

infer_c(blt::Bullet) = infer_c(blt.sol[:, end], blt.q0*blt.r0*initial_susceptible)
objectives(blt::Bullet) = objectives(blt.sol, blt.r0, blt.q0)
aim(blt::Bullet, c::Real) = aim(blt.r0, c, blt.q0, blt.λ0)

function reproduce(obj; callback=default_callback, save_everystep=false, kwargs...)
    blt = shoot(obj.r0, obj.q0, obj.λ0; callback=callback, save_everystep=save_everystep, kwargs...)
    if hasproperty(obj, :target_c)
        return Record(blt, obj.branch, obj.target_c, obj.cpop, obj.dc, obj.dλ0, obj.dJ1)
    end
    return blt
end

shoot(obj; kwargs...) = shoot(obj.r0, obj.q0, obj.λ0; kwargs...)

# ============================================================================
#   Record Interface
# ============================================================================

struct Record
    branch::Int           # 0/1: the reservoir is inactive/active
    target_c::Real
    cerr::Real

    t::Vector{Real}
    X::Matrix{Real}
    Ip::Real
    Ipp::Real
    pidx::Int             # indicates when infections reach its peak level
    pidxpop::Int          # indicates when infections reach its peak level

    J1::Real
    J2::Real
    J::Real

    J1pop::Real
    J2pop::Real
    Jpop::Real

    r0::Real
    c::Real
    cpop::Real
    q0::Real
    λ0::Real

    dcpop::Real
    dλ0::Real
    dJ1pop::Real
    dq0::Real
    dJ1::Real
end

Record(ntpl::NamedTuple) = Record([ntpl[f] for f in fieldnames(Record)]...)
as_ntuple(obj::T) where {T} = NamedTuple{fieldnames(T)}(getfield(obj, f) for f in fieldnames(T))
as_ntuple(ntpl::NamedTuple) = ntpl

"""Promote *Bullet* into *Record*"""
function Record(blt::Bullet, branch::Int, target_c::Real, cpop::Real,
                dcpop::Real, dλ0::Real, dJ1pop::Real, dq0::Real, dJ1::Real)
    t, X, pidx, pidxpop, Ip, Ipp = timeseries_processing(blt.sol, blt.r0)
    J1, J2, J1pop, J2pop = objectives(blt)
    c = infer_c(blt)
    cerr = c-target_c
    ntpl = (; branch, target_c, cerr,
            t, X, Ip, Ipp, pidx, pidxpop,
            J1, J2, J=J1+c*J2, J1pop=J1pop, J2pop=J2pop, Jpop=J1pop+cpop*J2pop,
            r0=blt.r0, c, cpop, q0=blt.q0, λ0=blt.λ0, dcpop, dλ0, dJ1pop, dq0, dJ1)
    Record(ntpl)
end

# ============================================================================
#   Processing & Interpreting Results
# ============================================================================

function infer_c(Xf, qS)
    Xf[3] + qS
    # mfpop = -Xf[4] * exp(-Xf[4] - Xf[5])
    # if (mfpop > 0) || (mfpop < Commons.IE)
    #     throw(DomainError("M/IE-1=$(mfpop/Commons.IE-1)"))
    # end
    # Sinfpop = -lambertw(mfpop, 0)
    # Sinf = Sinfpop*Xf[1]/Xf[4]
    # qS/Sinf
end

function objectives(sol::ODESolution, r0::Real, q0::Real)
    c = infer_c(sol[:, end], q0*r0*initial_susceptible)
    J1pop = if maximum(sol[6, :]) >= 1
        sol[end, end] / r0
    else
        0.0
    end
    J1, J2, J2pop = let
        mpop = -sol[4, end] * exp(-sol[4, end] - sol[5, end])
        spopinf = -lambertw(mpop, 0) / r0
        sinf = spopinf * sol[1, end] / sol[4, end]
        j1 = sol[3, end] / r0
        j1*sinf, (1.0 - sinf), (1.0 - spopinf)
    end
    Real[J1, J2, J1pop, J2pop]
end

function timeseries_processing(sol::ODESolution, r0::Real)
    Δt = preceding_t(r0, sol[1, 1])  # This is guaranteed to be zero, but kept for consistency with `Nash` and `Ctrl`
    t = sol.t .+Δt
    X = sol[:, :]
    Ip, pidx = findmax(X[2, :])
    Ipp, pidxpop = findmax(X[5, :])
    fewer(t, X, pidx, pidxpop)..., Ip, Ipp
end

timeseries_processing(blt) = timeseries_processing(blt.sol, blt.r0)

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

function dXdt(dX, X, para, t)
    S, I, J1, Spop, Ipop, varPhi = X  # S, I ∈ [0, r0]  while  φ >= 1
    vpop, v = action(X, para)
    sig = sigma(v)
    sigpop = sigma(vpop)
    dX[1] = -sig*S*Ipop
    dX[2] = +sig*S*Ipop -I
    dX[3] = v
    dX[4] = -sigpop*Spop*Ipop
    dX[5] = +sigpop*Spop*Ipop -Ipop
    dX[6] = (sigpop*(Spop+Ipop) -1.0)*varPhi +vpop*(Spop+Ipop)
    dX[7] = vpop*Spop
end

function initial_condition(r0, λ0)
    S0 = r0*initial_susceptible
    I0 = r0*initial_infected
    [S0, I0, 0.0, S0, I0, λ0*I0, 0.0]
end

function action(X, para)
    vpop = clipped_sqrt(X[6])
    v = clipped_sqrt(X[5]*para[2])
    return vpop, v
end

# ============================================================================
#   Callback Internals
# ============================================================================

function last_one_standing(X, t, integrator)
    max(X[2], X[5]) < infection_floor
end

function peak(X, t, integrator)
    vpop, v = action(X, integrator.p)
    sig = sigma(v)
    sig*X[1]*X[5] -X[2]
end

function peak_pop(X, t, integrator)
    vpop, v = action(X, integrator.p)
    sigpop = sigma(vpop)
    sigpop*X[4]-1.0
end

function start_distancing(out, X, t, integrator)
    out[1] = X[5]*integrator.p[2]-1.0
    out[2] = X[6]-1.0
end

function stop_distancing(out, X, t, integrator)
    out[1] = X[5]*integrator.p[2]-1.0
    out[2] = X[6]-1.0
end

function start_distancing!(integrator, idx)
    nothing
end

function stop_distancing!(integrator, idx)
    maxphi = max(integrator.u[5]*integrator.p[2], integrator.u[6])
    if maxphi <= 1.0 || isapprox(maxphi, 1.0)
        terminate!(integrator)
    end
end

function start_distancing_discrete(X, t, integrator)
    if integrator.p[1] == 0
        return max(X[5]*integrator.p[2], X[6]) >= 1.0
    else
        return false
    end
end

function stop_distancing_discrete(X, t, integrator)
    if integrator.p[1] == 1
        return max(X[5]*integrator.p[2], X[6]) < 1.0
    else
        return false
    end
end

function start_distancing_discrete!(integrator)
    integrator.p[1] = 1
end

function fcondition(out, X, t, integrator)
    start_distancing(out, X, t, integrator)
    out[3] = peak(X, t, integrator)
    out[4] = peak_pop(X, t, integrator)
end

function track_distancing!(integrator, idx)
    idx <= 2 && start_distancing!(integrator, idx)
    return nothing
end

instantiate_callback(callback::Function) = callback()
instantiate_callback(callback) = callback

function default_callback()
    CallbackSet(
        DiscreteCallback(last_one_standing, terminate!),
        DiscreteCallback(start_distancing_discrete, start_distancing_discrete!),
        DiscreteCallback(stop_distancing_discrete, terminate!),
        coalesced_vector_callback(start_distancing, start_distancing!, nothing, 2; rootfind=RightRootFind),
        coalesced_vector_callback(stop_distancing, nothing, stop_distancing!, 2),
    )
end

function fcallback()
    CallbackSet(
        DiscreteCallback(last_one_standing, terminate!),
        DiscreteCallback(start_distancing_discrete, start_distancing_discrete!),
        coalesced_vector_callback(fcondition, track_distancing!, nothing, 4; rootfind=RightRootFind),
    )
end

# ============================================================================
#   Misc.
# ============================================================================

function min_c(r0)
    Sinf = null_Sinf(r0)
    r0*initial_susceptible*min_adj0(r0)/Sinf * (1.0-Sinf)
end
