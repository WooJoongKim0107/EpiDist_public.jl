using Distributed: pmap
using Roots: find_zero, ConvergenceFailed
using SciMLBase: DEIntegrator, RightRootFind
using OrdinaryDiffEq: ODEFunction, ODEProblem, ODESolution, solve, Vern9, DiscreteCallback, ContinuousCallback, VectorContinuousCallback, CallbackSet, terminate!
using PyFormattedStrings: @f_str
using Optim: optimize
const zero_tol = pairs((; atol=0.0, rtol=0.0, xatol=0.0, xrtol=0.0))  # for find_zero

# ============================================================================
#   Shooting Interface
# ============================================================================

function shoot(r0::Real, λ0::Real;
               callback=default_callback, save_everystep=false, kwargs...)
    f = ODEFunction(dXdt)
    X0 = triggering_condition(r0, λ0)
    tspan = (below(r0, λ0) ? (0.0, 0.0) : (0.0, 1000.0))
    para = [0]
    prob = ODEProblem(f, X0, tspan, para)
    abstol = fill(1e-20, 4); abstol[4] = 1e-12
    reltol = fill(1e-15, 4); reltol[4] = 1e-10
    sol = solve(prob, Vern9(lazy=true), abstol=abstol, reltol=reltol,
                callback=callback, save_everystep=save_everystep; kwargs...)
    Bullet(r0, λ0, sol)
end

function aim(r0::Real, c::Real, λ0::Real)
    λ0 = (isnothing(λ0) ? 0.0 : max(λ0, 0.0))
    try
        blt = shoot(r0, λ0)
        infer_c(blt) -c
    catch e
        println("R0=$(r0), c=$(c), λ0=$(λ0)")
        rethrow(e)
    end
end

function aimed_shot(r0::Real, c::Real, λ0guess=nothing; kwargs...)
    λ0guess = (isnothing(λ0guess) ? 0.0 : max(λ0guess, 0.0))
    λ0 = find_zero(x->aim(r0, c, x), λ0guess; zero_tol...)
    blt = shoot(r0, λ0; kwargs...)
    Record(blt, c, 1)
end

# ============================================================================
#   Bullet Interface
# ============================================================================

struct Bullet
    r0::Real
    λ0::Real
    sol::ODESolution
end

infer_c(blt::Bullet) = infer_c(blt.sol[:, end])
objectives(blt::Bullet) = objectives(blt.sol, blt.r0, blt.λ0)
aim(blt::Bullet, c::Real) = aim(blt.r0, c, blt.λ0)

# ============================================================================
#   Record Interface
# ============================================================================

struct Record
    branch::Int
    target_c::Real
    cerr::Real

    t::Vector{Real}
    X::Matrix{Real}
    Ip::Real
    pidx::Int            # indicates when infections reach its peak level

    j1::Real
    j2::Real
    obj::Real

    r0::Real
    c::Real
    λ0::Real
end

Record(ntpl::NamedTuple) = Record([ntpl[f] for f in fieldnames(Record)]...)
as_ntuple(obj::T) where {T} = NamedTuple{fieldnames(T)}(getfield(obj, f) for f in fieldnames(T))
as_ntuple(ntpl::NamedTuple) = ntpl

"""Promote *Bullet* into *Record*"""
function Record(blt::Bullet, target_c::Real, branch::Int)
    t, X, pidx, Ip = timeseries_processing(blt.sol, blt.r0)
    j1, j2 = objectives(blt)
    c = infer_c(blt)
    cerr = c-target_c
    ntpl = (; branch, target_c, cerr,
            t, X, Ip, pidx,
            j1, j2, obj=j1+j2,
            r0=blt.r0, c, λ0=blt.λ0)
    Record(ntpl)
end

"""Default Record for (r0, c)"""
function Record(r0::Real, c::Real)
    t = [0.0]
    X = zeros(4, 1)
    X[1, 1] = r0*initial_susceptible
    X[2, 1] = r0*initial_infected
    Ip = null_Ip(r0)
    j2 = 1.0 - null_sinf(r0)
    λinf = c/(1-null_Sinf(r0))
    λ0 = λinf*null_sinf(r0) / initial_susceptible
    ntpl = (; branch=0, target_c=c, cerr=0.0,
            t, X, Ip, pidx=0,
            j1=0.0, j2, obj=j2,
            r0, c, λ0)
    Record(ntpl)
end

# ============================================================================
#   Processing & Interpreting Results
# ============================================================================

function infer_c(Xf)
    mf = -Xf[1] * exp(-Xf[1] - Xf[2])
    if mf < Commons.IE
        throw(DomainError("M/IE-1=$(mf/Commons.IE-1)"))
    end
    Sinf = -lambertw(mf, 0)
    Xf[1]*Xf[3] / (Xf[2]*Sinf) * (1.0-Sinf)
end

function objectives(sol::ODESolution, r0::Real, λ0::Real)
    c = infer_c(sol[:, end])
    j1 = let  # j1 ∈ [0, 1]  where 1 corresponds to the maximum damage (c*r0), i.e. when every individual gets infected
        sol[end, end] / (r0*c)
    end
    j2 = let
        m = -sol[1, end] * exp(-sol[1, end] - sol[2, end])
        sinf = -lambertw(m, 0) / r0  # S(∞) < 1/r0
        1.0 - sinf
    end
    Real[j1, j2]
end

function timeseries_processing(sol::ODESolution, r0::Real)
    Δt = preceding_t(r0, sol[1, 1])
    t = sol.t .+Δt
    X = sol[:, :]
    if sol[1, 1] < 1
        pidx = 0
        Ip = null_Ip(r0)
    else
        Ip, pidx = findmax(X[2, :])
    end
    fewer(t, X, pidx)..., Ip
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
    S, I, varPhi, pS = X  # S, I ∈ [0, r0]  while  Φ >= 1
    p = action(X)
    sig = sigma(p)
    dX[1] = -sig*S*I
    dX[2] = +sig*S*I -I
    dX[3] = (sig*(S+I)-1.0)*varPhi +p*(S+I)
    dX[4] = p*S
end

function triggering_condition(r0, λ0)
    s0 = initial_susceptible
    m = -r0 * s0 * exp(-r0)
    if below(r0, λ0)
        S = -m * exp(1)
        I = 1.0 - S
        λ = λ0*r0*s0/S
        return [S, I, λ*I, 0.0]
    else
        slope = 1 / (r0 * s0 * λ0)
        S, I = given_slope(slope, m, -1)
        return [S, I, 1.0, 0.0]
    end
end

function below(r0, λ0)
    λ0 <= min_adj0(r0)
end

function action(X)
    clipped_sqrt(X[3])
end

# ============================================================================
#   Callback Internals
# ============================================================================

function last_one_standing(X, t, integrator)
    X[2] < infection_floor
end

function distancing(X, t, integrator)
    X[3] - 1.0
end

function peak(X, t, integrator)
    v = action(X)
    sig = sigma(v)
    sig*X[1] -1.0
end

const cb1 = DiscreteCallback(last_one_standing, terminate!)
const cb2 = ContinuousCallback(distancing, nothing, terminate!, rootfind=RightRootFind)  # RightRootFind to ensure c=0 at t=0
const cb7 = ContinuousCallback(peak, identity)
const default_callback = CallbackSet(cb1, cb2, cb7)
const fcallback = CallbackSet(cb1, cb7)

# ============================================================================
#   Misc.
# ============================================================================

function min_c(r0)
    Sinf = null_Sinf(r0)
    r0*initial_susceptible*min_adj0(r0)/Sinf * (1.0-Sinf)
end

function max_Sinf(c)
    a = 1/(1-c)
    b = c*a
    a - Commons.lambertw(b*exp(b), 0)
end
