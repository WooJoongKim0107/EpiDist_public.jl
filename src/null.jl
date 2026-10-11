#include("commons.jl")
#using .Commons
using StaticArrays: SA
using Distributed
using OrdinaryDiffEq
using Integrals
using PyFormattedStrings

# ============================================================================
#   Time-Evolution Related
# ============================================================================
function dXdt(X, para, t)
    S, I = X  # S, I ∈ [0, r0]  while  φ >= 1
    dS = -S*I
    dI = +S*I -I
    SA[dS, dI]
end

# ============================================================================
#   Shooting Related -- Callbacks
# ============================================================================
function last_one_standing(X, t, integrator)
    X[2] < infection_floor
end

function peak(X, t, integrator)
    X[1] -1.0
end

const cb1 = DiscreteCallback(last_one_standing, terminate!)
const cb7 = ContinuousCallback(peak, identity)
const fcallback = CallbackSet(cb1, cb7)

# ============================================================================
#   Shooting Related -- Initial Conditions
# ============================================================================
function initial_condition(r0)
    S = r0*initial_susceptible
    I = r0*initial_infected
    SA[S, I]
end

# ============================================================================
#   Shooting-Related -- Shooter (convenient wrapper for error function)
# ============================================================================
function shoot(r0; tmax=1000.0, kwargs...)
    tspan = (0.0, tmax)
    f = ODEFunction(dXdt)
    X0 = initial_condition(r0)
    para = SA[0.0]
    prob = ODEProblem(f, X0, tspan, para)
    solve(prob, Vern9(), abstol=1e-20, reltol=1e-15, callback=fcallback; kwargs...)
end

function ishoot(X0; tmax=1000.0, kwargs...)
    tspan = (0.0, tmax)
    f = ODEFunction(dXdt)
    X0 = SA[X0...]
    para = SA[0.0]
    prob = ODEProblem(f, X0, tspan, para)
    solve(prob, Vern9(), abstol=1e-20, reltol=1e-15, callback=fcallback; kwargs...)
end

function end2end(r0::Real; tmax=1000.0, less=false)
    sol = shoot(r0)
    processing(sol, r0; less=less)
end

end2end(::Type{T}, args...; kwargs...) where {T} = T(end2end(args...; kwargs...))

function processing(sol::ODESolution, r0::Real; less=false)
    if less && length(sol.t) > 1
        pidx = findmax(sol[2, :])[2]
        t = sol.t[[1, pidx, end]]
        X = sol[:, :][:, [1, pidx, end]]  # S, I  ∈ [0, r0]  and  φ >= 1
        pidx = 2
    else
        pidx = findmax(sol[2, :])[2]
        t = sol.t
        X = sol[:, :]  # S, I  ∈ [0, r0]  and  φ >= 1
    end
    obj = null_sinf(r0)
    (; t, X, j1=0, j2=obj, r0, extra=(; pidx))
end

function processing(r0::Real; tmax=1000.0, less=false)
    sol = shoot(r0; tmax=tmax)
    processing(sol, r0; less=less)
end

processing(::Type{T}, args...; kwargs...) where {T} = T(processing(args...; kwargs...))

# ============================================================================
#   Summarize -- wrapper for root-findings of multiple parameters
# ============================================================================
const R0S = 1.1:0.1:11.0
const CS = 1.0:1.0:100.0
