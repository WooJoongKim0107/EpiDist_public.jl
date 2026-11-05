#include("commons.jl")
#using .Commons
using PropertyArrays
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
