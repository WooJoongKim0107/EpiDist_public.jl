module Commons
export lambertw, initial_susceptible, initial_infected, infection_floor,
    given_slope, null_sinf, null_Sinf, null_ip, null_Ip, min_adj0, clipped_sqrt, sigma,
    apparent_r0_Ip, apparent_r0_tot, null_tp, null_t1, residual_t, preceding_t, delta_t,
    coalesced_vector_callback

using LambertW: lambertw
using Integrals: IntegralProblem, QuadGKJL, solve
using OrdinaryDiffEq: VectorContinuousCallback
using SciMLBase: u_modified!

const IE = -exp(-1)
const epsilon = eps(1.0)
const initial_infected = 1e-13
const initial_susceptible = 1-initial_infected
const infection_floor = initial_infected / 10

function given_slope(slope::Real, m::Real, branch=0)
    arg = m * (1 + slope)
    if arg < IE
        s = -lambertw(IE, branch) * m/IE
        i = IE/m*s -s
    else
        s = -lambertw(arg, branch) / (1+slope)
        i = slope*s
    end
    [s, i]
end

"""s(∞) ∈ [0, 1]"""
function null_sinf(r0::Real)
    return null_Sinf(r0) / r0
end

"""S(∞) ∈ [0, R₀]"""
function null_Sinf(r0::Real)
    m = -r0 * initial_susceptible * exp(-r0)
    return -lambertw(m, 0)
end

function null_ip(r0::Real)
    return null_Ip(r0) / r0
end

function null_Ip(r0::Real)
    m0 = -r0 * initial_susceptible * exp(-r0)
    -log(-m0)-1
end

function min_adj0(r0::Real)
    1 / (exp(r0 - 1) - r0*initial_susceptible)
end

function clipped_sqrt(x::Real)
    if x <= 1.0
        return 0.0
    else
        return sqrt(x) - 1.0
    end
end

function clipped_log(x::Real)
    if x <= 1.0
        return 0.0
    else
        return log(x) - 1.0
    end
end

function sigma(a::Real)
    1.0 / (1.0 + a)
    # exp(-a)
end

"""
Ip = -log(-m0*exp(1))
-m0*exp(1) = exp(-Ip)
-m0 = exp(-Ip-1)
-r0*(1-I0)*exp(-r0) = -exp(-Ip-1)
"""
function apparent_r0_Ip(Ip::Real)
    -lambertw(-exp(-1-Ip)/initial_susceptible, -1)
end

"""
Sinf = -lambertw(m, 0)
-Sinf*exp(-Sinf) = -r0*(1-I0)*exp(-r0)
"""
function apparent_r0_tot(Sinf::Real)
    -lambertw(-Sinf*exp(-Sinf)/initial_susceptible, -1)
end

function null_tp(r0::Real)
    S0 = r0*initial_susceptible
    m0 = -S0*exp(-r0)
    delta_t(m0, S0, 1)
end

function null_t1(r0::Real; I0=initial_infected::Real, I1=infection_floor::Real)
    m0 = -r0 * (1 - I0) * exp(-r0)
    S0 = -lambertw(m0*exp(I0), -1)
    S1 = -lambertw(m0*exp(I1), 0)
    delta_t(m0, S0, S1)
end

function null_tx(r0::Real; I0=initial_infected::Real)
    m0 = -r0 * (1 - I0) * exp(-r0)
    S0 = -Commons.lambertw(m0*exp(I0), -1)
    Sx = -m0*exp(1)
    Commons.delta_t(m0, S0, Sx)
end

function preceding_t(r0::Real, S::Real)
    S0 = r0*initial_susceptible
    m0 = -S0*exp(-r0)
    delta_t(m0, S0, S)
end

function residual_t(Sf::Real, If::Real, I1::Real)
    mf = -Sf*exp(-Sf-If)
    S1 = -lambertw(mf*exp(I1), 0)
    delta_t(mf, Sf, S1)
end

"""
Calculate time duration between
  S=Sstart -> S=Sstop
"""
function delta_t(m0::Real, Sstart::Real, Sstop::Real)
    integrand(x, p) = inv(x*(log(-p[1]) -log(x) +x))
    prob = IntegralProblem(integrand, (Sstart, Sstop), [m0])
    solve(prob, QuadGKJL()).u
end

include("coalescing_callback.jl")

end
