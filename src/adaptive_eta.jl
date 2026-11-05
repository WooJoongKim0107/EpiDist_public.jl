"""Nash response to a scaled, preannounced public schedule. Julia only."""
module Eta
using LambertW, QuadGK, SparseArrays, LinearAlgebra
import ..Forward
import ...Commons
import ..Record
const AF = Forward
const CM = Commons
export eta_source, eta_response, eta_sweep, eta_timeseries

final_size(S, I, r) = -real(lambertw(-r * S * exp(-r * (S + I)))) / r

"""
    eta_source(rec; cutoff=:private, horizon=nothing)

Prepare the original AO public schedule with its physical time origin.
The default cutoff is the last positive private-control interval, or zero
if the original policy has no private effort. With cutoff zero, eta scales
the entire public schedule. No public reoptimization is performed.
"""
function eta_source(rec::Record; cutoff=:private, horizon=nothing)
    rec.c > 0 || throw(ArgumentError("infection cost must be positive"))
    iszero(rec.k0) || throw(ArgumentError("eta_source requires a forward AO Record (k0=0)"))
    b = AF.shoot(rec; tol=1e-13, save_everystep=true)
    full = AF.lshoot(rec; tol=1e-13, tmax=2000.0)
    idx = findfirst(u -> u[1] == b.sol.u[1][1] && u[2] == b.sol.u[1][2], full.sol.u)
    isnothing(idx) && error("Could not match the physical-time prefix")
    shift = full.sol.t[idx] - first(b.sol.t)
    ts = sort(unique(b.sol.t))
    private_end = 0.0
    public_seen = false
    for (lo, hi) in zip(ts[1:end-1], ts[2:end])
        p, v = AF.action(b.sol((lo + hi) / 2))
        v > 0 && (private_end = hi + shift)
        public_seen |= p > 0
    end
    cut = cutoff === :private ? private_end : Float64(cutoff)
    cut >= 0 || throw(ArgumentError("cutoff must be nonnegative"))
    total_end = rec.j1 == 0 ? 0.0 : last(b.sol.t) + shift
    peak = full.sol.t[argmax([u[2] for u in full.sol.u])]
    T =
        isnothing(horizon) ?
        min(max(total_end, peak) + max(35.0, 8 / (1 - rec.Sinf)), last(full.sol.t)) :
        Float64(horizon)
    T > max(total_end, cut) ||
        throw(ArgumentError("horizon must exceed policy termination and cutoff"))
    T <= last(full.sol.t) || error("Requested horizon exceeds the reconstructed source")
    split = cut > 0 ? cut : T / 2
    public(t) =
        !public_seen || t < first(b.sol.t) + shift || t > total_end ? 0.0 :
        AF.action(b.sol(t - shift))[1]
    (;
        rec,
        r0=Float64(rec.r0),
        c=Float64(rec.c),
        cutoff=cut,
        split,
        T,
        total_end,
        public,
        full,
        public_seen,
        private_seen=private_end > 0,
    )
end

function policy(src, eta, t, k)
    lo = k == 1 ? 0.0 : src.split
    hi = k == 1 ? src.split : src.T
    te = clamp(t, lo + 1e-9, hi - 1e-9) # one-sided controls at the phase boundary
    src.public(te) * (src.cutoff == 0 || k == 2 ? eta : 1.0)
end

function rhs(u, src, eta, t, k)
    S, I = exp.(clamp.(u[1:2], -700.0, 5.0))
    z = u[3]
    p = policy(src, eta, t, k)
    v = max(sqrt(max(z * src.r0 * src.c * I, zero(z))) - 1 - p, zero(z))
    rate = src.r0 / (1 + p + v)
    [-rate * I, rate * S - 1, rate * I * z + v / src.c]
end

function original_seed(src, x)
    parts = map(1:2) do k
        L = k == 1 ? src.split : src.T - src.split
        t = (k == 1 ? 0.0 : src.split) + L * x
        u = src.full.sol(t)
        S, I = u[1:2] ./ src.r0
        z = t > src.total_end ? src.rec.Sinf / u[1] : u[3] / u[2] / src.c
        [log(S), log(I), z]
    end
    vcat(parts...)
end

function boundary!(res, ua, ub, src)
    res[1] = ua[1] - log(CM.initial_susceptible)
    res[2] = ua[2] - log(CM.initial_infected)
    res[3:5] .= ua[4:6] .- ub[1:3]
    S, I = exp.(ub[4:5])
    res[6] = ub[6] - final_size(S, I, src.r0) / S
    nothing
end

include("adaptive_eta_collocation.jl")

"""
    eta_response(source, eta; initial=nothing, tol=1e-10, intervals=600, max_step=.15)

Solve the follower Nash BVP for this fixed public policy. `initial` can be a
previous result for continuation. S and I are population fractions; z=q/c.
The no-control tail is included through an exact final-size boundary.
`tol` controls the Newton defect; `max_step` bounds the physical mesh step.
Use a smaller max_step to check discretization error.
"""
function eta_response(src, eta; initial=nothing, tol=1e-10, intervals=600, max_step=0.15)
    eta >= 0 && isfinite(eta) || throw(ArgumentError("eta must be finite and nonnegative"))
    sol = solve_collocation(src, eta, initial; tol, intervals, max_step)
    res = zeros(6)
    boundary!(res, sol(0.0), sol(1.0), src)
    maximum(abs, res) <= max(tol, 1e-8) || error("Nash boundary mismatch")
    ST, IT = exp.(sol(1.0)[4:5])
    si = final_size(ST, IT, src.r0)
    s, i = src.r0 * ST, src.r0 * IT
    phi = src.r0 * src.c * si * IT / ST
    if s + i > 1
        sx = exp(1 - s - i + log(s))
        src.r0 * si < sx < s && (phi = max(phi, src.c * src.r0 * si * (1 - sx) / sx))
    end
    phi <= 1 + 10tol ||
        error("Private investment reactivates after the chosen horizon; increase horizon")
    all(u -> u[3] > 0 && u[6] > 0, eachcol(sol.U)) || error("Nonpositive costate")
    out = (;
        eta=Float64(eta),
        source=src,
        sol,
        Sinf=si,
        boundary_error=maximum(abs, res),
        tail_max_phi=phi,
        defect=sol.defect,
    )
    integrals = map(1:2) do k
        lo = k == 1 ? 0.0 : src.split
        L = k == 1 ? src.split : src.T - src.split
        function spending(x)
            u = sol(x)[3k-2:3k]
            S, I = exp.(u[1:2])
            p = policy(src, eta, lo + L * x, k)
            v = max(sqrt(max(0.0, src.r0 * src.c * u[3] * I)) - 1 - p, 0.0)
            L .* [v * S, p * S, v]
        end
        first(quadgk(spending, range(0, 1; length=21)...; rtol=1e-8, atol=1e-10))
    end
    total = sum(integrals)
    early = src.cutoff == 0 ? zeros(3) : integrals[1]
    metrics = (; V=total[1], P=total[2], V_early=early[1], U_early=early[3])
    merge(out, (; metrics))
end

"""Sample physical-time investment rates and epidemic states from a solved result."""
function eta_timeseries(result; n=1201)
    src = result.source
    t = sort(unique(vcat(range(0, src.T; length=n), [src.split, src.cutoff, src.total_end])))
    S = Float64[]
    I = Float64[]
    p = Float64[]
    v = Float64[]
    for time in t
        k = time < src.split ? 1 : 2
        lo = k == 1 ? 0.0 : src.split
        L = k == 1 ? src.split : src.T - src.split
        u = result.sol((time - lo) / L)[3k-2:3k]
        s, i = exp.(u[1:2])
        pp = policy(src, result.eta, time, k)
        vv = max(sqrt(max(0.0, src.r0 * src.c * u[3] * i)) - 1 - pp, 0.0)
        push!(S, s)
        push!(I, i)
        push!(p, pp)
        push!(v, vv)
    end
    (; t, S, I, p, v, a=p .+ v)
end

"""
    eta_sweep(rec; etas=0:0.05:2, kwargs...)

Return results in sorted eta order, continuing downward and upward from the
original AO (eta=1). Pass `source=eta_source(rec; ...)` to change cutoff/horizon.
"""
function eta_sweep(rec::Record; etas=0:0.05:2, source=eta_source(rec), kwargs...)
    values = sort(unique(Float64.(collect(etas))))
    isempty(values) && return []
    all(x -> isfinite(x) && x >= 0, values) ||
        throw(ArgumentError("etas must be finite and nonnegative"))
    base = eta_response(source, 1.0; kwargs...)
    results = Dict{Float64,Any}(1.0 => base)
    !source.public_seen && return [merge(base, (; eta)) for eta in values]
    for branch in (reverse(filter(<(1.0), values)), filter(>(1.0), values))
        previous = base
        for eta in branch
            previous = eta_response(source, eta; initial=previous, kwargs...)
            results[eta] = previous
        end
    end
    [results[eta] for eta in values]
end
end
