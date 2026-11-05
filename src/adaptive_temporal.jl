"""Forward-trajectory temporal diagnostics, shared by Record properties and batch analysis."""
module Temporal
import ..Forward
import ...Commons
using LinearAlgebra, Roots
const AF = Forward
const A = parentmodule(@__MODULE__)
const C = Commons
export measure, measure_bullet, minimum_sigma, reconstruct, clear_cache!, cache_info
const VERSION = "corrected-forward-temporal-v2"
const GL = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()
const NULLTP = Dict{Float64,Float64}()
const AUX_LOCK = ReentrantLock()
function gauss(n)
    lock(AUX_LOCK) do
        get!(GL, n) do
            off = [k / sqrt(4k * k - 1) for k = 1:n-1]
            e = eigen(SymTridiagonal(zeros(n), off))
            (e.values, 2 .* e.vectors[1, :] .^ 2)
        end
    end
end

function distinct_times(xs)
    vals = sort(unique(Float64.(xs)))
    out = Float64[]
    for t in vals
        isempty(out) || t - out[end] > 32eps(max(1.0, abs(t))) || continue
        push!(out, t)
    end
    out
end

function roots_on_interval(f, left, right; n=4)
    right > left || return Float64[]
    ts = collect(range(left, right; length=n + 1))
    ys = f.(ts)
    roots = Float64[]
    for (t, y) in zip(ts, ys)
        isfinite(y) && y == 0 && push!(roots, t)
    end
    for i = 1:length(ts)-1
        a, b = ys[i:i+1]
        isfinite(a) && isfinite(b) && a != 0 && b != 0 && signbit(a) != signbit(b) || continue
        root = find_zero(f, (ts[i], ts[i+1]), Bisection(); xatol=0.0, xrtol=0.0, atol=0.0, rtol=0.0)
        push!(roots, Float64(root))
    end
    distinct_times(roots)
end

function peak_and_incentive(b; root_scan=4)
    sol = b.sol
    ts = distinct_times(sol.t)
    start, stop = first(ts), last(ts)
    peak_times = copy(ts)
    inc_roots = Float64[]
    # All control switches and effective-reproduction crossings are saved by the forward callback.
    # Continuous completed-solution derivatives refine stationary-I points; no in-step integrator interpolation.
    gi(t) = sol(t, Val{1})[2]
    ph(t) = sol(t)[3] - 1
    for j = 1:length(ts)-1
        l, r = ts[j:j+1]
        append!(peak_times, roots_on_interval(gi, l, r; n=root_scan))
        append!(inc_roots, roots_on_interval(ph, l, r; n=root_scan))
    end
    peak_times = distinct_times(peak_times)
    candidates = [(time=t, height=Float64(sol(t)[2]), kind="active") for t in peak_times]
    u0 = sol.u[1]
    uf = sol.u[end]
    if start > 0 && u0[1] <= 1
        tp = lock(AUX_LOCK) do
            get!(NULLTP, b.r0) do
                Float64(C.null_tp(b.r0))
            end
        end
        push!(candidates, (time=tp, height=Float64(C.null_Ip(b.r0)), kind="before_investment"))
    end
    if uf[1] > 1
        m = -uf[1] * exp(-uf[1] - uf[2])
        height = -log(-m) - 1
        tp = stop + C.delta_t(m, uf[1], 1.0)
        push!(candidates, (time=Float64(tp), height=Float64(height), kind="after_investment"))
    end
    best = candidates[argmax([x.height for x in candidates])]
    # Earliest-time tie convention only for representationally equal maxima.
    tied = filter(x -> abs(x.height - best.height) <= 8eps(max(1.0, abs(best.height))), candidates)
    best = tied[argmin([x.time for x in tied])]
    inc_cuts = distinct_times(vcat(ts, inc_roots))
    tau_inc = 0.0
    inc_seen = false
    for j = 1:length(inc_cuts)-1
        l, r = inc_cuts[j:j+1]
        if ph((l + r) / 2) > 0
            inc_seen = true
            tau_inc = r
        end
    end
    # Analytic zero-distancing continuation handles any final roundoff-scale contact.
    tail_max_phi = uf[3]
    tail_max_F = uf[4]
    if uf[1] + uf[2] > 1
        invariant = uf[1] + uf[2] - log(uf[1])
        sx = exp(1 - invariant)
        if 0 < sx < uf[1]
            factor = (1 - sx) / sx / (uf[2] / uf[1])
            tail_max_phi = max(tail_max_phi, uf[3] * factor)
            tail_max_F = max(tail_max_F, uf[4] * factor)
        end
    end
    max(tail_max_phi, tail_max_F) <= 1 + 1e-8 || error("Zero-continuation activation check failed")
    if tail_max_phi > 1 + 1e-11
        # Public investment is zero throughout this continuation; its contribution to late spending is zero.
        si = AF._final_size(uf[1], uf[2])
        m = -uf[1] * exp(-uf[1] - uf[2])
        Q = uf[3] * uf[1] / uf[2]
        phi_of_s(s) = Q * (log(s / (-m)) - s) / s
        right = uf[3] > 1 ? uf[1] : exp(1 - (uf[1] + uf[2] - log(uf[1])))
        scut = find_zero(
            s -> phi_of_s(s) - 1,
            (nextfloat(si), right),
            Bisection();
            xatol=0.0,
            xrtol=0.0,
            atol=0.0,
            rtol=0.0,
        )
        tau_inc = stop + C.delta_t(m, uf[1], scut)
        inc_seen = true
    end
    best, Float64(tau_inc), inc_seen, ts, tail_max_phi, tail_max_F
end

function measure_bullet(b, c; quad_order=8, root_scan=4)
    b.status in (:terminated, :no_distancing) || error("Unfinished forward trajectory $(b.status)")
    rr = AF.aim(b, c)
    all(isfinite, rr) && maximum(abs, rr) <= 1e-7 || error("Endpoint residual mismatch $rr")
    peak, tau_inc, inc_seen, ts, tail_phi, tail_F = peak_and_incentive(b; root_scan)
    nodes, weights = gauss(quad_order)
    tau_real = 0.0
    private_seen = false
    public_seen = false
    first_investment = final_investment = public_end = NaN
    for j = 1:length(ts)-1
        l, r = ts[j:j+1]
        mid = (l + r) / 2
        p, v = AF.action(b.sol(mid))
        if v > 0
            private_seen = true
            tau_real = r
        end
        if p > 0
            public_seen = true
            public_end = r
        end
        if p > 0 || v > 0
            isnan(first_investment) && (first_investment = l)
            final_investment = r
        end
    end
    cuts = copy(ts)
    for t in (peak.time, tau_real, tau_inc)
        first(ts) < t < last(ts) && push!(cuts, t)
    end
    cuts = distinct_times(cuts)
    P = V = Ppre = Vpre = Plate = Pinc = Ptime = 0.0
    for j = 1:length(cuts)-1
        l, r = cuts[j:j+1]
        mid = (l + r) / 2
        half = (r - l) / 2
        pp = vv = 0.0
        for (node, w) in zip(nodes, weights)
            u = b.sol(mid + half * node)
            p, v = AF.action(u)
            p >= 0 && v >= 0 || error("Negative investment")
            pp += half * w * p * u[1]
            vv += half * w * v * u[1]
            Ptime += half * w * p * u[1] * (mid + half * node)
        end
        P += pp
        V += vv
        mid < peak.time && (Ppre += pp; Vpre += vv)
        mid >= tau_real && (Plate += pp)
        mid >= tau_inc && (Pinc += pp)
    end
    scale = b.r0 * c
    pre = (Ppre + Vpre) / scale
    pre_zero = pre <= 1e-13
    early = pre_zero ? 0.0 : Vpre / (Ppre + Vpre)
    late = P == 0 ? 0.0 : Plate / P
    conservative = P == 0 ? 0.0 : Pinc / P
    all(x -> isfinite(x) && 0 <= x <= 1, (early, late, conservative)) ||
        error("Ratio outside [0,1]")
    jp, jv, j2 = AF.objectives(b, c)
    perr = P / scale - jp
    verr = V / scale - jv
    max(abs(perr), abs(verr)) <= 1e-7 || error("Spending quadrature mismatch $perr $verr")
    rec = AF.Record(b, c)
    legacy_pre = _stored_prepeak_share(rec)
    late_state =
        jp <= 0 ? 0.0 :
        tau_real <= first(ts) ? 1.0 :
        tau_real >= last(ts) ? 0.0 : (b.sol.u[end][6] - b.sol(tau_real)[6]) / b.sol.u[end][6]
    # Spending quantiles share this ordinary trajectory and its existing time origin.
    # The fixed normalized-cost cutoff only suppresses unresolved quantiles, not events.
    private_status = jv == 0 ? :zero : jv <= 1e-8 ? :below_resolution : :resolved
    public_status = jp == 0 ? :zero : jp <= 1e-8 ? :below_resolution : :resolved
    private_status == :resolved &&
        !private_seen &&
        error("Resolved private spending has no positive control interval")
    public_status == :resolved &&
        !public_seen &&
        error("Resolved public spending has no positive control interval")
    u0 = b.sol.u[1]
    initial_private = u0[3] * u0[1] / u0[2]
    private, ev =
        private_status == :resolved ? _expenditure_quantiles(b.sol, 3, initial_private) :
        ((NaN, NaN, NaN), 0.0)
    public, ep =
        public_status == :resolved ? _expenditure_quantiles(b.sol, 6, initial_private) :
        ((NaN, NaN, NaN), 0.0)
    (
        prepeak_private_share=early,
        late_public_realized=late,
        late_public_incentive=conservative,
        peak_time=peak.time,
        peak_height=peak.height / b.r0,
        peak_location=peak.kind,
        tau_realized=tau_real,
        tau_incentive=tau_inc,
        private_seen,
        inc_seen,
        public_total=P / scale,
        private_total=V / scale,
        public_pre=Ppre / scale,
        private_pre=Vpre / scale,
        public_after_realized=Plate / scale,
        public_after_incentive=Pinc / scale,
        pre_zero_denominator=pre_zero,
        public_zero_denominator=P == 0,
        public_quad_error=perr,
        private_quad_error=verr,
        legacy_pre,
        legacy_pre_difference=early - legacy_pre,
        public_mean_time=P == 0 ? NaN : Ptime / P,
        public_time_fraction=P == 0 || last(ts) == first(ts) ? 0.0 :
                             clamp((Ptime / P - first(ts)) / (last(ts) - first(ts)), 0.0, 1.0),
        late_state_difference=late - late_state,
        nash_residual=Float64(rr[1]),
        leader_residual=Float64(rr[2]),
        start_time=first(ts),
        stop_time=last(ts),
        ode_nodes=length(ts),
        root_scan,
        quad_order,
        tail_max_phi=tail_phi,
        tail_max_F=tail_F,
        r0=Float64(b.r0),
        c=Float64(c),
        private10=private[1],
        private50=private[2],
        private90=private[3],
        public10=public[1],
        public50=public[2],
        public90=public[3],
        first_investment=Float64(first_investment),
        final_investment=Float64(final_investment),
        private_end=private_seen ? Float64(tau_real) : NaN,
        public_end=Float64(public_end),
        active=private_seen || public_seen,
        public_seen,
        private_status,
        public_status,
        quantile_error=max(ev, ep),
    )
end

function _stored_prepeak_share(rec)
    j = rec.pidx
    j == 0 && return 0.0
    X = rec.X
    scale = rec.r0 * rec.c
    v = (X[3, j] * X[1, j] / X[2, j] - X[3, 1] * X[1, 1] / X[2, 1]) / scale
    p = (X[6, j] - X[6, 1]) / scale
    v + p <= 1e-13 ? 0.0 : Float64(v / (v + p))
end

# Cache only scalar diagnostics, never Records or dense solutions.
const CACHE_LIMIT = 256
const CACHE = Dict{NTuple{10,Float64},Any}()
const CACHE_LOCK = ReentrantLock()
const CACHE_HITS = Ref(0)
const CACHE_MISSES = Ref(0)
function clear_cache!()
    lock(CACHE_LOCK) do
        empty!(CACHE)
        CACHE_HITS[] = 0
        CACHE_MISSES[] = 0
    end
    nothing
end
cache_info() =
    lock(CACHE_LOCK) do
        (entries=length(CACHE), limit=CACHE_LIMIT, hits=CACHE_HITS[], misses=CACHE_MISSES[])
    end

"""Reproduce a normal Record with the forward solver and verify its endpoints and objective."""
function reconstruct(
    rec::A.Record;
    long=false,
    tol=1e-13,
    save_everystep=true,
    maxiters=500000,
    tmax=long ? max(1000.0, 100 / max(1 - Float64(rec.Sinf), eps(Float64))) : 1000.0,
)
    iszero(rec.k0) || throw(ArgumentError("Forward reconstruction requires k0=0"))
    b = (long ? AF.lshoot : AF.shoot)(rec; tol, save_everystep, maxiters, tmax)
    residual = AF.aim(b, rec.c)
    all(isfinite, residual) && maximum(abs, residual) <= 1e-7 ||
        error("Record endpoint conditions were not reproduced")
    abs(sum(AF.objectives(b, rec.c)) - rec.obj) <= 1e-7 ||
        error("Record objective was not reproduced")
    b
end

function _expenditure_quantiles(sol, index, initial_private)
    accumulated(t) =
        if index == 6
            sol(t)[6]
        else
            u = sol(t)
            u[3] * u[1] / u[2] - initial_private
        end
    total = accumulated(last(sol.t))
    isfinite(total) && total > 0 || error("Positive spending could not be reconstructed")
    fractions = (0.1, 0.5, 0.9)
    times = map(fractions) do fraction
        Float64(
            find_zero(
                t -> accumulated(t) - fraction * total,
                extrema(sol.t),
                Bisection();
                xatol=1e-10,
                xrtol=1e-12,
            ),
        )
    end
    err = maximum(abs(accumulated(t) / total - q) for (t, q) in zip(times, fractions))
    issorted(times) && err < 1e-7 || error("Expenditure quantile mismatch: $err")
    times, Float64(err)
end

"""
    measure(rec; tol=1e-13, quad_order=8, root_scan=4, cache=true, maxiters=500000)

Reintegrate a normal forward Record and measure all temporal diagnostics together.
`prepeak_private_share`, `late_public_realized`, and `late_public_incentive` use
quadrature on the completed solution, split at control switches, the infection peak,
and the two private cutoffs. Undefined ratios return 0 for the existing property
interface; `pre_zero_denominator` and `public_zero_denominator` identify them.
The bounded cache retains scalar results only. Use `cache=false` or `clear_cache!()`
for a fresh solve; custom numerical settings have separate cache entries.

The same trajectory also supplies private/public 10%, 50%, 90% cumulative spending
times and first/final investment events. All times retain the ordinary forward
time convention used by the existing metrics, including `peak_time` and
`start_time`/`stop_time`. Sector costs <=1e-8 have unresolved (NaN) quantiles;
absent control events are NaN. Existing `tau_realized` keeps its zero convention.
No long trajectory or separate timing cache is constructed.
"""
function measure(rec::A.Record; tol=1e-13, quad_order=8, root_scan=4, cache=true, maxiters=500000)
    iszero(rec.k0) ||
        throw(ArgumentError("Temporal.measure requires a normal forward Record with k0=0"))
    tol > 0 && quad_order >= 1 && root_scan >= 1 && maxiters >= 1 ||
        throw(ArgumentError("Invalid diagnostic settings"))
    key = (
        Float64(rec.r0),
        Float64(rec.c),
        Float64(rec.q0),
        Float64(rec.μ0),
        Float64(rec.k0),
        Float64(rec.obj),
        Float64(tol),
        Float64(quad_order),
        Float64(root_scan),
        Float64(maxiters),
    )
    if cache
        found = lock(CACHE_LOCK) do
            if haskey(CACHE, key)
                CACHE_HITS[] += 1
                return CACHE[key]
            end
            CACHE_MISSES[] += 1
            nothing
        end
        isnothing(found) || return found
    end
    b = reconstruct(rec; tol, save_everystep=true, maxiters)
    m = measure_bullet(b, rec.c; quad_order, root_scan)
    abs(sum(AF.objectives(b, rec.c)) - rec.obj) <= 1e-7 ||
        error("Record objective was not reproduced")
    if cache
        lock(CACHE_LOCK) do
            length(CACHE) >= CACHE_LIMIT &&
                !haskey(CACHE, key) &&
                delete!(CACHE, first(keys(CACHE)))
            CACHE[key] = m
        end
    end
    m
end

"""Minimum transmission multiplier and its physical time on a forward trajectory."""
function minimum_sigma(rec::A.Record; tol=1e-13, root_scan=4)
    b = reconstruct(rec; tol, save_everystep=true)
    ts = distinct_times(b.sol.t)
    candidates = copy(ts)
    for j = 1:length(ts)-1, i in (3, 4)
        append!(
            candidates,
            roots_on_interval(t -> b.sol(t, Val{1})[i], ts[j], ts[j+1]; n=root_scan),
        )
    end
    best = 1.0
    best_t = first(ts)
    for t in distinct_times(candidates)
        u = b.sol(t)
        p, v = AF.action(u)
        sig = 1 / (1 + p + v)
        # At a saved switching contact, retain both one-sided control limits.
        if u[5] < 0 &&
           u[3] > 1 &&
           abs(u[4] - AF._threshold(u[3], u[5])) <= 1e-10 * max(1.0, abs(u[4]))
            sig = min(sig, 1 / sqrt(max(u[3], u[4])))
        end
        if sig < best
            best = sig
            best_t = t
        end
    end
    (sigma=best, t=best_t)
end
end
