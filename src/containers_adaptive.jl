# ============================================================================
#   Bullet Methods
# ============================================================================

function Base.show(io::IO, blt::Bullet)
    prefix = let parts = split(string(Bullet), '.')
        length(parts) >= 2 ? join(parts[end-1:end], '.') : parts[end]
    end
    print(io, "$(prefix)(Sinf=$(blt.Sinf), qinf=$(blt.qinf), μinf=$(blt.μinf), kinf=$(refer_kinf(blt)))")
end

function Base.show(io::IO, ::MIME"text/plain", blt::Bullet)
    Base.show(io, blt)
    silent_fields = (:sol,)
    fields = fieldnames(typeof(blt))
    n_fields = length(fields)
    for (i, field) in enumerate(fields)
        prefix = i == n_fields ? "└─ " : "├─ "
        value = getfield(blt, field)
        if field in silent_fields
            print(io, '\n', "$(prefix)$(field): ...")
        else
            print(io, '\n', "$(prefix)$(field): $(value)")
        end
    end
end

@register function kinf(blt::Bullet)
    refer_kinf(blt)
end

@register function t(blt::Bullet)
    blt.sol.t
end

@register function X(blt::Bullet)
    blt.sol[:, :]
end

@register function J(blt::Bullet)
    # body
end

@register function j1p(blt::Bullet)
    objectives(blt)[1]
end

@register function j1v(blt::Bullet)
    objectives(blt)[2]
end

@register function j1(blt::Bullet)
    blt.j1p + blt.j1v
end

@register function obj(blt::Bullet)
    objectives(blt) |> sum
end

function Commons.sigma(blt::Bullet)
    sum(action(blt); dims=1) |> vec .|> sigma
end

@register_fn sigma Bullet

@register function action(blt::Bullet)
    stack(action.(eachcol(blt.X)))
end

@register function p(blt::Bullet)
    blt.action[1, :]
end

@register function v(blt::Bullet)
    blt.action[2, :]
end

# ============================================================================
#   Record Methods
# ============================================================================

# New normal forward Records have exact k0=0. Retain legacy access for older data.
_uses_forward_diagnostics(rec::Record) = iszero(rec.k0)

function Base.show(io::IO, rec::Record)
    r0 = rec.r0
    c = rec.c
    Sinf = rec.Sinf
    μinf = rec.μinf
    J = rec.obj
    prefix = let parts = split(string(Record), '.')
        length(parts) >= 2 ? join(parts[end-1:end], '.') : parts[end]
    end
    optout_mark = "*# &"[1+rec.branch]
    print(io, f"{prefix}({optout_mark}r0={r0:.3f}, c={c:d}, Sinf={Sinf:3f}, μinf={μinf:.1f}, J={J:.4e})")
end

function Base.show(io::IO, ::MIME"text/plain", rec::Record)
    Base.show(io, rec)
    silent_fields = if length(rec.t) < 5
        (:X,)
    else
        (:t, :X)
    end
    fields = fieldnames(typeof(rec))
    n_fields = length(fields)
    for (i, field) in enumerate(fields)
        prefix = i == n_fields ? "└─ " : "├─ "
        value = getfield(rec, field)
        if field in silent_fields
            print(io, '\n', "$(prefix)$(field): ...")
        else
            print(io, '\n', "$(prefix)$(field): $(value)")
        end
    end
end

#
# Overview
#
@register function J1(rec::Record)
    #rec.j1 * rec.r0 * rec.c  # Rescaled to match with main text
    rec.j1 * rec.c
end

@register function J2(rec::Record)
    #rec.j2 * rec.r0  # Rescaled to match with main text
    rec.j2
end

@register function J(rec::Record)
    rec.J1 + rec.c*rec.J2
end

@register function Js(rec::Record)
    [rec.J1, rec.c*rec.J2]
end

@register function optout(rec::Record)
    rec.j1 < Commons.epsilon
end

@register function inactive(rec::Record)
    rec.j1p < Commons.epsilon
end

@register function sinf(rec::Record)
    rec.Sinf / rec.r0
end

# Infection probability conditional on susceptibility at total investment termination; NaN if no investment.
@register function residual_infection_pr(rec::Record)
    rec.j1 == 0 ? NaN : 1 - rec.Sinf / rec.S[rec.stop]
end

@register function duration(rec::Record)
    tg, tf = rec.tg, rec.tf
    if isnothing(tg)
        0.0
    else
        return tf-tg
    end
end

@register function vduration(rec::Record)
    tg, tf = rec.tgv, rec.tfv
    if isnothing(tg)
        0.0
    else
        return tf-tg
    end
end

@register function pduration(rec::Record)
    tg, tf = rec.tgp, rec.tfp
    if isnothing(tg)
        0.0
    else
        return tf-tg
    end
end

#
# Time-Series
#
@register function S(rec::Record)
    rec.X[1, :]
end

@register function I(rec::Record)
    rec.X[2, :]
end

@register function varphi(rec::Record)
    rec.X[3, :]
end

@register function F(rec::Record)
    rec.X[4, :]
end

@register function varPhi(rec::Record)
    map(rec.X) do X
        X[4]-X[3]*X[5]
    end
end

@register function action(rec::Record)
    rule = _uses_forward_diagnostics(rec) ? Forward.action : action
    stack(rule.(eachcol(rec.X)))
end

@register function p(rec::Record)
    action(rec)[1, :]
end

@register function v(rec::Record)
    action(rec)[2, :]
end

@register function pext(rec::Record)
    clipped_sqrt.(rec.X[4, :])
end

@register function vext(rec::Record)
    clipped_sqrt.(rec.X[3, :])
end

function Commons.sigma(rec::Record)
    p, v = action(rec)
    sigma.(p+v)
end

@register_fn sigma Record

function _sigma_extrema_condition(out, u, t, integrator)
    du = similar(u)
    dXdt(du, u, integrator.p, t)
    out[1] = du[4]
    out[2] = du[3]
    return nothing
end

_sigma_extrema_noop!(integrator, idx) = nothing

function sigma_extrema_callback(; kwargs...)
    VectorContinuousCallback(
        _sigma_extrema_condition,
        _sigma_extrema_noop!,
        _sigma_extrema_noop!,
        2;
        save_positions=(true, true),
        kwargs...,
    )
end

function _processed_time(sol, raw_t::Real, r0::Real)
    raw_t - sol.t[end] + preceding_t(r0, sol[1, end], sol[2, end])
end

@register function minsigmat(rec::Record)
    _uses_forward_diagnostics(rec) && return Temporal.minimum_sigma(rec)
    if length(rec.t) <= 1 || rec.j1 <= Commons.epsilon
        return (sigma=1.0, t=rec.t[1])
    end

    cb = adaptive_callback(; extra_callbacks=sigma_extrema_callback())
    sol = shoot(rec; callback=cb, save_everystep=true).sol

    best_sigma = Inf
    best_t = sol.t[1]
    for (raw_t, u) in zip(sol.t, sol.u)
        p, v = action(u)
        sig = sigma(p + v)
        if isfinite(sig) && sig < best_sigma
            best_sigma = sig
            best_t = raw_t
        end
    end

    return (sigma=best_sigma, t=_processed_time(sol, best_t, rec.target_r0))
end

@register function tX(rec::Record)
    vcat(rec.t', rec.X)
end

@register function rt(rec::Record)
    sig = sigma(rec)
    sig * rec.r0
end

@register function m(rec::Record)
    S = rec.X[1, :]
    I = rec.X[2, :]
    @. -S * exp(-(S + I))
end

#
# Indices
#
@register function start(rec::Record)
    idxv = startv(rec)
    idxp = startp(rec)
    if isnothing(idxv) && isnothing(idxp)
        return nothing
    elseif isnothing(idxv)
        return idxp
    elseif isnothing(idxp)
        return idxv
    else
        min(idxv, idxp)
    end
end

@register function startv(rec::Record)
    findfirst(rec.X[3, :]) do varphi
        varphi >= 1 || isapprox(varphi, 1)
    end
end

@register function startp(rec::Record)
    findfirst(rec.X[4, :]) do F
        F >= 1 || isapprox(F, 1)
    end
end

@register function stop(rec::Record)
    idxv = stopv(rec)
    idxp = stopp(rec)
    if isnothing(idxv) && isnothing(idxp)
        return nothing
    elseif isnothing(idxv)
        return idxp
    elseif isnothing(idxp)
        return idxv
    else
        max(idxv, idxp)
    end
end

@register function stopv(rec::Record)
    findlast(rec.X[3, :]) do varphi
        varphi >= 1 || isapprox(varphi, 1)
    end
end

@register function stopp(rec::Record)
    findlast(rec.X[4, :]) do F
        F >= 1 || isapprox(F, 1)
    end
end

#
# Time Points
#
@register function tg(rec::Record)
    idx = rec.start
    if isnothing(idx) # indicates neither indv. and gov. invests
        return nothing
    else
        rec.t[idx]
    end
end

@register function tgv(rec::Record)
    idx = rec.startv
    if isnothing(idx)
        return nothing
    else
        rec.t[idx]
    end
end

@register function tgp(rec::Record)
    idx = rec.startp
    if isnothing(idx)
        return nothing
    else
        rec.t[idx]
    end
end

@register function tf(rec::Record)
    idx = rec.stop
    if isnothing(idx)
        return nothing
    else
        rec.t[idx]
    end
end

@register function tfv(rec::Record)
    idx = rec.stopv
    if isnothing(idx)
        return nothing
    else
        rec.t[idx]
    end
end

@register function tfp(rec::Record)
    idx = rec.stopp
    if isnothing(idx)
        return nothing
    else
        rec.t[idx]
    end
end

#
# Special Qunatities
#
@register function m0(rec::Record)
    -rec.r0 * exp(-rec.r0)
end

@register function mf(rec::Record)
    Stf = rec.X[1, end]
    Itf = rec.X[2, end]
    -Stf * exp(-(Stf+Itf))
end

#
# State Point
#
@register function adj0(rec::Record)
    (rec.q0, rec.μ0, rec.k0)
end

@register function sp(rec::Record)
    if rec.pidx > 0
        return rec.X[1, rec.pidx] / rec.r0
    else
        return 1/rec.r0
    end
end

@register function ip(rec::Record)
    if rec.pidx > 0
        return rec.X[2, rec.pidx] / rec.r0
    else
        return null_ip(rec.r0)
    end
end

@register function qp(rec::Record)
    if rec.pidx > 0
        q = rec.X[3, rec.pidx] / ip(rec)
        return q / rec.c
    else
        q = rec.Sinf*rec.c/sp(rec)
        return q / rec.c
    end
end

@register function λp(rec::Record)
    if rec.pidx > 0
        λ = rec.X[4, rec.pidx] / ip(rec)
        return λ / rec.c
    else
        λ = rec.Sinf*rec.c/sp(rec)
        return λ / rec.c
    end
end

@register function tp(rec::Record)
    if rec.pidx > 0
        return rec.t[rec.pidx]
    else
        return null_tp(rec.r0)
    end
end

#
# Comparing with null model
#
function Commons.null_Sinf(rec::Record)
    -lambertw(m0(rec), 0)
end

function Commons.null_sinf(rec::Record)
    -lambertw(m0(rec), 0) / rec.r0
end

function Commons.null_Ip(rec::Record)
    null_Ip(rec.r0)
end

function Commons.null_ip(rec::Record)
    null_ip(rec.r0)
end

@register_fn null_Sinf Record
@register_fn null_sinf Record
@register_fn null_Ip Record
@register_fn null_ip Record

#
# Analysis
#
@register function raP(rec::Record)
    apparent_r0_Ip(ip(rec)*rec.r0)
end

@register function rap(rec::Record)
    lambertw((rec.ip-1)*exp(-1), -1)/(rec.ip-1)
end

@register function raT(rec::Record)
    apparent_r0_tot(rec.Sinf)
end

@register function rat(rec::Record)
    log(rec.sinf)/(rec.sinf-1)
end

@register function radp(rec::Record)
    x = 2*rec.ip -1
    lambertw(x*exp(-1), -1)/x
end

@register function sslowdown(rec::Record)
    if length(rec.t) <= 1
        return 1.0
    else
        return duration(rec) / sduration(rec)
    end
end

@register function islowdown(rec::Record)
    if length(rec.t) <= 1
        return 1.0
    else
        return duration(rec) / iduration(rec)
    end
end

@register function rslowdown(rec::Record)
    if length(rec.t) <= 1
        return 1.0
    else
        return duration(rec) / rduration(rec)
    end
end

@register function pslowdown(rec::Record)
    tp(rec) / null_tp(rec.r0)
end

@register function switching(rec::Record)
    Fs = @view rec.X[4, end:-1:1]
    varphis = @view rec.X[3, end:-1:1]
    flag = sign(Fs[1]-varphis[1])
    if flag < 0
        return nothing
    end
    for (i, (F, varphi)) in enumerate(zip(Fs, varphis))
        temp = sign(F-varphi)
        if temp*flag <= 0
            return i
        end
        flag = temp
    end
    return nothing
end

@register function latephase(rec::Record)
    _uses_forward_diagnostics(rec) && return Temporal.measure(rec).late_public_realized
    if (rec.μinf == 0) || (rec.j1p <= 0.0)
        return 0.0
    end
    sol = shoot(rec).sol
    idx = findfirst(sol[4, :] .< Fthr.(sol.u))
    if isnothing(idx)
        return 1.0
    else
        return sol[end, idx] / sol[end, end]
    end
end

function _threshold_crossing_value(
    sol,
    state_index::Int,
    value_index::Int,
    threshold::Real,
)
    values = sol[state_index, :]
    idx = findfirst(x -> x >= threshold, values)
    isnothing(idx) && return nothing
    idx == 1 && return sol[value_index, idx]

    prev = idx - 1
    x0, x1 = values[prev], values[idx]
    y0, y1 = sol[value_index, prev], sol[value_index, idx]

    if isfinite(x0) &&
            isfinite(x1) &&
            isfinite(y0) &&
            isfinite(y1) &&
            x0 != x1 &&
            min(x0, x1) <= threshold <= max(x0, x1)
        weight = (threshold - x0) / (x1 - x0)
        return y0 + weight * (y1 - y0)
    end

    return abs(x0 - threshold) <= abs(x1 - threshold) ? y0 : y1
end

@register function latephasepp(rec::Record)
    _uses_forward_diagnostics(rec) && return Temporal.measure(rec).late_public_incentive
    if (rec.μinf == 0) || (rec.j1p <= 0.0)
        return 0.0
    elseif any(x->x>=1.0, rec.X[3, :])
        sol = shoot(rec; save_everystep=true).sol
        value = _threshold_crossing_value(sol, 3, length(sol.u[1]), 1.0)
        isnothing(value) && return 1.0
        return value / sol[end, end]
    else
        return 1.0
    end
end

# 약간 precision 이 이상함. 다른 callback 과 같이 동작할 때 값이 이상하게 튀는 듯?
function _center_of_mass(out, u, t, intg)
    p, v = action(u, intg.p)
    out[1] = p*u[1]
    out[2] = t*p*u[1]
end

function _dXdt(dX, X, para, t)
    dXdt(dX, X, para, t)
    dX[7] = t*dX[6]
end

function _shoot(Sinf::Real, qinf::Real, μinf::Real;
               callback=adaptive_callback(), save_everystep=false,
               tol=1e-13, _retry_depth=0, alg=Vern9(lazy=false), kwargs...)
    try
        intg = initiate(
            Sinf,
            qinf,
            μinf;
            f=_dXdt,
            u0=(S, q, μ) -> [halting_condition(S, q, μ)..., 0.0],
            callback=callback,
            save_everystep=save_everystep,
            abstol=tol,
            reltol=tol,
            alg=alg,
            initializer=:zeroing,
            kwargs...,
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
            _shoot(Sinf, qinf, μinf; callback=callback, save_everystep=save_everystep, tol=tol, _retry_depth=_retry_depth, kwargs...)
        end
    end
end

@register function latephaseqq(rec::Record)
    _uses_forward_diagnostics(rec) && return Temporal.measure(rec).public_time_fraction
    #intg = IntegrandValuesSum(zeros(2))
    #intgcb = IntegratingSumCallback(_center_of_mass, intg, Float64[0.0, 0.0])
    #cb = CallbackSet(default_callback, intgcb)
    blt = _shoot(rec.Sinf, rec.qinf, rec.μinf)  #; callback=cb)
    new = Record(blt, rec.target_r0, rec.branch)
    #1.0 + (intg.integrand[2]/intg.integrand[1]-1000)/new.duration
    1.0 + (new.X[7, end]/new.X[6, end]-1000)/new.duration
end

@register function earlyphase(rec::Record)
    if _uses_forward_diagnostics(rec)
        m=Temporal.measure(rec)
        return (m.public_total+m.private_total<=1e-13 ? 0. : (m.public_pre+m.private_pre)/(m.public_total+m.private_total))
    end
    pidx = rec.pidx
    if (pidx == 0) || (rec.j1 <= 1e-13)
        return 0.0
    end
    qSp = rec.X[3, pidx] / rec.X[2, pidx] * rec.X[1, pidx]
    qSg = rec.X[3, 1] / rec.X[2, 1] * rec.X[1, 1]
    j1vp = (qSp-qSg) / (rec.r0 * rec.c)
    j1pp = (rec.X[6, pidx]-rec.X[6, 1]) / (rec.r0 * rec.c)
    (j1vp+j1pp) / rec.j1
end

@register function earlyphasep(rec::Record)
    if _uses_forward_diagnostics(rec)
        m=Temporal.measure(rec)
        return (m.public_total<=1e-13 ? 0. : m.public_pre/m.public_total)
    end
    pidx = rec.pidx
    if (pidx == 0) || (rec.j1p <= 1e-13)
        return 0.0
    end
    j1pp = (rec.X[6, pidx]-rec.X[6, 1]) / (rec.r0 * rec.c)
    j1pp / rec.j1p
end

@register function earlyphasev(rec::Record)
    if _uses_forward_diagnostics(rec)
        m=Temporal.measure(rec)
        return (m.private_total<=1e-13 ? 0. : m.private_pre/m.private_total)
    end
    pidx = rec.pidx
    if (pidx == 0) || (rec.j1v <= 1e-13)
        return 0.0
    end
    qSp = rec.X[3, pidx] / rec.X[2, pidx] * rec.X[1, pidx]
    qSg = rec.X[3, 1] / rec.X[2, 1] * rec.X[1, 1]
    j1vp = (qSp-qSg) / (rec.r0 * rec.c)
    j1vp / rec.j1v
end

@register function contribution_p(rec::Record)
    if _uses_forward_diagnostics(rec)
        m=Temporal.measure(rec)
        return (m.pre_zero_denominator ? 0. : 1-m.prepeak_private_share)
    end
    pidx = rec.pidx
    if pidx == 0
        return 0.0
    end
    qSp = rec.X[3, pidx] / rec.X[2, pidx] * rec.X[1, pidx]
    qSg = rec.X[3, 1] / rec.X[2, 1] * rec.X[1, 1]
    j1vp = (qSp-qSg) / (rec.r0 * rec.c)
    j1pp = (rec.X[6, pidx]-rec.X[6, 1]) / (rec.r0 * rec.c)
    if j1vp+j1pp <= 1e-13
        return 0.0
    end
    j1pp / (j1vp + j1pp)
end

@register function contribution_v(rec::Record)
    if _uses_forward_diagnostics(rec)
        m=Temporal.measure(rec)
        return m.prepeak_private_share
    end
    pidx = rec.pidx
    if pidx == 0
        return 0.0
    end
    qSp = rec.X[3, pidx] / rec.X[2, pidx] * rec.X[1, pidx]
    qSg = rec.X[3, 1] / rec.X[2, 1] * rec.X[1, 1]
    j1vp = (qSp-qSg) / (rec.r0 * rec.c)
    j1pp = (rec.X[6, pidx]-rec.X[6, 1]) / (rec.r0 * rec.c)
    if j1vp+j1pp <= 1e-13
        return 0.0
    end
    j1vp / (j1vp + j1pp)
end

@register function sduration(rec::Record)
    if length(rec.t) <= 1
        return 0.0
    else
        Si, Sf = rec.X[1, [1, end]]
        M0 = m0(rec)
        return delta_t(M0, Si, Sf)
    end
end

@register function iduration(rec::Record)
    if length(rec.t) <= 1
        return 0.0
    else
        return recidual_t(rec.X[1:2, 1]...; I1=rec.X[2, end])
    end
end

@register function rduration(rec::Record)
    if length(rec.t) <= 1
        return 0.0
    else
        Si = rec.X[1, 1]
        Sf, If = rec.X[1:2, end]
        M0 = m0(rec)
        null_Sf = -M0*exp(Sf+If)
        return delta_t(M0, Si, null_Sf)
    end
end

@register function fduration(rec::Record)
    if length(rec.t) <= 1
        return null_t1(rec.r0)
    else
        return rec.t[end] +tf2t1(rec)
    end
end

@register function tf2t1(rec::Record)
    if length(rec.t) <= 1
        return 0.0
    else
        return recidual_t(rec.X[1:2, end]...)
    end
end

#
# Reproduce
#
@register function args(rec::Record)
    (rec.Sinf, rec.qinf, rec.μinf)
end

@register function sol(rec::Record)
    (_uses_forward_diagnostics(rec) ? Temporal.reconstruct(rec; save_everystep=true) : shoot(rec; save_everystep=true)).sol
end
