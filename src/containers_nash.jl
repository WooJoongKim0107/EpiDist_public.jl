# ============================================================================
#   Bullet Methods
# ============================================================================

function Base.show(io::IO, blt::Bullet)
    prefix = let parts = split(string(Bullet), '.')
        length(parts) >= 2 ? join(parts[end-1:end], '.') : parts[end]
    end
    print(io, "$(prefix)(r0=$(blt.r0), q0=$(blt.q0))")
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

@register function t(blt::Bullet)
    blt.sol.t
end

@register function X(blt::Bullet)
    blt.sol[:, :]
end

@register function start(blt::Bullet)
    findfirst(blt.X[3, :] .>= 1)
end

@register function stop(blt::Bullet)
    findlast(blt.X[3, :]) do varphi
        varphi >= 1 || isapprox(varphi, 1)
    end
end

@register function tg(blt::Bullet)
    if isnothing(blt.start)
        return nothing
    else
        blt.t[blt.start]
    end
end

@register function tf(blt::Bullet)
    if isnothing(blt.stop)
        return nothing
    else
        blt.t[blt.stop]
    end
end

@register function Sinf(blt::Bullet)
    Sf, If = blt.X[[1, 2], end]
    mf = -Sf * exp(-(Sf+If))
    -lambertw(mf, 0)
end

# ============================================================================
#   Record Methods
# ============================================================================

function Base.show(io::IO, rec::Record)
    r0 = rec.r0
    c = rec.c
    J = rec.obj
    prefix = let parts = split(string(Record), '.')
        length(parts) >= 2 ? join(parts[end-1:end], '.') : parts[end]
    end
    optout_mark = "* "[1+rec.branch]
    print(io, f"{prefix}({optout_mark}r0={r0:.3f}, c={c:d}, J={J:.4e})")
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

@register function optout(rec::Record)
    rec.j1 < Commons.epsilon
end

@register function Sinf(rec::Record)
    -lambertw(rec.mf, 0)
end

@register function sinf(rec::Record)
    rec.Sinf / rec.r0
end

@register function duration(rec::Record)
    tg, tf = rec.tg, rec.tf
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

@register function q(rec::Record)
    rec.X[3, :] ./ rec.X[2, :]
end

@register function action(rec::Record)
    action.(eachcol(rec.X))
end

@register function v(rec::Record)
    action.(eachcol(rec.X))
end

function Commons.sigma(rec::Record)
    action(rec) .|> sigma
end

@register_fn sigma Record

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
    findfirst(rec.X[3, :] .>= 1)
end

@register function stop(rec::Record)
    findlast(rec.X[3, :]) do varphi
        varphi >= 1 || isapprox(varphi, 1)
    end
end

#
# Time Points
#
@register function tg(rec::Record)
    if isnothing(rec.start)
        return nothing
    else
        rec.t[rec.start]
    end
end

@register function tf(rec::Record)
    if isnothing(rec.stop)
        return nothing
    else
        rec.t[rec.stop]
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
# State Point
#
@register function Sg(rec::Record)
    start = rec.start
    if isnothing(start)
        return nothing
    else
        return rec.X[1, start]
    end
end

@register function Ig(rec::Record)
    start = rec.start
    if isnothing(start)
        return nothing
    else
        return rec.X[2, start]
    end
end

@register function qg(rec::Record)
    start = rec.start
    if isnothing(start)
        return nothing
    else
        return rec.X[3, start] / rec.X[2, start]
    end
end

@register function Sf(rec::Record)
    stop = rec.stop
    if isnothing(stop)
        return nothing
    else
        return rec.X[1, stop]
    end
end

@register function If(rec::Record)
    stop = rec.stop
    if isnothing(stop)
        return nothing
    else
        return rec.X[2, stop]
    end
end

@register function qf(rec::Record)
    stop = rec.stop
    if isnothing(stop)
        return nothing
    else
        return rec.X[3, stop] / rec.X[2, stop]
    end
end

@register function Sp(rec::Record)
    if rec.pidx > 0
        return rec.X[1, rec.pidx]
    else
        return 1
    end
end

@register function Ip(rec::Record)
    if rec.pidx > 0
        return rec.X[2, rec.pidx]
    else
        return null_Ip(rec.r0)
    end
end

@register function ip(rec::Record)
    rec.Ip/rec.r0
end

@register function qp(rec::Record)
    if rec.pidx > 0
        return rec.X[3, rec.pidx] / Ip(rec)
    else
        return rec.Sinf*rec.c / Sp(rec)
    end
end

#
# Special Qunatities
#
@register function m0(rec::Record)
    -rec.r0 * exp(-rec.r0)
end

@register function mf(rec::Record)
    stop = rec.stop
    Sf, If = if isnothing(stop)
        rec.X[[1, 2], end]
    else
        rec.X[[1, 2], stop]
    end
    -Sf * exp(-(Sf+If))
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
    (rec.r0, rec.q0)
end

@register function sol(rec::Record)
    shoot(rec; save_everystep=true).sol
end

@register function lsol(rec::Record)
    lshoot(rec).sol
end

@register_fn next Record

