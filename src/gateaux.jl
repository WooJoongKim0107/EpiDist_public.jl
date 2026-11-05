using Optim
using Integrals


# ============================================================================
#   PartialX Derivative of Time-Series
# ============================================================================

Scales = NamedTuple{(:none, :dcpop, :depth)}

struct PartialX <: PObject
    branch::Int
    J1::Real
    J1pop::Real

    base::Bullet
    next::Bullet

    dcpop::Float64
    dλ0::Float64
    dJ1pop::Float64
    dq0::Float64
    dJ1::Float64

    depth::Float64  # abs( \min_{t} \partial_{c} Ipop )
    scales::Scales
end

function PartialX(rec::Record; tmax=10000.0, kwargs...)  # tmax is deliberately chosen huge to ensure convergence.
    tmax = (rec.branch > 0) ? tmax : max(tmax, 1000.0)  # To reduce meaningless calculations
    options = (; callback=fcallback, save_everystep=true, dense=true)
    bltbase = shoot(rec; tmax=tmax, options..., kwargs...)
    bltnext = shoot(rec.r0, rec.q0+rec.dq0, rec.λ0+rec.dλ0; tmax=tmax, options..., kwargs...)
    scale = (bltnext.sol(rec.tpp)[5]-bltbase.sol(rec.tpp)[5]) |> abs |> inv
    f(t) = bltnext.sol(t[1])[5]-bltbase.sol(t[1])[5]
    opt = optimize(t->f(t)*scale, [rec.tpp]; x_abstol=1e-8)
    depth = abs(f(opt.minimizer))
    scales = Scales((; none=1, dcpop=inv(rec.dcpop), depth=inv(depth)))
    increments = (rec.dcpop, rec.dλ0, rec.dJ1pop, rec.dq0, rec.dJ1)
    PartialX(rec.branch, rec.J1, rec.J1pop, bltbase, bltnext, increments..., depth, scales)
end

"""norm ∈ {:none, :dcpop, :depth}"""
function (self::PartialX)(t::Real; norm=:dcpop)
    d = self.next.sol(t)-self.base.sol(t)
    scale = self.scales[norm]
    return d .* scale
end

"""norm ∈ {:none, :dcpop, :depth}"""
function (self::PartialX)(ts::AbstractVector{<:Real}; norm=:dcpop)
    self.(ts; norm=norm) |> stack
end

@register(PartialX, :∂Ap∂cp, self->self.dJ1pop/self.dcpop)
@register(PartialX, :r0, self->self.base.r0)

@register(PartialX, :t, self->self.base.sol.t)

@register function ΔSp(self::PartialX)
    t -> self(t; norm=:none)[4]
end

@register function ∂Sp(self::PartialX)
    t -> self(t; norm=:dcpop)[4]
end

@register function dSp(self::PartialX)
    t -> self(t; norm=:depth)[4]
end

@register function ΔIp(self::PartialX)
    t -> self(t; norm=:none)[5]
end

@register function ∂Ip(self::PartialX)
    t -> self(t; norm=:dcpop)[5]
end

@register function dIp(self::PartialX)
    t -> self(t; norm=:depth)[5]
end

@register function ΔSpt(self::PartialX)
    self.ΔSp.(self.t)
end

@register function ∂Spt(self::PartialX)
    self.∂Sp.(self.t)
end

@register function dSpt(self::PartialX)
    self.dSp.(self.t)
end

@register function ΔIpt(self::PartialX)
    self.ΔIp.(self.t)
end

@register function ∂Ipt(self::PartialX)
    self.∂Ip.(self.t)
end

@register function dIpt(self::PartialX)
    self.dIp.(self.t)
end

# ================================================================================
#   Functional Derivative (of cumulative investment w.r.t. reservoir perturbation)
# ================================================================================

struct FuncDeriv <: PObject
    sol
    r0::Real
    cSinf::Real
    tg::Real
    tf::Real
    intg
end

function FuncDeriv(sol, r0, cSinf)
    tg = findfirst(sol[3, :]) do varphi
        varphi >= 1 || isapprox(varphi, 1)
    end |> (idx -> isnothing(idx) ? 0.0 : sol.t[idx])
    tf = findlast(sol[3, :]) do varphi
        varphi >= 1 || isapprox(varphi, 1)
    end |> (idx -> isnothing(idx) ? 0.0 : sol.t[idx])
    prob = ODEProblem(0.0, [tg, tf]) do u, p, t
        X = sol(t)
        -X[5]^2 * X[3]^(-3/2) / X[1]
    end
    intg = solve(prob, Vern9(); reltol=1e-15, abstol=1e-20, save_everystep=true, dense=true)
    FuncDeriv(sol, r0, cSinf, tg, tf, intg)
end

function (self::FuncDeriv)(t::Real)
    off_phase(self, t) + on_phase(self, t)
end

function on_phase(self::FuncDeriv, t::Real)
    X = self.sol(t)
    if t <= self.tg
        return 0.0
    elseif t <= self.tf
        sig = action(X)[2] |> Commons.sigma
        slope = X[1]*X[3]/X[5]
        return sig*self.cSinf/(2*self.r0) * (slope*self.intg(t) + 1)
    else
        sig = action(X)[2] |> Commons.sigma
        slope = X[1]*X[3]/X[5]
        return sig*self.cSinf/(2*self.r0) * slope*self.intg(self.tf)
    end
end

function off_phase(self::FuncDeriv, t::Real)
    X = self.sol(t)
    sig = action(X)[2] |> Commons.sigma
    qS = X[1]*X[3]/X[5]
    sig*(qS - self.cSinf) / self.r0
end

function fv(self::FuncDeriv, t::Real)
    X = self.sol(t)
    self.cSinf*X[5]/X[3] / self.r0
end

function env_state(self::FuncDeriv, t::Real)
    X = self.sol(t)
    (X[1]*X[3]/X[5] - self.cSinf) / self.r0
end

function env_strategy(self::FuncDeriv, t::Real)
    X = self.sol(t)
    if self.tg <= t <= self.tf
        return self.cSinf/(2*self.r0)
    else
        return 0.0
    end
end

function revaluation(self::FuncDeriv, t::Real)
    X = self.sol(t)
    if self.tg <= t
        slope = X[1]*X[3]/X[5]
        return self.cSinf/(2*self.r0) * slope*self.intg(clamp(t, self.tg, self.tf))
    else
        return 0.0
    end
end

function Commons.sigma(self::FuncDeriv, t)
    X = self.sol(t)
    action(X)[2] |> Commons.sigma
end

function Commons.sigma(self::FuncDeriv)
    Xs = self.sol[:, :]
    map(eachcol(Xs)) do X
        action(X)[2] |> Commons.sigma
    end
end

@register env_state(self::FuncDeriv) = env_state.(Ref(self), self.sol.t)
@register env_strategy(self::FuncDeriv) = env_strategy.(Ref(self), self.sol.t)
@register revaluation(self::FuncDeriv) = revaluation.(Ref(self), self.sol.t)
@register_fn sigma FuncDeriv

const Integrated = NamedTuple{(
    :int_env_state,
    :int_env_strategy,
    :int_revaluation,
    :int_revaluation_pre,
    :int_revaluation_post
)}

function integrate_all(g::FuncDeriv, dIt; kwargs...)
    Integrated((
        integrate_env_state(g, dIt; kwargs...),
        integrate_env_strategy(g, dIt; kwargs...),
        integrate_revaluation(g, dIt; kwargs...),
        integrate_revaluation_pre(g, dIt; kwargs...),
        integrate_revaluation_post(g, dIt; kwargs...),
    ))
end

function inner_product(g::FuncDeriv, dIt, term, bounds;
                           alg=QuadGKJL(), abstol=1e-10, reltol=1e-8)  # Default tol is chosen large for safety.
    prob = IntegralProblem(bounds...) do t, p
        term(g, t) * Commons.sigma(g, t) * dIt(t)
    end
    solve(prob, alg; abstol=abstol, reltol=reltol).u
end

function integrate_env_state(g::FuncDeriv, dIt; kwargs...)
    inner_product(g, dIt, env_state, (0.0, g.tf); kwargs...)
end

function integrate_env_strategy(g::FuncDeriv, dIt; kwargs...)
    inner_product(g, dIt, env_strategy, (0.0, g.tf); kwargs...)
end

function integrate_revaluation(g::FuncDeriv, dIt; kwargs...)
    inner_product(g, dIt, revaluation, (0.0, g.sol.t[end]); kwargs...)
end

function integrate_revaluation_pre(g::FuncDeriv, dIt; kwargs...)
    inner_product(g, dIt, revaluation, (0.0, g.tf); kwargs...)
end

function integrate_revaluation_post(g::FuncDeriv, dIt; kwargs...)
    inner_product(g, dIt, revaluation, (g.tf, g.sol.t[end]); kwargs...)
end

# ============================================================================
#   Partial Derivative of Objective
# ============================================================================

struct PartialA <: PObject
    ∂X::PartialX
    δA::FuncDeriv
    intgs::Integrated
end

function PartialA(expt::Record; tmax=10000.0, intkwargs=(;), kwargs...)  # tighten `intkwargs=(; abstol=1e-12)` as much as possible for better precision.
    ∂X = PartialX(expt; tmax=tmax, kwargs...)
    δA = FuncDeriv(∂X.base.sol, expt.r0, expt.c*expt.Sinf)
    intgs = if expt.branch == 0
        Integrated(fill(NaN, 5))
    else
        integrate_all(δA, ΔIp(∂X); intkwargs...)
    end
    PartialA(∂X, δA, intgs)
end

function Commons.sigma(self::PartialA)
    Commons.sigma(self.δA)
end

@delegate PartialA begin
    ∂X => (
        :ΔSp, :∂Sp, :dSp,
        :ΔIp, :∂Ip, :dIp,
        :ΔSpt, :∂Spt, :dSpt,
        :ΔIpt, :∂Ipt, :dIpt,
        :branch, :J1, :J1pop,
        :dλ0, :dJ1pop, :dcpop, :dq0, :dJ1,
        :t, :∂Ap∂cp, :depth,
    )
    δA => (
        :env_state, :env_strategy, :revaluation, :sigma,
        :r0, :cSinf, :tg, :tf
    )
    intgs => (
        :int_env_state,
        :int_env_strategy,
        :int_revaluation,
        :int_revaluation_pre,
        :int_revaluation_post
    )
end

@register(PartialA, :δAδIp) do self
    (self.env_state + self.env_strategy + self.revaluation) .* self.sigma
end

@register(PartialA, :integrand) do self
    (self.env_state + self.env_strategy + self.revaluation) .* self.sigma .* self.∂Ipt
end

@register(PartialA, :integrand_env_state) do self
    self.env_state .* self.sigma .* self.∂Ipt
end

@register(PartialA, :integrand_env_strategy) do self
    self.env_strategy .* self.sigma .* self.∂Ipt
end

@register(PartialA, :integrand_passive) do self
    (self.env_state + self.env_strategy) .* self.sigma .* self.∂Ipt
end

@register(PartialA, :integrand_revaluation) do self
    self.revaluation .* self.sigma .* self.∂Ipt
end

@register(PartialA, :integrand_strategic) do self
    (self.env_strategy + self.revaluation) .* self.sigma .* self.∂Ipt
end

@register(PartialA, :gateaux) do self
    self.int_env_state + self.int_env_strategy + self.int_revaluation
end

@register(PartialA, :sensitivity) do self
    (self.int_env_state + self.int_env_strategy + self.int_revaluation) * inv(self.dJ1pop)
end

@register(PartialA, :sensitivity_passive) do self
    (self.int_env_state + self.int_env_strategy) * inv(self.dJ1pop)
end

@register(PartialA, :sensitivity_futility) do self
    self.int_env_state * inv(self.dJ1pop)
end

@register(PartialA, :sensitivity_adjust) do self
    self.int_env_strategy * inv(self.dJ1pop)
end

@register(PartialA, :sensitivity_revaluation) do self
    self.int_revaluation * inv(self.dJ1pop)
end

@register(PartialA, :sensitivity_revaluation_tail) do self
    self.int_revaluation_post * inv(self.dJ1pop)
end

@register(PartialA, :sensitivity_active) do self
    (self.int_env_strategy + self.int_revaluation) * inv(self.dJ1pop)
end

@register(PartialA, :∂A∂cp) do self
    (self.int_env_state + self.int_env_strategy + self.int_revaluation) * inv(self.dcpop)
end


# ============================================================================
#   Misc.
# ============================================================================

function Base.show(io::IO, res::FuncDeriv)
    prefix = string(FuncDeriv) |> x->split(x, '.')[end-1:end] |> x->join(x, '.')
    print(io, "$(prefix)(cSinf=$(res.cSinf))")
end

function Base.show(io::IO, ::MIME"text/plain", res::FuncDeriv)
    Base.show(io, res)
    silent_fields = (:sol, :intg)
    fields = fieldnames(typeof(res))
    n_fields = length(fields)
    for (i, field) in enumerate(fields)
        prefix = i == n_fields ? "└─ " : "├─ "
        value = getfield(res, field)
        if field in silent_fields
            print(io, '\n', "$(prefix)$(field): ...")
        else
            print(io, '\n', "$(prefix)$(field): $(value)")
        end
    end
end
