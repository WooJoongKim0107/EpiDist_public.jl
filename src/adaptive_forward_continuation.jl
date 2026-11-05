# Continuation is performed in z=(log(q0), mu0/q0).  It follows the zero
# set of the leader endpoint residual, then locates Nash matches along it.
# The pseudo-arclength corrector does not assume that either coordinate
# remains monotone; this is the important difference from a fixed-q scan.

_norm2(x) = sqrt(sum(abs2, x))
_dot2(x, y) = x[1] * y[1] + x[2] * y[2]
_inside(z, bounds) = bounds[1][1] <= z[1] <= bounds[1][2] && bounds[2][1] <= z[2] <= bounds[2][2]

function _distinct_coordinates(values; tolerance=1e-7)
    out = Float64[]
    for x in values
        isfinite(x) || continue
        any(y -> abs(x - y) <= tolerance * max(1.0, abs(x), abs(y)), out) || push!(out, Float64(x))
    end
    out
end

function _curve_gradient(f, z; h=1e-5)
    value = f(z)
    isnothing(value) && return nothing
    out = zeros(2)
    for j = 1:2
        step = h * max(1.0, abs(z[j]))
        zp, zm = copy(z), copy(z)
        zp[j] += step
        zm[j] -= step
        fp, fm = f(zp), f(zm)
        out[j] =
            !isnothing(fp) && !isnothing(fm) ? (fp - fm) / (2step) :
            !isnothing(fp) ? (fp - value) / step : !isnothing(fm) ? (value - fm) / step : NaN
    end
    all(isfinite, out) && _norm2(out) > 1e-12 ? out : nothing
end

function _arc_correct(f, prediction, tangent, bounds; tolerance, iterations=12)
    z = copy(prediction)
    for it = 1:iterations
        _inside(z, bounds) || return nothing
        value = f(z)
        isnothing(value) && return nothing
        arc = _dot2(z - prediction, tangent)
        max(abs(value), abs(arc)) <= tolerance && return (z=z, iterations=it)
        gradient = _curve_gradient(f, z)
        isnothing(gradient) && return nothing
        determinant = gradient[1] * tangent[2] - gradient[2] * tangent[1]
        abs(determinant) > 1e-12 || return nothing
        step = [
            (tangent[2] * value - gradient[2] * arc) / determinant,
            (-tangent[1] * value + gradient[1] * arc) / determinant,
        ]
        accepted = false
        for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625)
            next = z - fraction * step
            _inside(next, bounds) || continue
            nv = f(next)
            isnothing(nv) && continue
            na = _dot2(next - prediction, tangent)
            if max(abs(nv), abs(na)) < max(abs(value), abs(arc))
                z = next
                accepted = true
                break
            end
        end
        accepted || return nothing
    end
    nothing
end

"""Internal pseudo-arclength tracer; `nothing` means an undefined residual."""
function _trace_curve(
    f,
    seed,
    bounds;
    direction=1,
    step=0.04,
    min_step=1e-5,
    max_step=0.12,
    max_steps=120,
    tolerance=1e-7,
)
    points = Vector{Float64}[Float64.(seed)]
    gradient = _curve_gradient(f, seed)
    isnothing(gradient) && return (points=points, stop=:singular_seed, failures=0)
    tangent = direction .* [-gradient[2], gradient[1]] ./ _norm2(gradient)
    current_step = Float64(step)
    failures = 0
    for _ = 1:max_steps
        prediction = points[end] + current_step * tangent
        if !_inside(prediction, bounds)
            current_step /= 2
            current_step < min_step &&
                return (points=points, stop=:window_boundary, failures=failures)
            continue
        end
        corrected = _arc_correct(f, prediction, tangent, bounds; tolerance)
        if isnothing(corrected)
            failures += 1
            current_step /= 2
            current_step < min_step &&
                return (points=points, stop=:corrector_failed, failures=failures)
            continue
        end
        next = corrected.z
        gradient = _curve_gradient(f, next)
        isnothing(gradient) && return (points=points, stop=:singular_point, failures=failures)
        new_tangent = [-gradient[2], gradient[1]] ./ _norm2(gradient)
        _dot2(new_tangent, tangent) < 0 && (new_tangent = -new_tangent)
        push!(points, next)
        if length(points) > 12 &&
           _norm2(next - points[1]) < 0.6current_step &&
           _dot2(new_tangent, points[2] - points[1]) > 0
            return (points=points, stop=:closed_curve, failures=failures)
        end
        tangent = new_tangent
        current_step = min(max_step, current_step * (corrected.iterations <= 4 ? 1.25 : 0.8))
    end
    (points=points, stop=:step_limit, failures=failures)
end

"""
    archive_seeds(path, r0, c; stride=25)

Read ordered initial-coordinate paths from the scalar branch-projection CSV.
The expected columns are n,r0,c,stored_qinf,branch,q0,mu0,k0 (additional
columns are allowed). Old k0 is retained as provenance, never as a test of
the corrected forward problem. No backward integration is invoked.
"""
function archive_seeds(path::AbstractString, r0::Real, c::Real; stride::Int=25)
    stride > 0 || throw(ArgumentError("stride must be positive"))
    groups = Dict{String,Vector{NamedTuple}}()
    order = String[]
    open(path) do io
        header = split(readline(io), ',')
        idx = Dict(name => i for (i, name) in enumerate(header))
        for key in ("n", "r0", "c", "stored_qinf", "branch", "q0", "mu0", "k0")
            haskey(idx, key) || throw(ArgumentError("Missing archive column $key"))
        end
        for line in eachline(io)
            parts = split(line, ',')
            rr = parse(Float64, parts[idx["r0"]])
            cc = parse(Float64, parts[idx["c"]])
            isapprox(rr, r0; atol=1e-10, rtol=1e-12) && isapprox(cc, c; atol=1e-10, rtol=1e-12) ||
                continue
            stored = parse(Float64, parts[idx["stored_qinf"]])
            stored == cc ||
                throw(ArgumentError("Archive row has stored_qinf=$stored, nominal c=$cc"))
            branch = parts[idx["branch"]]
            if !haskey(groups, branch)
                groups[branch] = NamedTuple[]
                push!(order, branch)
            end
            q = parse(Float64, parts[idx["q0"]])
            mu = parse(Float64, parts[idx["mu0"]])
            k = parse(Float64, parts[idx["k0"]])
            n = parse(Int, parts[idx["n"]])
            isfinite(q) && q > 0 && isfinite(mu) && isfinite(k) || continue
            push!(groups[branch], (; n, q0=q, mu0=mu, old_k0=k))
        end
    end
    paths = NamedTuple[]
    for branch in order
        rows = sort(groups[branch]; by=x -> x.n)
        indices = unique(vcat(collect(1:stride:length(rows)), [length(rows)]))
        isempty(rows) || push!(paths, (; branch, points=rows[indices]))
    end
    paths
end

function _leader_roots_at_q(
    r0,
    c,
    q,
    betas,
    options;
    leader_tol,
    sign_probes=Ref(0),
    verification_probes=Ref(0),
    invalid_signs=Ref(0),
    invalid_matches=Ref(0),
)
    cache = Dict{Float64,Float64}()
    function value(beta)
        get!(cache, Float64(beta)) do
            sign_probes[] += 1
            try
                answer = _leader_sign(shoot(r0, q, q * beta; options...), c)
                isfinite(answer) || (invalid_signs[] += 1)
                answer
            catch err
                (err isa DomainError || err isa ErrorException) || rethrow()
                invalid_signs[] += 1
                NaN
            end
        end
    end
    mesh = sort(unique(Float64.(betas)))
    vals = value.(mesh)
    found = Float64[]
    for (beta, v) in zip(mesh, vals)
        abs(v) <= leader_tol && push!(found, beta)
    end
    for i = 1:length(mesh)-1
        a, b = vals[i:i+1]
        isfinite(a) && isfinite(b) && signbit(a) != signbit(b) || continue
        beta = try
            find_zero(
                value,
                (mesh[i], mesh[i+1]),
                Bisection();
                xatol=0.0,
                xrtol=0.0,
                atol=0.0,
                rtol=0.0,
            )
        catch err
            (err isa DomainError || err isa ErrorException || err isa ConvergenceFailed) ||
                rethrow()
            continue
        end
        append!(found, (prevfloat(beta), beta, nextfloat(beta)))
    end
    roots = Vector{Float64}[]
    for beta in unique(found)
        z = [log(q), beta]
        verification_probes[] += 1
        r = _try_residual(r0, c, z, options)
        isnothing(r) && (invalid_matches[] += 1)
        isnothing(r) ||
            abs(r[2]) > leader_tol ||
            any(x -> _norm2(x - z) < 1e-7, roots) ||
            push!(roots, z)
    end
    roots
end

function _near_curve(seed, traces, distance)
    for trace in traces
        for i = 1:length(trace)-1
            a, b = trace[i], trace[i+1]
            delta = b - a
            fraction = clamp(_dot2(seed - a, delta) / max(sum(abs2, delta), eps()), 0.0, 1.0)
            _norm2(seed - (a + fraction * delta)) <= distance && return true
        end
        length(trace) == 1 && _norm2(seed - trace[1]) <= distance && return true
    end
    false
end

# A coarse chord crossing need not be a good seed for an unconstrained 2D
# Newton solve. In particular, near the LF boundary a Newton step can cross
# the beta=1 change of outgoing control before resolving a narrow mixed arc.
# Refine the scalar Nash sign change while correcting BACK to the connected
# leader-zero curve. The hyperplane corrector permits folds in either input.
function _curve_crossing_root(
    residual,
    left,
    right,
    bounds;
    aim_tol=1e-7,
    leader_tol=min(aim_tol / 4, 1e-10),
    iterations=80,
)
    a, b = copy(left), copy(right)
    ra, rb = residual(a), residual(b)
    if isnothing(ra) || isnothing(rb)
        return nothing
    end
    maximum(abs, ra) <= aim_tol && return (exp(a[1]), exp(a[1]) * a[2])
    maximum(abs, rb) <= aim_tol && return (exp(b[1]), exp(b[1]) * b[2])
    signbit(ra[1]) != signbit(rb[1]) || return nothing
    function leader(z)
        r = residual(z)
        isnothing(r) ? nothing : r[2]
    end
    for _ = 1:iterations
        chord = b - a
        width = _norm2(chord)
        width > 16eps(Float64) * max(1.0, _norm2(a), _norm2(b)) || return nothing
        prediction = (a + b) / 2
        corrected = _arc_correct(
            leader,
            prediction,
            chord / width,
            bounds;
            tolerance=leader_tol,
            iterations=16,
        )
        isnothing(corrected) && return nothing
        z = corrected.z
        r = residual(z)
        isnothing(r) && return nothing
        maximum(abs, r) <= aim_tol && return (exp(z[1]), exp(z[1]) * z[2])
        if signbit(r[1]) == signbit(ra[1])
            a, ra = z, r
        else
            b, rb = z, r
        end
    end
    nothing
end

"""
    trace_branches(r0,c; seeds=[], archive=nothing, q_bounds=nothing, ...)

Discover leader-endpoint roots on q slices, follow their connected curves
in both directions using pseudo-arclength, and polish Nash crossings.
`archive` accepts ordered paths returned by `archive_seeds`; they supply
additional discovery slices and beta probes, not certified solutions.

The return value includes candidate argument pairs, ordered trace points,
stopping reasons, failed probes and the finite search window. It is an
auditable numerical search, not a completeness/global-optimality theorem.
Only `aimed_shot` performs final reintegration and candidate selection.
"""
function trace_branches(
    r0::Real,
    c::Real;
    seeds=Tuple{Float64,Float64}[],
    archive=nothing,
    q_bounds=nothing,
    beta_bounds=nothing,
    scan_points::Int=7,
    continuation_step::Real=0.04,
    continuation_min_step::Real=1e-5,
    continuation_max_step::Real=0.12,
    continuation_steps::Int=120,
    leader_tol::Real=1e-7,
    aim_tol::Real=1e-7,
    iterations::Int=30,
    tol::Real=1e-12,
    kwargs...,
)
    c > 0 || throw(ArgumentError("c must be positive"))
    scan_points >= 3 || throw(ArgumentError("scan_points must be at least 3"))
    0 < continuation_min_step <= continuation_step <= continuation_max_step ||
        throw(ArgumentError("Invalid continuation step limits"))
    continuation_steps > 0 || throw(ArgumentError("continuation_steps must be positive"))
    lf, co = _references(r0, c)
    qlf = Float64(lf.q0)
    qco = Float64(c * (1 - co.j2) / initial_susceptible)
    qlo, qhi = isnothing(q_bounds) ? (0.5min(qlf, qco), 1.5max(qlf, qco)) : Float64.(q_bounds)
    blo, bhi =
        isnothing(beta_bounds) ? (-2.0, max(2.0, 1.25Float64(co.λ0) / qco)) : Float64.(beta_bounds)
    0 < qlo < qhi && isfinite(qhi) && isfinite(blo) && isfinite(bhi) && blo < bhi ||
        throw(ArgumentError("Invalid continuation window"))
    bounds = ((log(qlo), log(qhi)), (blo, bhi))
    opts = (; tol, save_everystep=false, kwargs...)
    seedpairs = Tuple{Float64,Float64}[(Float64(q), Float64(mu)) for (q, mu) in seeds]
    archive_count = 0
    if !isnothing(archive)
        for path in archive, point in path.points
            push!(seedpairs, (Float64(point.q0), Float64(point.mu0)))
            archive_count += 1
        end
    end
    qs = [q for (q, mu) in seedpairs if qlo <= q <= qhi]
    append!(qs, (qlf, qco))
    append!(qs, exp.(range(log(qlo), log(qhi); length=scan_points)))
    betas = collect(range(blo, bhi; length=2scan_points + 1))
    append!(betas, (beta for beta in (0.0, 1.0) if blo <= beta <= bhi))
    # Preserve ordered archive discoveries, then cover the user-visible mesh.
    # Near-identical q coordinates on the old vertical LF branch must not
    # create hundreds of redundant discovery slices.
    qs = _distinct_coordinates(q for q in qs if qlo <= q <= qhi)
    betas = sort(_distinct_coordinates(betas))
    cache = Dict{Tuple{Float64,Float64},Any}()
    probes = Ref(0)
    invalid = Ref(0)
    function residual(z)
        _inside(z, bounds) || return nothing
        get!(cache, (Float64(z[1]), Float64(z[2]))) do
            probes[] += 1
            r = _try_residual(r0, c, z, opts)
            isnothing(r) && (invalid[] += 1)
            r
        end
    end
    function leader(z)
        r = residual(z)
        isnothing(r) ? nothing : r[2]
    end
    seedzs = Vector{Float64}[]
    for (q, mu) in seedpairs
        q > 0 || continue
        z = [log(q), mu / q]
        r = residual(z)
        !isnothing(r) && abs(r[2]) <= leader_tol && push!(seedzs, z)
    end
    slice_failures = 0
    beta_probe_count = 0
    sign_probes = Ref(0)
    verification_probes = Ref(0)
    invalid_signs = Ref(0)
    invalid_matches = Ref(0)
    for q in qs
        qlo <= q <= qhi || continue
        # Ordered archive coordinates are local predictors, not a quadratic
        # Cartesian product of every old q and every old beta.
        nearby = sort(
            filter(x -> x[1] > 0 && isfinite(x[1]) && isfinite(x[2]), seedpairs);
            by=x -> abs(log(x[1] / q)),
        )
        local_betas = copy(betas)
        append!(
            local_betas,
            (mu / qq for (qq, mu) in Iterators.take(nearby, 6) if blo <= mu / qq <= bhi),
        )
        local_betas = sort(_distinct_coordinates(local_betas))
        beta_probe_count += length(local_betas)
        roots = _leader_roots_at_q(
            r0,
            c,
            q,
            local_betas,
            opts;
            leader_tol,
            sign_probes,
            verification_probes,
            invalid_signs,
            invalid_matches,
        )
        isempty(roots) && (slice_failures += 1)
        append!(seedzs, roots)
    end
    traces = Vector{Vector{Vector{Float64}}}()
    reports = NamedTuple[]
    for z in seedzs
        _near_curve(z, traces, max(10continuation_min_step, 0.05continuation_step)) && continue
        backward = _trace_curve(
            leader,
            z,
            bounds;
            direction=-1,
            step=continuation_step,
            min_step=continuation_min_step,
            max_step=continuation_max_step,
            max_steps=continuation_steps,
            tolerance=leader_tol,
        )
        forward = _trace_curve(
            leader,
            z,
            bounds;
            direction=1,
            step=continuation_step,
            min_step=continuation_min_step,
            max_step=continuation_max_step,
            max_steps=continuation_steps,
            tolerance=leader_tol,
        )
        curve = vcat(reverse(backward.points[2:end]), forward.points)
        push!(traces, curve)
        points = [
            (q0=exp(p[1]), mu0=exp(p[1]) * p[2], nash=residual(p)[1], leader=residual(p)[2]) for
            p in curve
        ]
        push!(
            reports,
            (;
                points,
                backward_stop=backward.stop,
                forward_stop=forward.stop,
                corrector_failures=backward.failures + forward.failures,
            ),
        )
    end
    rootseeds = Tuple{Float64,Float64}[]
    polish_attempts = 0
    failed_matches = 0
    curve_refinement_successes = 0
    curve_refinement_failures = 0
    newton_fallbacks = 0
    for trace in reports
        for point in trace.points
            abs(point.nash) <= aim_tol && push!(rootseeds, (point.q0, point.mu0))
        end
        for i = 1:length(trace.points)-1
            a, b = trace.points[i:i+1]
            signbit(a.nash) != signbit(b.nash) || continue
            fraction = abs(a.nash) / (abs(a.nash) + abs(b.nash))
            q = exp((1 - fraction) * log(a.q0) + fraction * log(b.q0))
            beta = (1 - fraction) * a.mu0 / a.q0 + fraction * b.mu0 / b.q0
            polish_attempts += 1
            za = [log(a.q0), a.mu0 / a.q0]
            zb = [log(b.q0), b.mu0 / b.q0]
            root = _curve_crossing_root(residual, za, zb, bounds; aim_tol=aim_tol / 4)
            if isnothing(root)
                curve_refinement_failures += 1
                newton_fallbacks += 1
                root = _polish(
                    r0,
                    c,
                    (q, q * beta);
                    aim_tol=aim_tol / 4,
                    iterations,
                    options=opts,
                    bounds,
                    residual_probe=residual,
                )
            else
                curve_refinement_successes += 1
            end
            isnothing(root) ? (failed_matches += 1) : push!(rootseeds, root)
        end
    end
    (;
        roots=unique(rootseeds),
        traces=reports,
        q_bounds=(qlo, qhi),
        beta_bounds=(blo, bhi),
        q_slices=length(qs),
        beta_probes=beta_probe_count,
        slice_failures,
        discovery_roots=length(seedzs),
        continuation_residual_probes=probes[],
        invalid_continuation_probes=invalid[],
        discovery_sign_probes=sign_probes[],
        discovery_verification_probes=verification_probes[],
        invalid_discovery_signs=invalid_signs[],
        invalid_discovery_matches=invalid_matches[],
        archive_seed_count=archive_count,
        polish_attempts,
        failed_matches,
        curve_refinement_successes,
        curve_refinement_failures,
        newton_fallbacks,
        complete=false,
    )
end
