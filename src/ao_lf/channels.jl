module PostChannels
using EpiDist, QuadGK, SparseArrays, LinearAlgebra
include("decomposition.jl")
const PD = PathDecomposition
const EA = EpiDist.Adpt.Eta
const OD = EpiDist.Adpt
export response, channels
struct Sensitivity
    x::Vector{Float64}
    U::Matrix{Float64}
    Fa::Matrix{Float64}
    Fb::Matrix{Float64}
    residual::Float64
end

function (s::Sensitivity)(x)
    j = clamp(searchsortedlast(s.x, x), 1, length(s.x) - 1)
    h = s.x[j+1] - s.x[j]
    t = (x - s.x[j]) / h
    (2t^3 - 3t^2 + 1) .* s.U[:, j] +
    (t^3 - 2t^2 + t) * h .* s.Fa[:, j] +
    (-2t^3 + 3t^2) .* s.U[:, j+1] +
    (t^3 - t^2) * h .* s.Fb[:, j]
end

function coefficients(u, src, p, g, active)
    J = zeros(6, 6)
    b = zeros(6)
    for k = 1:2
        idx = 3k-2:3k
        S, I = exp.(u[idx[1:2]])
        z = u[3k]
        w = sqrt(src.r0 * src.c * z * I)
        v = max(w - 1 - p[k], 0.0)
        r = src.r0 / (1 + p[k] + v)
        dv = active[k] ? [0.0, w / 2, w / (2z)] : zeros(3)
        dr = -r / (1 + p[k] + v) .* dv
        jp = zeros(3, 3)
        for j = 1:3
            jp[1, j] = -I * (dr[j] + (j == 2 ? r : 0.0))
            jp[2, j] = S * (dr[j] + (j == 1 ? r : 0.0))
            jp[3, j] =
                I * z * dr[j] + (j == 2 ? r * I * z : 0.0) + (j == 3 ? r * I : 0.0) + dv[j] / src.c
        end
        L = k == 1 ? src.split : src.T - src.split
        rp = active[k] ? 0.0 : -r / (1 + p[k] + v)
        vp = active[k] ? -1.0 : 0.0
        J[idx, idx] .= L .* jp
        b[idx] .= L * g[k] .* [-I * rp, S * rp, I * z * rp + vp / src.c]
    end
    J, b
end

function sensitivity(out)
    src = out.source
    sol = out.sol
    x = sol.x
    n = length(x)
    U = sol.U
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    rhs = zeros(6n)
    function entry(r, c, v)
        v == 0 && return
        push!(rows, r)
        push!(cols, c)
        push!(vals, v)
    end
    entry(1, 1, 1.0)
    entry(2, 2, 1.0)
    for j = 1:3
        entry(2 + j, 3 + j, 1.0)
        entry(2 + j, 6(n - 1) + j, -1.0)
    end
    ST, IT = exp.(U[4:5, end])
    sf = out.Sinf
    z = sf / ST
    D = 1 - src.r0 * sf
    entry(6, 6(n - 1) + 4, src.r0 * z * (ST - sf) / D)
    entry(6, 6(n - 1) + 5, src.r0 * z * IT / D)
    entry(6, 6n, 1.0)
    Ja = Array{Float64}(undef, 6, 6, n - 1)
    Jb = similar(Ja)
    ba = zeros(6, n - 1)
    bb = similar(ba)
    eye = Matrix{Float64}(I, 6, 6)
    policies(t, eta) = [
        EA.policy(src, eta, (k == 1 ? src.split * t : src.split + (src.T - src.split) * t), k)
        for k = 1:2
    ]
    for j = 1:n-1
        h = x[j+1] - x[j]
        m = (x[j] + x[j+1]) / 2
        ua = U[:, j]
        ub = U[:, j+1]
        um = sol(m)
        active = [PD.at(out, m, k).v > 0 for k = 1:2]
        ja, ga = coefficients(ua, src, policies(x[j], out.eta), policies(x[j], 1.0), active)
        jb, gb = coefficients(ub, src, policies(x[j+1], out.eta), policies(x[j+1], 1.0), active)
        jm, gm = coefficients(um, src, policies(m, out.eta), policies(m, 1.0), active)
        Ja[:, :, j] = ja
        Jb[:, :, j] = jb
        ba[:, j] = ga
        bb[:, j] = gb
        A = -eye - h / 6 * (ja + 4jm * (eye / 2 + h / 8 * ja))
        B = eye - h / 6 * (jb + 4jm * (eye / 2 - h / 8 * jb))
        rhs[6j+1:6j+6] .= h / 6 * (ga + 4gm + gb + h / 2 * jm * (ga - gb))
        for k = 1:6, l = 1:6
            entry(6j + k, 6(j - 1) + l, A[k, l])
            entry(6j + k, 6j + l, B[k, l])
        end
    end
    mat = sparse(rows, cols, vals, 6n, 6n)
    y = mat \ rhs
    Y = reshape(y, 6, :)
    fa = stack(Ja[:, :, j] * Y[:, j] + ba[:, j] for j = 1:n-1)
    fb = stack(Jb[:, :, j] * Y[:, j+1] + bb[:, j] for j = 1:n-1)
    Sensitivity(x, Y, fa, fb, maximum(abs, mat * y - rhs))
end
include("public_mesh.jl")
function channels(base; ode_tol=1e-9, quad_tol=1e-7)
    src = base.source
    dy = sensitivity(base)
    cuts = [public_cuts(base, k) for k = 1:2]
    data(x, k) = begin
        a = PD.at(base, x, k)
        y = dy(x)
        (; a, dI=a.I * y[3k-1], dq=src.c * y[3k], dp=EA.policy(src, 1.0, a.t, k))
    end
    last = [0.0, src.c * dy(1.0)[6], 0.0]
    aux = Dict{Int,Any}()
    for k in (2, 1)
        function rhs!(du, u, p, x)
            d = data(x, k)
            a = d.a
            du[1] = -a.L * a.v * a.S
            du[2] = a.L * (a.R * a.I * u[2] + a.R * a.q * d.dI)
            du[3] = a.L * (a.R * a.I * u[3] + a.Rp * a.I * a.q * d.dp)
        end
        sol = OD.solve(
            OD.ODEProblem(rhs!, last, (1.0, 0.0)),
            OD.Vern9();
            abstol=ode_tol / 10,
            reltol=ode_tol,
            dtmax=0.01,
            tstops=reverse(cuts[k][2:end-1]),
        )
        EpiDist.Forward.successful_retcode(sol.retcode) || error("Sensitivity auxiliary ODE failed")
        aux[k] = sol
        last = sol(0.0)
    end
    # Restrict the forcing time, not the expenditure time, to after the final
    # private-investment end of this eta-specific equilibrium.
    tf = 0.0
    for k = 1:2, (lo, hi) in zip(cuts[k][1:end-1], cuts[k][2:end])
        PD.at(base, (lo + hi) / 2, k).v > 0 && (tf = max(tf, PD.at(base, hi, k).t))
    end
    kf = tf < src.split ? 1 : 2
    xf = clamp(kf == 1 ? tf / src.split : (tf - src.split) / (src.T - src.split), 0.0, 1.0)
    af = PD.at(base, xf, kf)
    _, qIf, qpf = aux[kf](xf)
    postI = af.S * qIf
    postp = af.S * qpf
    vals = zeros(10)

    for k = 1:2
        breaks = sort(
            unique(
                vcat(
                    range(0, 1; length=21),
                    cuts[k],
                    [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.005, 0.01],
                ),
            ),
        )
        vals .+= first(
            quadgk(breaks...; rtol=quad_tol, atol=1e-10) do x
                d = data(x, k)
                a = d.a
                B, qI, qp = aux[k](x)
                C = a.S - a.Rp * a.I * B
                act = a.v > 0
                alpha = a.R / 2
                vi = act ? alpha * (a.q * d.dI + a.I * d.dq) - d.dp : 0.0
                a.L .* [
                    -B * a.R * d.dI,
                    -B * a.Rp * a.I * d.dp,
                    act ? C * alpha * a.q * d.dI : 0.0,
                    act ? -C * d.dp : 0.0,
                    act ? C * alpha * a.I * qI : 0.0,
                    act ? C * alpha * a.I * qp : 0.0,
                    act ? C * alpha * a.I * postI / a.S : 0.0,
                    act ? C * alpha * a.I * postp / a.S : 0.0,
                    -B * (a.R * d.dI + a.Rp * a.I * d.dp) + C * vi,
                    act ? C * alpha * a.I * (qI + qp - d.dq) : 0.0,
                ]
            end,
        )
    end
    # Saved order: IR(I,p), Fut(I,p), Fut-post(I,p), FR, DS.
    (;
        input=vals[[1, 2, 5, 6, 7, 8, 3, 4]],
        post_cutoff=tf,
        slope=vals[9],
        closure=sum(vals[1:6]) - vals[9],
        q_gap=vals[10],
        linear_residual=dy.residual,
    )

end
end
