# Fourth-order Hermite–Simpson collocation with a sparse block Jacobian.
struct EtaSolution
    x::Vector{Float64}
    U::Matrix{Float64}
    F::Matrix{Float64}
    defect::Float64
    iterations::Int
end

function (sol::EtaSolution)(x::Real)
    j = clamp(searchsortedlast(sol.x, x), 1, length(sol.x) - 1)
    h = sol.x[j+1] - sol.x[j]
    s = (x - sol.x[j]) / h
    (2s^3 - 3s^2 + 1) .* sol.U[:, j] +
    (s^3 - 2s^2 + s) * h .* sol.F[:, j] +
    (-2s^3 + 3s^2) .* sol.U[:, j+1] +
    (s^3 - s^2) * h .* sol.F[:, j+1]
end

function local_rhs_jac(u, src, p, L)
    S, I = exp.(u[1:2])
    z = u[3]
    w = sqrt(max(0.0, src.r0 * src.c * z * I))
    v = max(w - 1 - p, 0.0)
    r = src.r0 / (1 + p + v)
    dv = v > 0 ? [0.0, w / 2, w / (2z)] : zeros(3)
    dr = -r / (1 + p + v) .* dv
    f = L .* [-r * I, r * S - 1, r * I * z + v / src.c]
    J = zeros(3, 3)
    for j = 1:3
        J[1, j] = -I * (dr[j] + (j == 2 ? r : 0.0))
        J[2, j] = S * (dr[j] + (j == 1 ? r : 0.0))
        J[3, j] =
            I * z * dr[j] + (j == 2 ? r * I * z : 0.0) + (j == 3 ? r * I : 0.0) + dv[j] / src.c
    end
    f, L .* J
end

function combined_rhs_jac(u, src, pp)
    f = zeros(6)
    J = zeros(6, 6)
    for k = 1:2
        idx = 3k-2:3k
        L = k == 1 ? src.split : src.T - src.split
        ff, jj = local_rhs_jac(u[idx], src, pp[k], L)
        f[idx] .= ff
        J[idx, idx] .= jj
    end
    f, J
end

function collocation_system(U, x, src, pnode, pmid; matrix=true)
    n = length(x)
    R = zeros(6n)
    F = zeros(6, n)
    J = Array{Float64}(undef, 6, 6, n)
    boundary!(view(R, 1:6), U[:, 1], U[:, end], src)
    for j = 1:n
        F[:, j], J[:, :, j] = combined_rhs_jac(U[:, j], src, pnode[:, j])
    end
    rows = Int[]
    cols = Int[]
    values = Float64[]
    Id = Matrix{Float64}(I, 6, 6)
    function entry(r, c, v)
        v == 0 && return
        push!(rows, r)
        push!(cols, c)
        push!(values, v)
    end
    if matrix
        entry(1, 1, 1.0)
        entry(2, 2, 1.0)
        for j = 1:3
            entry(2 + j, 3 + j, 1.0)
            entry(2 + j, 6(n - 1) + j, -1.0)
        end
        ST, IT = exp.(U[4:5, end])
        sf = final_size(ST, IT, src.r0)
        q = sf / ST
        D = 1 - src.r0 * sf
        entry(6, 6(n - 1) + 4, src.r0 * q * (ST - sf) / D)
        entry(6, 6(n - 1) + 5, src.r0 * q * IT / D)
        entry(6, 6n, 1.0)
    end
    for j = 1:n-1
        h = x[j+1] - x[j]
        a = U[:, j]
        b = U[:, j+1]
        fa = F[:, j]
        fb = F[:, j+1]
        mid = (a + b) / 2 + h / 8 * (fa - fb)
        fm, Jm = combined_rhs_jac(mid, src, pmid[:, j])
        R[6j+1:6j+6] .= b - a - h / 6 * (fa + 4fm + fb)
        if matrix
            Ja = J[:, :, j]
            Jb = J[:, :, j+1]
            A = -Id - h / 6 * (Ja + 4Jm * (Id / 2 + h / 8 * Ja))
            B = Id - h / 6 * (Jb + 4Jm * (Id / 2 - h / 8 * Jb))
            for k = 1:6, l = 1:6
                entry(6j + k, 6(j - 1) + l, A[k, l])
                entry(6j + k, 6j + l, B[k, l])
            end
        end
    end
    R, matrix ? sparse(rows, cols, values, 6n, 6n) : nothing, F
end

function solve_collocation(src, eta, initial; tol=1e-10, intervals=600, max_step=0.15, maxiters=30)
    n = max(intervals, ceil(Int, max(src.split, src.T - src.split) / max_step))
    x = collect(range(0.0, 1.0; length=n + 1))
    U =
        isnothing(initial) ? stack(original_seed(src, t) for t in x) :
        stack(initial.sol(t) for t in x)
    pnode = [
        policy(src, eta, (k == 1 ? src.split * t : src.split + (src.T - src.split) * t), k) for
        k = 1:2, t in x
    ]
    mids = (x[1:end-1] + x[2:end]) / 2
    pmid = [
        policy(src, eta, (k == 1 ? src.split * t : src.split + (src.T - src.split) * t), k) for
        k = 1:2, t in mids
    ]
    for iteration = 1:maxiters
        R, A, F = collocation_system(U, x, src, pnode, pmid)
        err = maximum(abs, R)
        err < tol && return EtaSolution(x, U, F, err, iteration)
        direction = reshape(-(A \ R), 6, :)
        alpha = 1.0
        accepted = false
        for trial = 1:20
            candidate = U + alpha * direction
            if all(isfinite, candidate) &&
               minimum(candidate[[3, 6], :]) > 0 &&
               maximum(candidate[[1, 2, 4, 5], :]) < 1.0
                rr, _, _ = collocation_system(candidate, x, src, pnode, pmid; matrix=false)
                if maximum(abs, rr) < err * (1 - 1e-4alpha)
                    U = candidate
                    accepted = true
                    break
                end
            end
            alpha /= 2
        end
        accepted || error("Nash Newton line search failed: eta=$eta, defect=$err")
    end
    error("Nash collocation did not converge: eta=$eta")
end
