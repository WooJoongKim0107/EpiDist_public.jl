using LinearAlgebra

function solve_mesh(src, eta, seed, x; tol=1e-11)
    U = stack(seed.sol(t) for t in x)
    pnode = [
        EA.policy(src, eta, (k == 1 ? src.split * t : src.split + (src.T - src.split) * t), k)
        for k = 1:2, t in x
    ]
    mids = (x[1:end-1] + x[2:end]) / 2
    pmid = [
        EA.policy(src, eta, (k == 1 ? src.split * t : src.split + (src.T - src.split) * t), k)
        for k = 1:2, t in mids
    ]
    for iteration = 1:30
        R, A, F = EA.collocation_system(U, x, src, pnode, pmid)
        err = maximum(abs, R)
        err < tol && return EA.EtaSolution(x, U, F, err, iteration)
        step = reshape(-(A \ R), 6, :)
        scale = 1.0
        accepted = false
        for trial = 1:20
            next = U + scale * step
            if all(isfinite, next) &&
               minimum(next[[3, 6], :]) > 0 &&
               maximum(next[[1, 2, 4, 5], :]) < 1
                rr, _, _ = EA.collocation_system(next, x, src, pnode, pmid; matrix=false)
                if maximum(abs, rr) < err * (1 - 1e-4scale)
                    U = next
                    accepted = true
                    break
                end
            end
            scale /= 2
        end
        accepted || error("Aligned Nash Newton line search failed")
    end
    error("Aligned Nash Newton did not converge")
end

function refresh_solution(old, sol)
    src = old.source
    S, I = exp.(sol(1.0)[4:5])
    si = EA.final_size(S, I, src.r0)
    s = src.r0 * S
    i = src.r0 * I
    phi = src.r0 * src.c * si * I / S
    if s + i > 1
        sx = exp(1 - s - i + log(s))
        src.r0 * si < sx < s && (phi = max(phi, src.c * src.r0 * si * (1 - sx) / sx))
    end
    phi <= 1 + 1e-8 || error("Private tail reactivation after aligned solve")
    residual = zeros(6)
    EA.boundary!(residual, sol(0.0), sol(1.0), src)
    merge(
        old,
        (;
            sol,
            Sinf=si,
            defect=sol.defect,
            boundary_error=maximum(abs, residual),
            tail_max_phi=phi,
        ),
    )
end

function aligned_metrics(out, cuts)
    src = out.source
    ints = map(1:2) do k
        breaks = sort(unique(vcat(range(0, 1; length=31), cuts[k])))
        first(quadgk(breaks...; rtol=1e-10, atol=1e-12) do t
            a = at(out, t, k)
            a.L .* [a.v * a.S, a.p * a.S, a.v]
        end)
    end
    total = sum(ints)
    early = src.cutoff == 0 ? zeros(3) : ints[1]
    (; V=total[1], P=total[2], V_early=early[1], U_early=early[3])
end
