module PathDecomposition
using EpiDist, QuadGK
const EA = EpiDist.Adpt.Eta

function at(result, x, k)
    src = result.source
    u = result.sol(x)[3k-2:3k]
    S, I = exp.(u[1:2])
    q = src.c * u[3]
    lo = k == 1 ? 0.0 : src.split
    L = k == 1 ? src.split : src.T - src.split
    t = lo + L * x
    p = EA.policy(src, result.eta, t, k)
    w = sqrt(max(0.0, src.r0 * q * I))
    v = max(w - 1 - p, 0.0)
    R = src.r0 / (1 + p + v)
    Rp = -src.r0 / (1 + p + v)^2
    (; S, I, q, p, v, w, R, Rp, L, t, psi=src.r0 * q * I - (1 + p)^2)
end

# Locate continuous active-set crossings in each fixed policy phase. Breaking
# quadrature at all three solutions' crossings resolves small activation slivers.
function active_cuts(result, k)
    grid = sort(unique(vcat(result.sol.x, (result.sol.x[1:end-1] + result.sol.x[2:end]) / 2)))
    roots = Float64[]
    lastx = grid[1]
    lasty = at(result, lastx, k).psi
    for x in grid[2:end]
        y = at(result, x, k).psi
        if isfinite(y) && isfinite(lasty) && signbit(y) != signbit(lasty)
            lo, hi = lastx, x
            flo = lasty
            for j = 1:38
                mid = (lo + hi) / 2
                fm = at(result, mid, k).psi
                signbit(fm) == signbit(flo) ? (lo = mid; flo = fm) : (hi = mid)
            end
            push!(roots, (lo + hi) / 2)
        end
        lastx = x
        lasty = y
    end
    sort(unique(vcat(0.0, roots, 1.0)))
end

include("aligned_nash.jl")

"""Prepare whole-path scaling while retaining the original policy phase boundary."""
function path_source(rec)
    original = EA.eta_source(rec)
    # Scale both phases, retaining the original private-end split so the
    # original public-policy discontinuity remains a mesh boundary.
    merge(original, (; cutoff=0.0))
end
end
