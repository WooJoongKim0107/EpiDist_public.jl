using EpiDist

function equilibria(r0, c)
    lf, co = Forward._references(r0, c)
    ao = if lf.j1 == 0 && co.j1 == 0
        s = Commons.initial_susceptible
        i = Commons.initial_infected
        final = Forward._final_size(r0 * s, r0 * i)
        q = c * final / (r0 * s)
        Forward.Record(Forward.shoot(r0, q, q / (1 - final); tol=1e-13), c, 0)
    else
        Forward.aimed_shot(
            r0,
            c;
            scan_points=3,
            continuation_steps=40,
            save_everystep=false,
            maxiters=300000,
        )
    end
    (; lf, co, ao)
end
