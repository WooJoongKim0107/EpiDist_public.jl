using Test
using EpiDist
using SciMLBase: successful_retcode

@testset "shared and null model" begin
    r0 = 3.0
    @test 0 < EpiDist.Commons.null_Sinf(r0) < 1
    @test 0 < EpiDist.Commons.null_sinf(r0) < 1

    sol = EpiDist.Null.shoot(r0)
    @test successful_retcode(sol.retcode)
    @test size(sol, 1) == 2
    @test all(isfinite, sol[:, end])
end

@testset "Nash and control searches" begin
    r0, c = 3.0, 20.0
    nash = EpiDist.Nash.aimed_shot(r0, c; save_everystep=true)
    ctrl = EpiDist.Ctrl.aimed_shot(r0, c; save_everystep=true)

    for (record, rows) in ((nash, 3), (ctrl, 4))
        @test abs(record.cerr) < 1e-8
        @test size(record.X, 1) == rows
        @test length(record.t) == size(record.X, 2)
        @test all(isfinite, record.X)
        @test all(isfinite, (record.j1, record.j2, record.obj))
    end
end

@testset "adaptive model" begin
    bullet = EpiDist.Adpt.shoot(0.2, 10.0, 12.0; save_everystep=true)
    @test successful_retcode(bullet.sol.retcode)
    @test size(bullet.sol, 1) == 6
    @test all(isfinite, bullet.sol[:, end])

    records = (
        (EpiDist.Adpt.hunt_active(1.5, 40.0), 2),
        (EpiDist.Adpt.hunt_inactive(1.5, 40.0), 1),
        (EpiDist.Adpt.hunt_doormat(1.5, 40.0), 3),
    )
    for (record, branch) in records
        @test record.branch == branch
        @test abs(record.r0err) < 1e-7
        @test size(record.X, 1) == 6
        @test length(record.t) == size(record.X, 2)
        @test all(isfinite, record.X)
        @test all(isfinite, (record.j1, record.j2, record.obj))
    end
end

@testset "deviator models" begin
    r0 = 7.5
    q0 = 6.0554544523933395e-6
    λ0 = 0.0005577161565235696

    for model in (EpiDist.DevC, EpiDist.DevL)
        bullet = model.shoot(r0, q0, λ0; save_everystep=true)
        @test successful_retcode(bullet.sol.retcode)
        @test all(isfinite, bullet.sol[:, end])

        target_c = model.infer_c(bullet)
        @test isfinite(target_c)
        record = model.aimed_shot(r0, target_c, target_c, λ0, q0)
        @test abs(record.cerr) < 1e-8
        @test length(record.t) == size(record.X, 2)

        derivative = model.aimed_shot(
            r0, target_c, target_c, λ0, q0;
            dcpop=0.1, dλ0=1e-6, dJ1pop=0.01,
        )
        @test derivative.dcpop == 0.1
        @test derivative.dλ0 == 1e-6
        @test derivative.dJ1pop == 0.01
        @test all(isfinite, (derivative.dq0, derivative.dJ1))
    end
end

@testset "removed public surface" begin
    @test !isdefined(EpiDist, :Expt)
    @test !isdefined(EpiDist, :Expt2)
    @test !isdefined(EpiDist, :Expt3)
    for model in (EpiDist.Nash, EpiDist.Ctrl, EpiDist.Adpt, EpiDist.DevC, EpiDist.DevL)
        @test !isdefined(model, :PArray)
        @test !isdefined(model, :PObject)
    end
end
