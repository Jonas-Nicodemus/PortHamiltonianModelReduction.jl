using Test

using PortHamiltonianModelReduction

using LinearAlgebra, SkewLinearAlgebra, ControlSystemsBase
using PortHamiltonianSystems, OrdinaryDiffEq

@testset "test_phdmd.jl" begin
    J = [0. 1.; -1. 0.]
    R = [2. 0.; 0. 1.]
    Q = [1. 0.; 0. 1.]
    G = [6.; 0.;;]

    Σ = ss(phss(J, R, Q, G))

    n, m = Σ.nx, Σ.nu

    Δt = 1e-1
    t = 0:Δt:1
    x0 = zeros(n)
    u_(x,t) = [exp(-0.5 * t) * sin(t^2)]

    res = lsim(Σ, u_, t ;x0=x0, alg=OrdinaryDiffEq.ImplicitMidpoint(), dt=Δt)
    H = 1//2 * sum(res.x .* (Q * res.x), dims=1)[1,:]
    data = tddata(res)

    @testset "pod" begin
        r = 1

        Σr = pod(Σ, data.X, r) 
        @test Σr.nx == r
        @test norm(Σ - Σr, Inf) < 1e1
    end

    @testset "opinf" begin
        Σr, res = opinf(data)
        @test norm(Σ - Σr, Inf) < 1e-12 

        r = 1
        V = PortHamiltonianModelReduction.podbasis(data.X, r)
        @test size(V, 2) == r

        Σrpod = pod(Σ, V)
        Σropinf, res = opinf(data, V)

        @test norm(Σrpod - Σropinf, Inf) < 1e1
        @test norm(Σrpod.A - Σropinf.A) < 1e0
    end

    @testset "phdmd_datamatrices" begin
        T, Z = phdmd_datamatrices(data, Q)
        @test size(T) == (n+m, length(t))
        @test size(Z) == (n+m, length(t))
    end

    @testset "skew_proc" begin
        J = [0 1 2 3; -1 0 4 5; -2 -4 0 6; -3 -5 -6 0]
        T = reshape(1:20, 4, 5)
        Z = J * T

        Jr, err = PortHamiltonianModelReduction.skew_proc(T, Z)
        @test norm(Z - Jr * T) < 1e-2
    end

    @testset "phdmd_sdp" begin
        Σr, model = PortHamiltonianModelReduction.phdmd_sdp(data)
        @test norm(Σ - Σr, Inf) < 1e-5

        Σr, model = PortHamiltonianModelReduction.phdmd_sdp(data, Q)
        @test norm(Σ - Σr, Inf) < 1e-5
    end

    @testset "estimate_ham" begin
        Qr, err = estimate_ham(data.X, H)
        @test norm(Qr - Q) < 1e-8  
    end

    @testset "phdmd_init" begin
        Γ, W, err = phdmd_initial_guess(data, Q)
        Σr = phss(Γ, W, Q)
        @test norm(Σ - Σr, Inf) < 1e-10
    end

    @testset "phdmd_fgm" begin
        T, Z = phdmd_datamatrices(data, Q)
        Γ0 = zeros(n+m, n+m)
        W0 = I(n+m)
        norm(T'*Z - T'*(Γ0 - W0) * T)

        Γ, W, err = PortHamiltonianModelReduction.phdmd_fgm(T, Z, Γ0, W0)
        @test err[end] < 1e0

        Γ0, W0, _ =  phdmd_initial_guess(data, Q)
        Γ, W, err = PortHamiltonianModelReduction.phdmd_fgm(T, Z, Γ0, W0)
        @test err[end] < 1e-12
    end
end
