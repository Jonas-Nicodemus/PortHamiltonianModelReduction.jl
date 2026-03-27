using Test

using PortHamiltonianModelReduction

using LinearAlgebra, ControlSystemsBase
using PortHamiltonianSystems

@testset "test_projection.jl" begin
    J = [0. 1.; -1. 0.]
    R = [2. 0.; 0. 1.]
    Q = [1. 0.; 0. 1.]
    G = [6.; 0.;;]
    P = zero(G)
    S = [1.;;]
    N = zero(S)

    Σph = phss(J, R, Q, G, P, S, N)
    Σ = ss(Σph)
    V = [1; 0;;]
    r = size(V, 2)

    @testset "pgprojection" begin
        # Galerkin projection
        Σr = pgprojection(Σ, V)
        @test size(Σr.A) == (r,r)

        # Petrov-Galerkin projection
        W = Σph.Q * V / (V' * Σph.Q * V)
        Σphr = pgprojection(Σph, V, W)
        @test size(Σphr.J) == (r,r)
        @test isposdef(Σphr.R) 
        @test isposdef(Σphr.Q)

        Σr2 = pgprojection(Σ, V, W)
        @test norm(Σr2.A - (Σphr.J - Σphr.R) * Σphr.Q) < 1e-10
    end
end
