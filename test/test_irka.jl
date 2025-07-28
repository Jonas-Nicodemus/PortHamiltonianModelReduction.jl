using Test

using PortHamiltonianModelReduction
using LinearAlgebra, ControlSystemsBase

@testset "test_irka.jl" begin
    A = [-2. 1.; -1. -1.]
    B = [6.; 0.;;]
    C = B'
    D = [1.;;]
    
    Σ = ss(A, B, C, D)

    @testset "irka" begin
        @testset "r=n" begin
            r = 2
            Σr = irka(Σ, r)
            @test norm(Σ - Σr) < 1e-10
        end

        @testset "r=1" begin
            r = 1
            Σr = irka(Σ, r)
            @test norm(Σ - Σr) < 1e1
        end

        @testset "r=1" begin
            r = 1
            Σr = irka(Σ, r, 3)
            @test norm(Σ - Σr) < 1e1
        end
    end
end
