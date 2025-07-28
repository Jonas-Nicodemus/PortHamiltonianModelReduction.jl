using Test

using PortHamiltonianModelReduction

using LinearAlgebra, ControlSystemsBase, JuMP, FiniteDifferences
using PortHamiltonianSystems

@testset "test_passivation.jl" begin

    @testset "passivate_lmi" begin
        A = [-1 4; -2 -1];
        B = [1; 2];
        C = [1 0];
        D = 1//8;
        
        Σ = ss(A, B, C, D)
    
        @testset "passivate_lmi" begin
            Σp, model = PortHamiltonianModelReduction.passivate_lmi(Σ)
            X = value.(object_dictionary(model)[:X])
            @test isposdef(X)
            @test isposdef(kypmat(Σp, X) + 1e-6 * I)
            @test norm(Σp - Σ)^2 < 2e-1
        end

        @testset "passivate_lmi_tp" begin
            Σp, model = PortHamiltonianModelReduction.passivate_lmi_tp(Σ)
            W = value.(object_dictionary(model)[:W])
            @test isposdef(W + 1e-6 * I)
            @test norm(Σp - Σ)^2 < 2e-1

            Σpref, model = PortHamiltonianModelReduction.passivate_lmi(Σ)
            X = value.(object_dictionary(model)[:X])
            @test norm(Σp.C - Σpref.C) < 1e-6
            @test norm(W - kypmat(Σpref, X)) < 1e-6
        end
    end

    @testset "klap" begin
        A = [-1 4; -2 -1];
        B = [1; 2];
        C = [1 0];
        D = 1//8;
        
        Σ = ss(A, B, C, D)
    
        n, m = Σ.nx, Σ.nu
        M = sqrt(Σ.D + Σ.D')
        P = gram(Σ, :c)
    
        L = ones(typeof(Σ.A[1,1]), n, m)
    
        # precompute Schur of A
        F = schur(Σ.A)
    
        @testset "X" begin
            X = PortHamiltonianModelReduction.X(Σ, L)
            @test norm(Σ.A' * X + X * Σ.A + L * L') < 1e-10
            X = PortHamiltonianModelReduction.X(F, L)
            @test norm(Σ.A' * X + X * Σ.A + L * L') < 1e-10
        end
    
        @testset "Cr" begin
            Cr = PortHamiltonianModelReduction.Cr(Σ, L, M)
            @test norm(PortHamiltonianModelReduction.X(Σ, L) * Σ.B - Cr' + L*M) < 1e-10
            Cr = PortHamiltonianModelReduction.Cr(Σ, F, L, M)
            @test norm(PortHamiltonianModelReduction.X(F, L) * Σ.B - Cr' + L*M) < 1e-10
        end
    
        @testset "J" begin
            Σp = PortHamiltonianModelReduction.Σp(Σ, L, M)
            @test norm(PortHamiltonianModelReduction.J(Σ, L, M, P) - norm(Σ - Σp)^2) < 1e-12
        end
    
        @testset "∇J" begin
            f(L) = PortHamiltonianModelReduction.J(Σ, L, M, P)
            g(L) = PortHamiltonianModelReduction.∇J(Σ, L, M, P)
            @test norm(grad(central_fdm(5,1), f, L)[1] - g(L)) < 1e-8
        end
    
        @testset "fg!" begin
            f = Inf
            G = zero(L)
            
            f = PortHamiltonianModelReduction.fg!(f, G, L, Σ, P, M)
            @test norm(f - PortHamiltonianModelReduction.J(Σ, L, M, P)) < 1e-10
            @test norm(G - PortHamiltonianModelReduction.∇J(Σ, L, M, P)) < 1e-10
    
            f = Inf
            G = zero(L)
    
            f = PortHamiltonianModelReduction.fg!(f, G, L, Σ, F, P, M)
            @test norm(f - PortHamiltonianModelReduction.J(Σ, L, M, P)) < 1e-10
        end
    
        @testset "klap" begin
            L0 = [-1.5; 1.5;;]
            Σp, _ = klap(Σ, L0)
            @test norm(Σ - Σp)^2 ≈ 2.5
    
            L0 = [1.5; 1.5;;]
            Σp, _ = klap(Σ, L0)
            @test norm(Σ - Σp)^2 ≈ 0.12751294803962673
    
            L0 = [-1.5; 1.5;;]
            result = klap(Σ, L0; restart=true)
            @test length(result) == 3
            Σp = result[1]
            @test norm(Σ - Σp)^2 ≈ 0.12751294803962673
        end
    end
    
end
