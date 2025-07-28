using Test

using PortHamiltonianModelReduction

using LinearAlgebra, VectorizationTransformations, ControlSystemsBase, FiniteDifferences
using QuadraticOutputSystems, PortHamiltonianSystems

@testset "test_energymatching.jl" begin
    
    @testset "minreal" begin
        U = UpperTriangular([1.0 2.0 3.0; 0.0 4.0 5.0; 0.0 0.0 6.0])
    
        J = U' - U 
        R = U'*U
    
        G = [1; 0; 0]
        P = zero(G)
        S = [1.;;]
        N = zero(S)
        
        @testset "truncno" begin
            Q = U[1:2,:]'*U[1:2,:]
            Σph = phss(J, R, Q, G, P, S, N)
            Σphr = PortHamiltonianModelReduction.truncno(Σph)
            @test ss(Σphr).nx == 2
            @test norm(Σph - Σphr) < 1e-6
        end
            
        @testset "truncnc" begin
            Q = U'*U
            Σph = phss(J, R, Q, G, P, S, N)
            Σphr = PortHamiltonianModelReduction.truncnc(Σph; ϵ=1e-4)
            @test ss(Σphr).nx == 2
            @test norm(Σph - Σphr) < 1e-2
        end
    
        @testset "ephminreal" begin
            Q = U'*U
            Σph = phss(J, R, Q, G, P, S, N)
            Σphr = ephminreal(Σph)
            @test norm(Σph - Σphr) < 1e-2
        end
    end

    @testset "test_matchnrg_barrier" begin
        A = [-2. 1.; -1. -1.]
        B = [6.; 0;;]
        C = B'
        D = [1;;]
    
        M = 1//2*vec([1 0; 0 1])'
        
        Σ = qoss(A, B, M)
        Σr = ss(A[1:1,1:1], B[1:1,1:1], C[1:1,1:1], D)
    
        Xmin = kypmin(Σr)
        Xmax = kypmax(Σr)
        X0 = 1/2 * (Xmin + Xmax)
        M0 = 1/2 * vec(X0)'
        x = 1/2 * vech(X0) 
    
        @testset "objective" begin
            f, g = PortHamiltonianModelReduction.objective(Σ, Σr)
    
            @testset "f" begin
                @test f(x) ≈ norm(Σ - qoss(Σr.A, Σr.B, vec(M0)'))^2
            end
    
            @testset "g" begin
                @test norm(grad(central_fdm(5,1), f, x)[1] - g(x)) < 1e-10
            end
        end
    
        @testset "constraint" begin
            f, g = PortHamiltonianModelReduction.constraint(Σr)
    
            @testset "g" begin
                @test norm(grad(central_fdm(5,1), f, x)[1] - g(x)) < 1e-10
            end
        end
    
        @testset "combined" begin
            fo, go = PortHamiltonianModelReduction.objective(Σ, Σr)
            fc, gc = PortHamiltonianModelReduction.constraint(Σr)
    
            α = 1e-3
            f = (x) -> fo(x) + α * fc(x)
            g!(g, x) = begin
                g .= go(x) + α * gc(x)
            end
    
            @testset "g" begin
                g = zeros(length(x))
                g!(g, x)
                @test norm(grad(central_fdm(5,1), f, x)[1] - g) < 1e-10
            end
        end
    
        @testset "matchnrg_barrier" begin
            Σphr = PortHamiltonianModelReduction.matchnrg_barrier(Σ, Σr)
            @test Σphr.Q ≈ [160/169;;]
            @test norm(Σ - hdss(Σphr))^2 ≈ 19 + 81/4 * (160/169)^2 - 2 * 3240/169 * (160/169) 
    
            Σr = ss(A, B, C, D)
            Σr0 = phss(Σr)
            Σphr = PortHamiltonianModelReduction.matchnrg_barrier(Σ, Σr; Σr0=Σr0)
            @test norm(Σphr.Q - 2*unvec(M)) < 1e-6
        end
    end

    @testset "test_matchnrg_sdp" begin
        A = [-2. 1.; -1. -1.]
        B = [6.; 0;;]
        C = B'
        D = [1;;]
        Q = [1. 0.; 0. 1.]
        
        Σ = phss(ss(A, B, C, D), Q)
        Σr = phss(ss(A[1:1,1:1], B[1:1,1:1], C[1:1,1:1], D), Q[1:1,1:1])  
            
        @testset "matchnrg_sdp" begin
            for solver in [:Hypatia, :COSMO]
                Σphr = matchnrg(Σ, Σr; solver=solver)
                @test norm(Σphr.Q - [160/169;;]) < 1e-3
                @test norm(hdss(Σ) - hdss(Σphr))^2 ≈ 19 + 81/4 * (160/169)^2 - 2 * 3240/169 * (160/169) atol = 1e-3
            end
        end
    end

    @testset "test_matchnrg_are" begin
        A = [-2. 1.; -1. -1.]
        B = [6.; 0;;]
        C = B'
        D = [1;;]
    
        M = 1//2*vec([1 0; 0 1])'
        
        Σ = qoss(A, B, M)
        Σr = ss(A[1:1,1:1], B[1:1,1:1], C[1:1,1:1], D)  
            
        @testset "matchnrg_are" begin
            Σphr = PortHamiltonianModelReduction.matchnrg_are(Σ, Σr)
            @test Σphr.Q ≈ kypmin(Σr)
        end
    end
end
