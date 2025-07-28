using Test

using PortHamiltonianModelReduction

import PortHamiltonianModelReduction: step

@testset "test_inputsignals.jl" begin
    t = 0:0.01:1
    ti = 0.5

    u_chirp = chirp()
    u_prbs = prbs(t)
    u_sawtooth = sawtooth()
    u_step = step()

    for u1 in [u_chirp, u_prbs, u_sawtooth, u_step]
        u(x,t) = [u1(x, t)]

        @test isa(u(0, t[1]), Vector) 
        @test length(u(0, t[1])) == 1

        @test length(u.(0, t)) == length(t)
        @test size(stack(u.(0, t))) == (1, length(t))
    end
end