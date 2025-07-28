using PortHamiltonianModelReduction
using Test, Aqua
@testset "PortHamiltonianModelReduction.jl" begin
    Aqua.test_all(PortHamiltonianModelReduction; piracies=false)

    include("test_energymatching.jl")
    include("test_inputsignals.jl")
    include("test_irka.jl")
    include("test_passivation.jl")
    include("test_phdmd.jl")
    include("test_phirka.jl")
    include("test_prbt.jl")
    include("test_projection.jl")
end
