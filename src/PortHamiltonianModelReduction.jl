module PortHamiltonianModelReduction

using LinearAlgebra, SkewLinearAlgebra, VectorizationTransformations
using ControlSystemsBase, MatrixEquations
using NLSolversBase, Optim, LineSearches, JuMP, Hypatia, COSMO
using PortHamiltonianSystems, QuadraticOutputSystems

export tddata
export opinf, pgopinf, pod, podg, podpg, opinf_datamatrices, phdmd, estimate_ham, phdmd_initial_guess, phdmd_datamatrices
export irka, phirka, bt, prbt, passivate, klap, matchnrg, ephminreal, hdss
export chirp, sawtooth, prbs, step # inputsignals.jl
export pgprojection # projection.jl

include("types/TimeDomainData.jl")
include("energymatching.jl")
include("inputsignals.jl")
include("passivation.jl")
include("phirka.jl")
include("irka.jl")
include("phdmd.jl")
include("prbt.jl")
include("projection.jl")

end
