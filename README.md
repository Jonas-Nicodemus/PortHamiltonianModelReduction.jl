# PortHamiltonianModelReduction

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://Jonas-Nicodemus.github.io/PortHamiltonianModelReduction.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://Jonas-Nicodemus.github.io/PortHamiltonianModelReduction.jl/dev/)
[![Build Status](https://github.com/Jonas-Nicodemus/PortHamiltonianModelReduction.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Jonas-Nicodemus/PortHamiltonianModelReduction.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/Jonas-Nicodemus/PortHamiltonianModelReduction.jl/graph/badge.svg?token=BBUDG0AHZC)](https://codecov.io/gh/Jonas-Nicodemus/PortHamiltonianModelReduction.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

A Julia package implementing model reduction techniques for port-Hamiltonian (pH) systems.

## Installation

Install with the Julia package manager [Pkg](https://pkgdocs.julialang.org/):
```julia
pkg> add PortHamiltonianModelReduction # Press ']' to enter the Pkg REPL mode.
```
or
```julia
julia> using Pkg; Pkg.add("PortHamiltonianModelReduction")
```

For the best experience, we recommend also installing [`PortHamiltonianSystems.jl`](https://github.com/Jonas-Nicodemus/PortHamiltonianSystems.jl) and [`ControlSystems.jl`](https://github.com/JuliaControl/ControlSystems.jl). Furthermore, for accessing (pH) Benchmark systems check out [`PortHamiltonianBenchmarkSystems.jl`](https://github.com/Algopaul/PortHamiltonianBenchmarkSystems.jl).

## Documentation

Some available commands are:
##### Interpolation based methods
`irka, phirka`
##### Balancing methods
`bt, prbt`
##### Passivation methods
`klap`
##### Data driven methods
`phdmd, opinf, pod`
##### Energy matching
`matchnrg`

### Example

```julia
import Random
Random.seed!(1234) # for reproducibility

using LinearAlgebra, ControlSystemsBase
using PortHamiltonianSystems, PortHamiltonianModelReduction
using PortHamiltonianBenchmarkSystems

J, R, Q, B = construct_system(SingleMSDConfig());
Σ = phss(J, R, Q, B, zero(B), 1e-6*I(2), zeros(2,2)); # create pH system with the artificial feedthrough term 1e-6*I(2)
size(Σ.G) # (100, 2)

Σ = phminreal(Σ); # structure preserving minimal realization
size(Σ.G) # (73, 2)

r = 10 # reduced order

Σr1 = phirka(Σ, r);
norm(Σ - Σr1) # 0.08325552559202813

Σr2 = prbt(Σ, r);
norm(Σ - Σr2) # 0.0031763828389874994

Σr3 = irka(ss(Σ), r);
norm(Σ - Σr3) # 0.0016571113588666688
ispassive(Σr3) # false

Σr4, _ = klap(Σr3);
norm(Σ - Σr4) # 0.0016577342723135616
ispassive(Σr4) # true
```
