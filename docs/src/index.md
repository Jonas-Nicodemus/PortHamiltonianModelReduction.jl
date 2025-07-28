```@meta
CurrentModule = PortHamiltonianModelReduction
```

# PortHamiltonianModelReduction

Documentation for [PortHamiltonianModelReduction](https://github.com/Jonas-Nicodemus/PortHamiltonianModelReduction.jl).

A Julia package implementing model reduction techniques for port-Hamiltonian systems.

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

## References
```@bibliography
```
