"""
    Σr = prbt(Σ, r; Lx=prgrampd(Σ, :o), Ly=prgrampd(Σ, :c))

Reduces the state dimension of the system `Σ` to `r` using positive real balanced truncation (PRBT) [DP84](@cite).
The cholesky factors of the positive real Gramians can passed as optional arguments `Lx` and `Ly`, in order to avoid recomputation. 
`Σ` can be a `StateSpace` or a `PortHamiltonianStateSpace`, which will then also be the return type. In the case of a `PortHamiltonianStateSpace`, 
for the conversion to the ROM the minimal solution of the KYP inequality is used as the Hessian of the Hamiltonian.
"""
function prbt(Σ::StateSpace, r; Lx=prgrampd(Σ, :o), Ly=prgrampd(Σ, :c))
    return bt(Σ, r; Lx=Lx, Ly=Ly)
end

function prbt(Σph::PortHamiltonianStateSpace, r; Lx=prgrampd(Σph, :o), Ly=prgrampd(Σph, :c))
    Σr = prbt(ss(Σph), r; Lx=Lx, Ly=Ly)
    return phss(Σr) 
end

"""
    Σr = bt(Σ::StateSpace, r; Lx=grampd(Σ, :o), Ly=grampd(Σ, :c)')

Reduces the state dimension of the system `Σ` to `r` using standard square root balanced truncation (see for instance [BS21](@cite)).
The cholesky factors of the Gramians can passed as optional arguments `Lx` and `Ly`.
The default values are the cholesky factors of the observability and controllability Gramians.
"""
function bt(Σ::StateSpace, r; Lx=grampd(Σ, :o), Ly=grampd(Σ, :c)')
    U, σ, Z = svd(Ly * Lx')
    U = U[:, 1:r]
    Z = Z[:, 1:r]
    σ = σ[1:r]
    V = Ly' * U * Diagonal(σ .^ (-1 / 2))
    W = Lx' * Z * Diagonal(σ .^ (-1 / 2))
    Ar = W' * Σ.A * V
    Br = W' * Σ.B
    Cr = Σ.C * V

    return ss(Ar, Br, Cr, Σ.D)
end
