"""
    pgprojection(Σph::PortHamiltonianStateSpace, V::Matrix, W::Matrix=Σph.Q * V / (V' * Σph.Q * V))
    pgprojection(Σ::StateSpace, V::Matrix, W::Matrix=V)
    pgprojection(data::TimeDomainData, W::Matrix)

Applies a (Petrov-)Galerkin projection to a pH system, LTI system or time-domain data. In the pH case, the `W` matrix has to be chosen
such that the structure is preserved. The default choice is `W = Σph.Q * V / (V' * Σph.Q * V)`, which is proposed in [GPBS12](@cite).
"""
function pgprojection(Σph::PortHamiltonianStateSpace, V::Matrix, W::Matrix=Σph.Q * V / (V' * Σph.Q * V))
    return phss(skewhermitian(W' * Σph.J * W), hermitianpart(W' * Σph.R * W), hermitianpart(V' * Σph.Q * V), W' * Σph.G, W' * Σph.P, Σph.S, Σph.N)
end

function pgprojection(Σ::StateSpace, V::Matrix, W::Matrix=V)
   return ss(W' * Σ.A * V, W' * Σ.B, Σ.C * V, Σ.D)
end

function pgprojection(data::TimeDomainData, W::Matrix)
    return tddata(W' * data.Ẋ, W' * data.X, data.U, data.Y, data.t)
end
