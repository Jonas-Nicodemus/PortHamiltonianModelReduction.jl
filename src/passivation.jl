"""
    passivate(Σ::StateSpace, method=:klap, args...; kwargs...) -> Σp, res

Passivate the system `Σ` using the method `method`. The available methods are:

- `:klap`: KLAP optimization [NVGU25](@cite)
- `:lmi`: LMI optimization [GS21](@cite)
- `:lmi_tp`: LMI optimization with trace parametrization [Dum02, CPS04](@cite)

The remaining arguments `args` and keyword arguments `kwargs` are passed to the corresponding passivation function.
"""
function passivate(Σ::StateSpace, method=:klap, args...; kwargs...)
    if method == :klap
        return klap(Σ, args...; kwargs...)
    elseif method == :lmi
        return passivate_lmi(Σ, args...; kwargs...)
    elseif method == :lmi_tp
        return passivate_lmi_tp(Σ, args...; kwargs...)
    else
        error("Unknown passivation method $method")
    end
end

"""
    klap(Σ::StateSpace; L0=L0(Σ), M=M(Σ), P=gram(Σ, :c); recycl=:schur, restart=false, α=1e-8, ε=1e-4, verbose=true, kwargs...) -> Σp, res

Passivates a system `Σ` using KLAP [NVGU25](@cite). The optimization problem is solved using LBFGS.
"""
function klap(Σ::StateSpace, L0=L0(Σ), M=M(Σ), P=gram(Σ, :c); recycl=:schur, restart=false, α=1e-8, ε=1e-4, verbose=true, kwargs...)
    if isdiag(Σ.A)
        @info "Diagonal Σ.A detected, using specialized fg! implementation"
        Ã = diag(Σ.A) .+ diag(Σ.A)'
        d = Optim.only_fg!((f,G,L) -> fg!(f, G, L, Σ, Diagonal(Σ.A), Ã, P, M))
    elseif recycl == :schur
        S = schur(Σ.A)
        d = Optim.only_fg!((f,G,L) -> fg!(f, G, L, Σ, S, P, M))
    elseif recycl === nothing
        @info "No recycling method specified, using generic fg! implementation"
        d = Optim.only_fg!((f,G,L) -> fg!(f, G, L, Σ, P, M))
    else 
        @error "Recycling method $recycl not recognized"
    end
    
    res = Optim.optimize(d, L0, LBFGS(),  
        Optim.Options(;
            # f_reltol = 1e-12,
            g_abstol = 1e-12,
            g_tol = 1e-12,
            show_trace = verbose,
            show_every = 1,
            extended_trace = false,
            store_trace = false,
            iterations = 10_000,
            kwargs...
        ))

    if verbose
        @info "Optimization result: $(res)"
    end

    L = Optim.minimizer(res)
    Σp = PortHamiltonianModelReduction.Σp(Σ, L, M)

    if restart 
        Y = PortHamiltonianModelReduction.Y(Σp, L, M)
        @info "Checking eigvals of Y ..."
       
        if norm(real(eigvals(Y))) / norm(Σ.A) > ε
            @info "Possible stuck in local minimum, performing unconstrained gradient step ..."
           
            Gc = 2*(Σp.C-Σ.C)*P
            Σtmp = ss(Σ.A, Σ.B, Σp.C - α*Gc, Σ.D)
            L, ΔD = klap_inital_guess(Σtmp)

            if ΔD > 0
                @warn "The system was not passive after the unconstrained gradient step. Perturbation ΔD = $ΔD was used.
                Consider decreasing α or increasing ε."
            end

            return klap(Σ, L, M, P; recycl=recycl, restart=restart, α=α, verbose=verbose, kwargs...)..., res
        else
            @info "Global minimum detected, no restart needed."
        end
    end

    return Σp, res
end

"""
    klap_inital_guess(Σ, ΔD=0.0; ε=1e-8) -> L0, ΔD

Computes an initial guess for KLAP [NVGU25](@cite). The initial guess is computed by perturbing the feedthrough matrix to achieve a passive realization.
Then the perturbed system is used to compute the initial guess. The perturbation `ΔD` can be specified, otherwise it is computed using `ΔD(Σ)`.
"""
function klap_inital_guess(Σ, ΔD=0.0; ε=1e-8)
    ΔD = ΔD > 0 ? ΔD : PortHamiltonianModelReduction.ΔD(Σ, ε=ε)

    Σpert = ss(Σ.A, Σ.B, Σ.C, Σ.D + ΔD * I)

    L0 = zeros(Σ.nx, Σ.nu)
    M = PortHamiltonianModelReduction.M(Σpert)
    try 
        L0 .= L(Σpert, M)
    catch
        @warn "Initial guess computation failed, trying with bigger perturbation ΔD = $(10*ΔD) ..."
        return klap_inital_guess(Σ, 10*ΔD)
    end

    return L0, ΔD
end

function ΔD(Σ; ε=0)
    Λmin, _, _, _, _, _ = sampopov(Σ)
    ΔD = maximum([0, -minimum(Λmin)/2]) + ε    
    return ΔD
end

function L0(Σ; ε=0)
    return klap_inital_guess(Σ, ε=ε)[1]
end

function L(Σ, M=M(Σ))
    Xmin = kypmin(Σ)
    return (Σ.C' - Xmin * Σ.B)*inv(M)
end

function M(Σ)
    return sqrt(Σ.D + Σ.D')
end

function J(Σ::StateSpace, L, M=sqrt(Σ.D + Σ.D'), P=gram(Σ, :c))
    Ce = Σ.C - Cr(Σ, L, M)
    return J(Ce, P)
end

function J(Ce, P)
    return real(tr(Ce * (P * Ce')))
end

function X(F::LinearAlgebra.Factorization, L)
    return lyapc(F, L*L'; adj=true)
end

function X(Ã, U, U⁻¹, L)
    return lyapc(Ã, U, U⁻¹, L*L'; adj=true)
end

function X(A::LinearAlgebra.Diagonal, Ã, L)
    return lyapc(A, L*L', Ã; adj=true)
end

function X(Σ, L)
    U = plyapc(Array(Σ.A'), L)
    return U * U'
end

function Y(Σ, X)
    return Σ.A - Σ.B * inv(Σ.D + Σ.D') * (Σ.C - Σ.B' * X)
end

function Y(Σ, L, M)
    return Σ.A - Σ.B * inv(Σ.D + Σ.D') * M * L'
end

function Cr(Σ, L, M=sqrt(Σ.D + Σ.D'))
    return Σ.B' * X(Σ, L) + M * L'
end

function Cr(Σ, F::LinearAlgebra.Factorization, L, M=sqrt(Σ.D + Σ.D'))
    return Σ.B' * X(F, L) + M * L'
end

function Cr(Σ, Ã, U, U⁻¹, L, M=sqrt(Σ.D + Σ.D'))
    return Σ.B' * X(Ã, U, U⁻¹, L) + M * L'
end

function Cr(Σ, A::LinearAlgebra.Diagonal, F, L, M=sqrt(Σ.D + Σ.D'))
    return Σ.B' * X(A, F, L) + M * L'
end

function Σp(Σ, L, M=sqrt(Σ.D + Σ.D'))
    return ss(Σ.A, Σ.B, Cr(Σ, L, M), Σ.D)
end

function ∇J(Σ, L, M=sqrt(Σ.D + Σ.D'), P=gram(Σ, :c))
    Cr = PortHamiltonianModelReduction.Cr(Σ, L, M) 
    Ce = Cr - Σ.C
    XX = lyapc(Σ.A, 2*hermitianpart(Σ.B*Ce*P)) 
    return 2XX*L + 2*P*Ce'*M
end

function fg!(f, G, L, Σ, P, M)
    Ce = Cr(Σ, L, M) - Σ.C

    if G !== nothing
        XX = lyapc(Σ.A, 2*hermitianpart(Σ.B*Ce*P))
        G .= 2XX*L + 2*P*Ce'*M
    end
    if f !== nothing
      return J(Ce, P)
    end
end

function fg!(f, G, L, Σ, A::LinearAlgebra.Diagonal, F, P, M)
    Ce = Cr(Σ, A, F, L, M) - Σ.C

    if G !== nothing
        XX = lyapc(A, 2*hermitianpart(Σ.B*Ce*P), F)
        G .= 2XX*L + 2*P*Ce'*M
    end
    if f !== nothing
      return J(Ce, P)
    end
end

function fg!(f, G, L, Σ, F::LinearAlgebra.Factorization, P, M)
    Ce = Cr(Σ, F, L, M) - Σ.C

    if G !== nothing
        XX = lyapc(F, 2*hermitianpart(Σ.B*Ce*P))
        G .= 2XX*L + 2*P*Ce'*M
    end
    if f !== nothing
      return J(Ce, P)
    end
end

function MatrixEquations.lyapc(S::LinearAlgebra.Schur, C::AbstractMatrix; adj=false)
    X = MatrixEquations.utqu(C,S.Z)
    MatrixEquations.lyapcs!(S.T, X; adj=adj)
    MatrixEquations.utqu!(X,S.Z')
    return X
end

function MatrixEquations.lyapc(A::Diagonal, C::AbstractMatrix, F::AbstractMatrix = diag(A) .+ diag(A)'; adj=false)
    return adj ? -C ./ conj(F) : -C ./ F 
end

"""
    passivate_lmi_tp(Σ::StateSpace; kwargs...) -> Σp, model

Solves the passivation problem for a given state-space system `Σ` using the positive real LMI constraints with trace parametrization [Dum02, CPS04](@cite).
"""
function passivate_lmi_tp(Σ::StateSpace; kwargs...)
    # Dimensions
    n = Σ.nx
    m = Σ.nu

    Qc = cholesky(gram(Σ, :c)).U

    ζ = zeros(n, n, n, m)
    for i in 1:n
        for j in 1:m
            ζ[:,:,i,j] = lyapc(Σ.A, hermitianpart(Σ.B[:,j] * I[1:n, i]'))
        end
    end

    # Passivity Enforcement
    model = Model(Hypatia.Optimizer)
    for kwarg in keys(kwargs)
        set_optimizer_attribute(model, String(kwarg), kwargs[kwarg])
    end

    @variable(model, W[1:n+m, 1:n+m], PSD)
    fix.(W[n+1:end, n+1:end], Σ.D + Σ.D')

    Ψ1 = W[1:n, 1:n]
    Ψ2 = W[1:n, n+1:end]

    @variable(model, t)
    @variable(model, Δ[1:m, 1:n])
    @constraint(model, c[i=1:n, j=1:m], Σ.C[j,i] + Δ[j,i] == tr(ζ[:,:,i,j] * Ψ1) + Ψ2[i, j])
    @constraint(model, [t; vec(Δ * Array(Qc'))] in SecondOrderCone())
    @objective(model, Min, t)

    JuMP.optimize!(model)

    return ss(Σ.A, Σ.B, Σ.C + value.(Δ), Σ.D), model;
end

"""
    passivate_lmi(Σ::StateSpace; kwargs...) -> Σp, model

Solves the passivation problem for a given state-space system `Σ` using the positive real LMI constraints [GS21](@cite).
"""
function passivate_lmi(Σ::StateSpace; kwargs...)
    # Dimensions
    n = Σ.nx
    m = Σ.nu
    
    # Passivity Enforcement
    Wc = gram(Σ, :c)
    Qc = cholesky(Wc).U
    
    model = Model(Hypatia.Optimizer)
    for kwarg in keys(kwargs)
        set_optimizer_attribute(model, String(kwarg), kwargs[kwarg])
    end

    @variable(model, X[1:n, 1:n], PSD)
    @variable(model, Δ[1:m, 1:n])

    C = Σ.C + Δ
    W = -[Σ.A'*X+X'*Σ.A X*Σ.B-C'; Σ.B'*X-C -Σ.D-Σ.D']

    @variable(model, t)
    @objective(model, Min, t)
    @constraint(model, [t; vec(Δ * Array(Qc'))] in SecondOrderCone())
  
    @constraint(model, W in PSDCone())
    
    JuMP.optimize!(model)
    
    return ss(Σ.A, Σ.B, value.(C), Σ.D), model
end
