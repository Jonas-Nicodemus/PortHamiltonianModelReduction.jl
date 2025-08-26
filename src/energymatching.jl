"""
    Σqo = hdss(J, R, Q, G, P, S, N)
    Σqo = hdss(Σph)

Converts a `PortHamiltonianStateSpace` to a `QuadraticOutputStateSpace` (Hamiltonian dynamic).
"""
function hdss(J, R, Q, G, P, S, N)
    A, B, _, _ = compose(J, R, Q, G, P, S, N)
    return qoss(A, B, 1/2*vec(Q)')
end

function hdss(Σph::PortHamiltonianStateSpace)
    return hdss(Σph.J, Σph.R, Σph.Q, Σph.G, Σph.P, Σph.S, Σph.N)
end

"""
    Σrem = matchnrg(Σ::PortHamiltonianStateSpace, Σr::PortHamiltonianStateSpace; solver=:BFGS, kwargs...)

Applies energy matching [HNSU25](@cite) to the ROM `Σr` to match the Hamiltonian dynamics of the original system `Σ`. 
If solver is `:Hypatia` or `:COSMO`, it uses semidefinite programming with the specified optimizer. 
If solver is `:BFGS`, it uses the barrier method with BFGS optimization. 
If solver is `:ARE`, the best solution of the ARE is returned.
"""
function matchnrg(Σ::PortHamiltonianStateSpace, Σr::PortHamiltonianStateSpace; solver=:BFGS, kwargs...)
    Σqo = hdss(Σ)

    if solver == :Hypatia
        Σrem = matchnrg_sdp(Σqo, ss(Σr), Hypatia.Optimizer; kwargs...)
    elseif solver == :COSMO
        Σrem = matchnrg_sdp(Σqo, ss(Σr), COSMO.Optimizer; max_iter=100000, eps_abs=1e-7, eps_rel=1e-7, kwargs...)
    elseif solver == :BFGS
        Σrem = matchnrg_barrier(Σqo, ss(Σr); Σr0 = Σr, kwargs...)
    elseif solver == :BR_BFGS
        Σrem = matchnrg_barrier(Σqo, ss(Σr); kwargs...)
    elseif solver == :ARE
        Σrem = matchnrg_are(Σqo, ss(Σr); kwargs...)
    else
        error("Unknown solver $solver")
    end

    return Σrem
end

"""
    Σph = ephminreal(Σph::PortHamiltonianStateSpace; kwargs...)

Computes a minimal realization of the (extended) pH system (see [HNSU25; Cor. 4.6](@cite))
"""
function ephminreal(Σph::PortHamiltonianStateSpace; kwargs...)
    if !isposdef(Σph.Q)
        Σph = truncno(Σph; kwargs...)
    end

    return truncnc(Σph; kwargs...)
end

function truncno(Σph::PortHamiltonianStateSpace; ϵ=1e-12)
    F = eigen(Σph.Q)
    i = findall(x->abs(x)>ϵ, F.values)
    V = F.vectors[:,i]
    return phss(skewhermitian(V'*Σph.J*V), hermitianpart(V'*Σph.R*V), hermitianpart(V'*Σph.Q*V), V'*Σph.G, V'*Σph.P, Σph.S, Σph.N)
end

function truncnc(Σph::PortHamiltonianStateSpace; ϵ=1e-12)
    L = cholesky(Σph.Q).L

    Σph1 = phss(skewhermitian(L'*Σph.J*L), hermitianpart(L'*Σph.R*L), I(size(L,1)), L'*Σph.G, L'*Σph.P, Σph.S, Σph.N)

    P = gram(Σph1, :c)
    F = eigen(hermitianpart(P), sortby=-)
    i = findall(x->x>ϵ, F.values)
    r = length(i)
    V = F.vectors[:,i]
    
    return phss(skewhermitian(V'*Σph1.J*V), hermitianpart(V'*Σph1.R*V), I(r), V'*Σph1.G, V'*Σph1.P, Σph.S, Σph.N)
end

"""
    Σphr = matchnrg_sdp(Σ::QuadraticOutputStateSpace, Σr::StateSpace; optimizer=COSMO.Optimizer, ε=1e-8, kwargs...)

Solves the energy matching problem using semidefinite programming.
"""
function matchnrg_sdp(Σ::QuadraticOutputStateSpace, Σr::StateSpace, optimizer=Hypatia.Optimizer; ε=1e-8, kwargs...)
    r = Σr.nx

    # Precompute
    Pr = gram(Σr, :c)
    Y =  sylvc(Σ.A, Σr.A', -Σ.B * Σr.B')
    h2 = norm(Σ)^2

    model = JuMP.Model(optimizer)
    for kwarg in keys(kwargs)
        set_optimizer_attribute(model, String(kwarg), kwargs[kwarg])
    end

    @variable(model, X[1:r, 1:r], PSD)
    Qr = 2 * X

    # Constraint
    W = [-Σr.A'*Qr-Qr*Σr.A Σr.C'-Qr*Σr.B;Σr.C-Σr.B'*Qr Σr.D+Σr.D']
    @constraint(model, W - ε * I >= 0, PSDCone())
    
    # Objective and optimize   
    @objective(model, Min, h2 + tr(Pr * X * Pr * X) - 2*tr(Y' * unvec(Σ.M) * Y * X))
    JuMP.optimize!(model)

    return phss(Σr, value.(Qr))
end

"""
    Σph = matchnrg_barrier(Σ, Σr; Σr0=nothing, kwargs...)

Solves the energy matching problem using the barrier method.
"""
function matchnrg_barrier(Σ::QuadraticOutputStateSpace, Σr::StateSpace; Σr0=nothing, kwargs...)

    if Σr0 === nothing
        Σr0 = matchnrg_are(Σ, Σr)
    end
    
    x = 1//2 * vech(Σr0.Q)

    eps=1e-8
    W = kypmat(Σr, Σr0.Q)
    λ = minimum(eigvals(W))
    if λ < 0.0
        eps = -λ + 1e-8
    end
    @info "Using eps = $eps"
    
    fo, go = objective(Σ, Σr)
    fc, gc = constraint(Σr; eps=eps)
        
    for α in exp10.(-3:-1:-15)
        
        # barrier method objective
        f = (x) -> fo(x) + α * fc(x)
        g!(g, x) = begin
            g .= go(x) + α * gc(x)
        end
        
        res = Optim.optimize(f, g!, x, Optim.BFGS(linesearch = LineSearches.BackTracking()),
            Optim.Options(
                g_abstol = 1e-16,
                allow_f_increases = true,
                iterations = 50000,
                show_trace = true,
                show_every = 100,
            ))
        @info "α = $α, f = $(res.minimum),\n $(res))"
        x = res.minimizer
    end

    Qr = 2 * unvech(x)
    
    return phss(Σr, Qr)
end

function objective(Σ::QuadraticOutputStateSpace, Σr::StateSpace)
    r = Σr.nx
    
    # precomputations
    P = gram(Σ, :c)
    h2 = tr(P*unvec(Σ.M)*P*unvec(Σ.M))
    Pr = gram(Σr, :c)
    Y =  sylvc(Σ.A, Σr.A', -Σ.B * Σr.B')
    Dr = duplication_matrix(r)

    function f(x)
        X = unvech(x)
        
        # value
        f = h2 + tr(Pr * X * Pr * X) - 2*tr(Y' * unvec(Σ.M) * Y * X)
        # f = tr(Pr * X * Pr * X) - 2*tr(Y' * unvec(Σ.M) * Y * X)

        return f
    end

    function g(x)
        X = unvech(x)
        
        # gradient
        G = 2 * (Pr * X * Pr - Y' * unvec(Σ.M) * Y)
        g = Dr' * vec(G)
    end

    # function h(x)
    #     return 2 * Dr' * kron(Pr, Pr) * Dr
    # end

    return f, g
end

function constraint(Σr::StateSpace; eps=1e-8)
    r, m = Σr.nx, Σr.nu 
    Dr = duplication_matrix(r)

    function fg(x)
        X = unvech(x)
        W = hermitianpart(kypmat(Σr, 2*X))
        W = W + eps * I(size(W, 1))
        
        if det(W) < 0.0 || any(eigvals(W) .< 0.0)
            f = Inf
            g = Inf * ones(length(x))
    
            return f, g
        else
            # value
            f = -log(det(W))

            # gradient
            A = [Σr.A Σr.B]
            B = [I(r); zeros(m, r)]
            G = 2 * A * (W \ B) + 2 * B' * (W \ A') 
            g = Dr' * vec(G)

            return f, g
        end
    end

    f(x) = fg(x)[1]
    g(x) = fg(x)[2]

    return f, g
end

"""
    Σphr = matchnrg_are(Σ::QuadraticOutputStateSpace, Σr::StateSpace; kwargs...)
    
Solves the energy matching problem where the solution set is replaced with solutions of the ARE.
"""
function matchnrg_are(Σ::QuadraticOutputStateSpace, Σr::StateSpace; kwargs...)
    Xmin = kypmin(Σr)
    Xmax = kypmax(Σr)
    
    Ar = Σr.A
    Br = Σr.B
    Cr = Σr.C
    Dr = Σr.D

    Amin = Ar + Br * inv(-Dr - Dr') * (-Br' * Xmin + Cr)
    @assert all(real(eigvals(Amin)) .<= 0)
    BVs = projectors(Amin)
    Δ = Xmin - Xmax

    Σrmin = qoss(Ar, Br, 1/2*vec(Xmin)')
    Σrmax = qoss(Ar, Br, 1/2*vec(Xmax)')
    
    Vmin = norm(Σ - Σrmin)
    Vmax = norm(Σ - Σrmax)
    
    Vstar = Inf
    Xstar = zero(Xmin)
    if Vmin < Vmax
        Xstar .= Xmin
        Vstar = Vmin
    else
        Xstar .= Xmax
        Vstar = Vmax
    end

    for Z in combinations(BVs)
        P = Z * inv(Z' * Δ * Z) * Z' * Δ
        X = hermitianpart(Xmin * P + Xmax * (I - P))
        Σqor = qoss(Σr.A, Σr.B, 0.5*vec(X)')
        Vn =  norm(Σ - Σqor)
        if Vn < Vstar
            Xstar .= X
            Vstar = Vn
        end
    end
    return phss(Σr, Xstar)
end

function projectors(Ap)
    D, Z = eigen(Ap)
    BVs = Vector{Matrix{Float64}}(undef, 0)
    i = 1
    while i < length(D)
        if isreal(D[i])
            push!(BVs, real(Z[:, i][:, :]))
            i += 1
        else
            push!(
                BVs,
                hcat(
                    real(Z[:, i]) / norm(real(Z[:, i])),
                    imag(Z[:, i]) / norm(imag(Z[:, i])),
                ),
            )
            i += 2
        end
    end
    return BVs
end

function combinations(BVs)
    n = length(BVs)
    AC = Vector{Matrix{Float64}}(undef, 0)
    for i = 1:n
        for j = i:n
            Z, _, _ = svd(hcat(BVs[i:j]...))
            push!(AC, Z)
        end
    end
    return AC
end
