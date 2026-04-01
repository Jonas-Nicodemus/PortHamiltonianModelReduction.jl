import JuMP: value

"""
    Σr, res = opinf(data::TimeDomainData, Vr::Matrix)

Computes a ROM `Σr` from time-domain data via (Galerkin) operator inference (OpInf) [PW16](@cite). 
The error of the OpInf problem is returned as `res`.
The data is projected via the matrix `V`, which is typically a POD basis.
"""
function opinf(data::TimeDomainData, V::Matrix)
    return opinf(pgprojection(data, V))
end

"""
    Σr, res = pgopinf(data::TimeDomainData, Vr::Matrix, Wr::Matrix)

Computes a ROM `Σr` from time-domain data via Petrov-Galerkin operator inference (PG-OpInf). 
The error of the OpInf problem is returned as `res`.
The data is projected via the oblique projection matrix `inv(W'V)W'` for given projection matrices `V` and `W`.
"""
function pgopinf(data::TimeDomainData, V::Matrix, W::Matrix)
    return opinf(pgprojection(data, V, W))
end

function opinf(data::TimeDomainData)
    n = size(data.X, 1)
    T, Z = opinf_datamatrices(data)
    O = Z * pinv(T)

    res = norm(Z - O * T)

    return ss(O[1:n, 1:n], O[1:n, n+1:end], O[n+1:end, 1:n], O[n+1:end, n+1:end]), res
end

function opinf_datamatrices(data::TimeDomainData)
    return [data.X; data.U], [data.Ẋ; data.Y]
end

"""
    podbasis(X::AbstractMatrix, r::Int; kwargs...)
    podbasis(F::SVD, r::Int; kwargs...)

Returns the POD basis via the truncated singular value decomposition of a data matrix `X`.
Alternatively, it can be cast on a `F::SVD` object, such that the SVD is not recomputed.
"""    
function pod(args...; kwargs...)
    F = tsvd(args...; kwargs...)
    return F.U
end

"""
    Σr = podg(Σ, X, r)
    Σr = podg(Σ, F, r)
    Σr = podg(Σ, V)
    Σr = podpg(Σ, V, W)

Returns a ROM `Σr` of size `r` for the system `Σ` using the POD (Petrov-)Galerkin method.
"""
function podg(Σ, X, r)
    Vr = pod(X, r)
    return podg(Σ, Vr)
end

function podg(Σ::StateSpace, V::Matrix)
    return pgprojection(Σ, V, V)
end

function podpg(Σph::PortHamiltonianStateSpace, V::Matrix)
    return pgprojection(Σph, V, Σph.Q * V / (V' * Σph.Q * V))
end

function podpg(Σ::StateSpace, V::Matrix, W::Matrix)
    return pgprojection(Σ, V, W) 
end

"""
    Σph, err = phdmd(data::TimeDomainData, H::AbstractVector; kwargs...)
    Σph, err = phdmd(data::TimeDomainData, Q::AbstractMatrix; kwargs...)
    
Computes a pH system `Σph` that approximates given time-domain `data` and a candidate `Q` for the Hessian of the Hamiltonian or measurements `H` of the Hamiltonian using the pHDMD method [MNU23](@cite).
"""
function phdmd(data::TimeDomainData, H::AbstractVector; kwargs...)
    Q, _ = estimate_ham(data, H; kwargs...)
    return phdmd(data, Q; kwargs...)
end

function phdmd(data::TimeDomainData, Q::AbstractMatrix; kwargs...)
    T, Z = phdmd_datamatrices(data, Q)
    F = svd(T)
    _, W, _ = phdmd_initial_guess(F, Z)
    Γ, _ = skew_proc(F, Z + (W * T))
    Γ, W, err = phdmd_fgm(T, Z, Γ, W; kwargs...)

    return phss(Γ, W, Q), err
end

"""
    Σph, model = phdmd_sdp(data::TimeDomainData; optimizer=Clarabel.Optimizer, kwargs...)
    Σph, model = phdmd_sdp(data::TimeDomainData, Q::AbstractMatrix; optimizer=Clarabel.Optimizer, kwargs...)

Computes a pH system `Σph` that approximates given time-domain `data` using semidefinite programming.
Either the Hessian of the Hamiltonian `Q` is provided or also learned from the data.
"""
function phdmd_sdp(data::TimeDomainData; optimizer=Clarabel.Optimizer, kwargs...)
    n = size(data.X, 1)
    m = size(data.U, 1)

    model = Model(optimizer)
    for kwarg in keys(kwargs)
        set_optimizer_attribute(model, String(kwarg), kwargs[kwarg])
    end
    @variable(model, W[1:n+m, 1:n+m] in PSDCone())
    @variable(model, Γ[1:n+m, 1:n+m] in SkewSymmetricMatrixSpace())
    @variable(model, H[1:n, 1:n] in PSDCone())
   
    @variable(model, res)
    @constraint(model, [res; vec([H*data.Ẋ; -data.Y] - (Γ-W)*[data.X; data.U])] in SecondOrderCone());
    @objective(model, Min, res)
    JuMP.optimize!(model)

    return phss(value.(Γ), value.(W), inv(value.(H))), model
end

function phdmd_sdp(data::TimeDomainData, Q::AbstractMatrix; optimizer=Clarabel.Optimizer, kwargs...)
    Γ, W, model = phdmd_sdp(phdmd_datamatrices(data, Q)...; kwargs...)

    return phss(Γ, W, Q), model
end

function phdmd_sdp(T::AbstractMatrix, Z::AbstractMatrix; optimizer=Clarabel.Optimizer, kwargs...)
    n = size(T, 1)

    model = Model(optimizer)
    for kwarg in keys(kwargs)
        set_optimizer_attribute(model, String(kwarg), kwargs[kwarg])
    end
    @variable(model, W[1:n, 1:n] in PSDCone())
    @variable(model, Γ[1:n, 1:n] in SkewSymmetricMatrixSpace())
   
    @variable(model, res)
    @constraint(model, [res; vec(Z - (Γ-W)*T)] in SecondOrderCone());
    @objective(model, Min, res)
    JuMP.optimize!(model)

    return skewhermitian(value.(Γ)), hermitianpart(value.(W)), model
end

function phdmd_datamatrices(data::TimeDomainData, Q::AbstractMatrix)
    T = [hermitianpart(Q) * data.X; data.U]
    Z = [data.Ẋ; -data.Y]
    return T, Z
end

"""
    Q, err = estimate_ham(data::TimeDomainData, H::AbstractVector; kwargs...)

Estimates the Hamiltonian Hessian `Q` from time-domain `data` and measurements `H` of the Hamiltonian.
"""
function estimate_ham(data::TimeDomainData, H::AbstractVector; kwargs...)
    return estimate_ham(data.X, H; kwargs...)
end

function estimate_ham(X::AbstractMatrix, H::AbstractVector; kwargs...)
    XX = 1//2 * vcat([kron(X[:,i]', X[:,i]') for i in 1:length(H)]...)
    XXr = XX * duplication_matrix(size(X, 1))

    qr = XXr \ H
    Qr = unvech(qr)

    if !isposdef(Qr)
        @warn "Qr is not positive definite with λmin = $(minimum(eigvals(Qr))), projecting to nearest psd matrix. 
        Consider using the sdp-based estimation of the Hamiltonian for better results, see `estimate_ham_sdp`."
        Qr = project_psd(Qr; eigtol=1e-8)
    end
    err = norm(XXr * vech(Qr) - H)
    
    return Qr, err
end

function estimate_ham_sdp(X::AbstractMatrix, H::AbstractVector; optimizer=Clarabel.Optimizer, kwargs...)
    n, N = size(X)

    # Precompute symmetric outer products (also only need upper triangle)
    outer = [Symmetric(X[:, i] * X[:, i]') for i in 1:N]

    model = Model(optimizer)
    for kwarg in keys(kwargs)
        set_optimizer_attribute(model, String(kwarg), kwargs[kwarg])
    end
    Q = @variable(model, [1:n, 1:n], PSD)
    
    # dot() on Symmetric matrices uses only upper triangle — avoids redundant ops
    residuals = [0.5 * dot(outer[i], Q) - H[i] for i in 1:N]
    
    @variable(model, t)
    @objective(model, Min, t)
    @constraint(model, [t; residuals] in SecondOrderCone())
    
    JuMP.optimize!(model)
    return value.(Q), JuMP.objective_value(model)
end

"""
    Γ, W, err = phdmd_initial_guess(data::TimeDomainData, Q::AbstractMatrix; kwargs...)
Computes an initial guess `Γ, W` for the pHDMD problem from time-domain data `data` and a candidate `Q` for the Hessian of the Hamiltonian (see [MNU23; Thm. 3.7](@cite)).
"""
function phdmd_initial_guess(data::TimeDomainData, Q::AbstractMatrix; kwargs...)
    return phdmd_initial_guess(phdmd_datamatrices(data, Q)...; kwargs...)
end

function phdmd_initial_guess(T::AbstractMatrix, Z::AbstractMatrix; kwargs...)
    F = svd(T)
    return phdmd_initial_guess(F, Z; kwargs...)
end

function phdmd_initial_guess(F::SVD, Z::AbstractMatrix; kwargs...)
    F = tsvd(F; kwargs...)
    Σ = Diagonal(F.S)

    Zr = Σ * F.U' * Z * F.V

    Γr = skewhermitian(Zr)
    Wr = project_psd(-Zr)

    Γ = skewhermitian(F.U * inv(Σ) * Γr * inv(Σ) * F.U')
    W = hermitianpart(F.U * inv(Σ) * Wr * inv(Σ) * F.U')
 
    err = norm(Matrix(F)' * Z - Matrix(F)' * (Γ - W) * Matrix(F))
    
    return Γ, W, err
end

"""
    Γ, err = skew_proc(T::AbstractMatrix, Z::AbstractMatrix; kwargs...)

Computes the analytic solution of the skew-symmetric Procrustes problem (see for instance [MNU23; Thm. 3.4](@cite)).
"""
function skew_proc(T::AbstractMatrix, Z::AbstractMatrix; kwargs...)
    F = svd(T)
    return skew_proc(F, Z; kwargs...)
end

function skew_proc(F::SVD, Z::AbstractMatrix; ε=1e-12)
    Fr = tsvd(F; ε=ε)
    Σ = Diagonal(Fr.S)

    n = size(Z, 1)
    r = length(Fr.S)

    Zr = F.U' * Z * F.V
    Z1 = Zr[1:r, 1:r]
    Z2 = Zr[r+1:end, 1:r]

    Φ = 1 ./ (F.S[1:r].^2 .+ (F.S[1:r].^2)')

    Γ = skewhermitian(F.U * [Φ .* (2*skewhermitian(Z1 * Σ)) -inv(Σ) * Z2'; Z2 * inv(Σ) zeros(n-r, n-r)] * F.U')

    err = norm(Z - Γ * Matrix(F))

    return Γ, err 
end

function phdmd_fgm(T::AbstractMatrix, Z::AbstractMatrix, Γ0::AbstractMatrix, W0::AbstractMatrix; 
    max_iter=10_000, x_reltol=1e-8, f_abstol=1e-8, f_reltol=1e-8, show_every=1, verbose=false)
    Γ = Γ0
    W = W0
    
    # Precomputations
    F = svd(T)
    TTt = hermitianpart(T * T')
    w = eigvals(TTt)
    l = maximum(w) # Lipschitz constant
    μ = minimum(w)
    q = μ / l

    β = zeros(max_iter)
    α = zeros(max_iter + 1)
    e = zeros(max_iter + 1)

    # Parameters and initialization
    α0 = 0.1  # Parameter of the FGM in (0,1) - can be tuned.

    Q = W
    α[1] = α0
    e[1] = norm(Z - (Γ - W) * T) / norm(Z)

    if verbose
        @info "Iter 0: f = $(e[1])"
    end

    for k in 1:max_iter
        # Previous iterate
        Γp = Γ
        Wp = W

        Z1 = Z + (W * T)
        # Solution of the skew-symmetric Procrustes
        Γ, _ = skew_proc(F, Z1) 

        Z2 = (Γ * T) - Z  # use Γ updated from Skew Procrustes
        # Projected gradient step from Y
        ∇ = (Q * TTt) - (Z2 * T')
        W = project_psd(Q - ∇ ./ l) # R value is updated here

        # FGM Coefficients
        α[k + 1] = (sqrt((α[k]^2 - q)^2 + 4 * α[k]^2) + (q - α[k]^2)) / 2
        β[k] = α[k] * (1 - α[k]) / (α[k] ^ 2 + α[k + 1])

        # Linear combination of iterates
        Q = W + β[k] * (W - Wp)

        e[k + 1] = norm(Z - (Γ - W) * T) / norm(Z)
        x_rel = (norm(Γp - Γ) / norm(Γ)) + (norm(Wp - W) / norm(W))
        f_abs = abs(e[k + 1] - e[k])
        f_rel = f_abs / e[k + 1]

        if verbose & (k % show_every == 0 || k == 1)
            @info "Iter $k: f = $(e[k + 1]), x_rel = $x_rel, f_abs = $f_abs, f_rel = $f_rel"
        end
       
        if x_rel <= x_reltol || f_abs <= f_abstol || f_rel <= f_reltol
            # || e[k+1] - e[k] < 0
            if verbose
                @info "Iter $k: f = $(e[k + 1]), x_rel = $x_rel, f_abs = $f_abs, f_rel = $f_rel"
                @info "Converged after $k iterations"
                @info "f = $(e[k+1])"
                @info  x_rel <= x_reltol ? "x_rel ≤ $x_reltol" : "x_rel ≰ $x_reltol"
                @info  f_rel <= f_reltol ? "f_rel ≤ $f_reltol" : "f_rel ≰ $f_reltol"
                @info  f_abs <= f_abstol ? "f_abs ≤ $f_abstol" : "f_abs ≰ $f_abstol"
            end
            e = e[1:k+1] # converged after i iterations, keep values till it converges
            break
        end
    end

    return Γ, W, e
end
