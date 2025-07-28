"""
    Σr = irka(Σ::StateSpace, r; tol=1e-3, max_iter=200)

Reduces the state dimension of the system `Σ` to `r` using the iterative rational Krylov algorithm (IRKA) [GAB08](@cite).
"""
function irka(Σ::StateSpace, r, num_runs; tol=1e-3, max_iter=200)
    h2 = Inf

    m = Σ.nu
    Σr = ss(zeros(r,r), zeros(r,r), zeros(r,r), zeros(r,m), zeros(r,m), zeros(m,m), zeros(m,m))

    for i in 1:num_runs
        Σrᵢ = irka(Σ, r; tol=tol, max_iter=max_iter)
        h2ᵢ = norm(Σ - Σrᵢ)
        if h2ᵢ < h2
            h2 = h2ᵢ
            Σr = Σrᵢ
            @info "Found a better rom with h2 = $h2"
        end
    end

    return Σr 
end

function irka(Σph::PortHamiltonianStateSpace, r, num_runs; kwargs...)
    return irka(ss(Σph), r, num_runs; kwargs...)
end

function irka(Σ::StateSpace, r; tol=1e-3, max_iter=50)
    V = Matrix(qr(randn(Σ.nx, r)).Q)
    W = Matrix(qr(randn(Σ.nx, r)).Q)
    s_prev = zeros(ComplexF64, r)
    sort_by = x -> (abs(x), angle(x))
    conv_crit = Inf

    Σr = project(Σ, V, W)
    s, b, c = interpolation_data(Σr)

    for i in 1:max_iter
        construct_V_and_W!(V, W, Σ, s, b, c)
        Σr = project(Σ, V, W)
        s, b, c = interpolation_data(Σr)

        s = sort(s, by=sort_by)
        conv_crit = norm(s-s_prev)/norm(s_prev)
        if conv_crit < tol
            @info "IRKA converged after $i iterations"
            break
        end
        s_prev = s
    end

    #  ensure that the ROM is stable
    if !isstable(Σr)
        @warn "IRKA did not converge to a stable ROM, restart irka ..."
        return irka(Σ, r; tol=tol, max_iter=max_iter);
    end
    
    if conv_crit >= tol
        @warn "IRKA not converged after $max_iter iterations"
    end

    return Σr
end

function interpolation_data(Σr::StateSpace)
    λ, U = eigen(Σr.A)
    s = -λ
    b = (U \ Σr.B)'
    c = Σr.C * U
    return s, b, c
end

function project(Σ::StateSpace, V, W)
    Ar = (W'*V)\(W'*Σ.A*V);
    Br = (W'*V)\(W'*Σ.B);
    Cr = Σ.C * V
    return ss(Ar, Br, Cr, Σ.D) 
end

function construct_V_and_W!(V, W, Σ::StateSpace, s, b, c)
    r = length(s)
    
    j = 1
    while j < r + 1
        x = (s[j] * I - Σ.A) \ (Σ.B * b[:,j])
        y = (s[j] * I - Σ.A') \ (Σ.C' * c[:,j]);
        if abs(imag(s[j])) > 0
            V[:, j] = real(x)
            W[:, j] = real(y)
            V[:, j+1] = imag(x)
            W[:, j+1] = imag(y)

            j += 2
        else
            V[:, j] = real(x)
            W[:, j] = real(y)
            j += 1
        end
    end

    #  orthonormalize V and W
    V .= Matrix(qr(V).Q)
    W .= Matrix(qr(W).Q)
    # return V, W
end
