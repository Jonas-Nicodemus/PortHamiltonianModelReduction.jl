"""
    Σr = phirka(Σph::PortHamiltonianStateSpace, r; tol=1e-3, max_iter=50)

Reduces the state dimension of the port-Hamiltonian system `Σph` to `r` using pHIRKA [GPBS12](@cite).
"""
function phirka(Σph::PortHamiltonianStateSpace, r, num_runs; kwargs...)
    h2 = Inf

    m = size(Σph.G, 2)
    Σphr = phss(zeros(r,r), zeros(r,r), zeros(r,r), zeros(r,m), zeros(r,m), zeros(m,m), zeros(m,m))

    for i in 1:num_runs
        Σphrᵢ = phirka(Σph, r; kwargs...)
        h2ᵢ = norm(Σph - Σphrᵢ)
        if h2ᵢ < h2
            h2 = h2ᵢ
            Σphr = Σphrᵢ
            @debug "Found a better rom with h2 = $h2"
        end
    end

    return Σphr 
end

function phirka(Σph::PortHamiltonianStateSpace, r; tol=1e-3, max_iter=50)
    Σ = ss(Σph)
    A = Σ.A
    B = Σ.B
    Q = Σph.Q

    V = Matrix(qr(randn(Σ.nx, r)).Q)
    W = Q * V / (V' * Q * V)
    s_prev = zeros(ComplexF64, r)
    Σphr = pgprojection(Σph, V, W)
    Σr = ss(Σphr)

    s, b = interpolation_data(Σr.A, Σr.B)

    for i in 1:max_iter
        Ar, Br, V, W = interpolate(V, W, A, B, Q, s, b)
        s, b = interpolation_data(Ar, Br)

        if isconverged(s, s_prev, tol)
            @info "pHIRKA converged after $i iterations"
            break
        end

        if i == max_iter
            @warn "pHIRKA not converged after $max_iter iterations"
        end

        s_prev = s
    end

    return pgprojection(Σph, V, W)
end

function interpolation_data(A, B)
    λ, U = eigen(A)
    s = -λ
    b = (U \ B)'
    return s, b
end

function interpolate(V, W, A, B, Q, s, b)
    construct_V_and_W!(V, W, A, B, Q, s, b)
    Ar = W' * A * V
    Br = W' * B
    return Ar, Br, V, W
end

function construct_V_and_W!(V, W, A, B, Q, s, b)
    r = length(s)
    
    j = 1
    while j < r + 1
        x = (s[j] * I - A) \ (B * b[:,j])
        if abs(imag(s[j])) > 0
            V[:, j] = real(x)
            V[:, j+1] = imag(x)
            j += 2
        else
            V[:, j] = real(x)
            j += 1
        end
    end

    #  Orthonormalizing
    V .= Matrix(qr(V).Q)
    W .= Q * V / (V' * Q * V)
end

function isconverged(s, s_prev, tol)
    sort_by = x -> (abs(x), angle(x))
    s = sort(s, by=sort_by)
    s_prev = sort(s_prev, by=sort_by)
    d = norm(s-s_prev)/norm(s_prev)
    return d < tol
end
