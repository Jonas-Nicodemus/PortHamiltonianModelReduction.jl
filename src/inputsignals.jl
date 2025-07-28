"""
    chirp(A=1.0, f0=1e-2, f1=1e1, T=2.0)

Generates a chirp signal with amplitude `A`, starting frequency `f0`, ending frequency `f1`, and duration `T`.
"""
function chirp(A::Float64=1.0, f0::Float64=1e-2, f1::Float64=1e1, T::Float64=2.0)
    return (x,t) -> A * sin(2π * ((f1 - f0) / (2T) * t^2 + f0 * t))
end

"""
    sawtooth(A=1.0, T=1.0)

Generates a sawtooth wave signal with amplitude `A` and period `T`.
"""
function sawtooth(A::Float64=1.0, T::Float64=1.0)
    return (x,t) -> 2A * ((t / T) - floor(t / T)) - A
end

"""
    step(A=1.0, ts=1.0)
    
Generates a step signal with amplitude `A` and step time `ts`.
"""
function step(A::Float64=1.0, ts::Float64=1.0)
    return (x,t) -> t .< ts ? 0.0 : A
end

"""
    prbs(t, A=1.0, order=7)

Generates a pseudo-random binary sequence (PRBS) signal with amplitude `A` and specified `order`.
"""
function prbs(t, A::Float64=1.0, order::Int=7)
    prbs_seq_ = prbs_seq(t, A, order)  # Generate PRBS sequence
    time_vector = collect(t)  # Time vector
    
    function u_prbs(x,t)
        i = findfirst(t .<= time_vector)
        return i === nothing ? prbs_seq_[end] : prbs_seq_[i]
    end
    return u_prbs
end

function prbs_seq(t, A::Float64=1.0, order::Int=7)
    shift_reg = rand(0:1, order)  # Initial state
    prbs_seq = Vector{Float64}(undef, length(t))
    taps = [7, 6]  # Example for maximal-length LFSR

    for i in 1:length(t)
        prbs_seq[i] = shift_reg[end] == 0 ? -A : A
        new_bit = xor(shift_reg[taps[1]], shift_reg[taps[2]])
        shift_reg = vcat(new_bit, shift_reg[1:end-1])
    end

    return prbs_seq
end
