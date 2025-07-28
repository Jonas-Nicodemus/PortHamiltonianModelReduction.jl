abstract type AbstractData end

import ControlSystemsBase: SimResult

"""
TimeDomainData{T}

Data structure to hold time-domain simulation data.
See [`tddata`](@ref) for a convenient constructor.

# Fields
- `Ẋ`: State derivative matrix.
- `X`: State matrix.
- `U`: Input matrix.
- `Y`: Output matrix.
- `t`: Time vector.
"""
struct TimeDomainData{T<:Real} <: AbstractData
    Ẋ::Matrix{T}
    X::Matrix{T}
    U::Matrix{T}
    Y::Matrix{T}
    t::Vector{T}
    TimeDomainData{T}(Ẋ,X,U,Y,t) where {T<:Real} = size(Ẋ,2) != length(t) || size(X,2) != length(t) || size(U,2) != length(t) || size(Y,2) != length(t) ? error("Number of time samples does not match the number of data samples.") : new(Ẋ,X,U,Y,t)
end
TimeDomainData(Ẋ::Matrix{T},X::Matrix{T},U::Matrix{T},Y::Matrix{T},t::Vector{T}) where {T<:Real} = TimeDomainData{T}(Ẋ,X,U,Y,t)

"""
    data = tddata(Ẋ::Matrix{T},X::Matrix{T},U::Matrix{T},Y::Matrix{T},t::Vector{T})
    data = tddata(res::SimResult)

Creates a `TimeDomainData` object with element type `T`.
"""
tddata(args...; kwargs...) = TimeDomainData(args...; kwargs...)

function tddata(res::SimResult)
    return tddata(res, 1:length(res.t))
end

function tddata(res::SimResult, mask::AbstractVector, noise::Float64=0.0)
    X = res.x[:,mask]
    U = res.u[:,mask]
    Y = res.y[:,mask]
    t = res.t[mask]
    Ẋ = res.sys.A * X + res.sys.B * U

    if noise > 0.0
        Ẋ = data.Ẋ + σ * randn(size(data.Ẋ))
        X = data.X + σ * randn(size(data.X))
        U = data.U
        Y = data.Y + σ * randn(size(data.Y))
    end

    return TimeDomainData(Ẋ, X, U, Y, Array(t))
end

function tddata(data::TimeDomainData, mask::AbstractVector)
    return TimeDomainData(data.Ẋ[:,mask], data.X[:,mask], data.U[:,mask], data.Y[:,mask], data.t[mask])
end
