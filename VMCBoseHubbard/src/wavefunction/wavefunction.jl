using SpecialFunctions: loggamma
using LogExpFunctions: logsumexp

abstract type Wavefunction end

#=
Purpose: generate coefficients of the Gutzwiller variational wavefunction
Input: kappa (variational parameter), n_max (maximum number of particles on a given site); logfact (log factorial table)
Output: struct of Gutzwiller Wavefunction containing coefficients
Author: Will Mallah
Last Updated: 07/16/25
    Summary: Since coefficient are globally the same, generate them as a vector rather than matrix to reduce number of computations and confusion around indexing
=#

# Precompute factorial values to avoid redundant calculations
const LOGFACTORIAL_TABLE = [loggamma(m + 1) for m in 0:100]  # supports up to n_max=100

function generate_coefficients(κ::Real, n_max::Int; logfact=LOGFACTORIAL_TABLE)
    log_f = [
        -0.5 * κ * n^2 - 0.5 * logfact[n + 1]
        for n in 0:n_max
    ]

    # Per-site normalization ONLY
    log_Z = logsumexp(2 .* log_f)
    log_f .-= 0.5 * log_Z

    # println(minimum(abs.(exp.(log_f))))
    # println(maximum(abs.(exp.(log_f))))


    return GutzwillerWavefunction(log_f)
end

struct GutzwillerWavefunction{T <: Real} <: Wavefunction
    # Vector for coefficients
    f::Vector{T}
end

struct JastrowParams{T <: Real} <: Wavefunction
    vq::Vector{T}   # length = L/2
end