# wavefunction.jl

abstract type Wavefunction end

# ── Gutzwiller ────────────────────────────────────────────────────────────────

const LOGFACTORIAL_TABLE = [loggamma(m + 1) for m in 0:100]

struct GutzwillerWavefunction{T <: Real} <: Wavefunction
    κ::T
    log_f::Vector{T}
end

function GutzwillerWavefunction(κ::Real, n_max::Int; logfact = LOGFACTORIAL_TABLE)
    log_f = [-0.5 * κ * n^2 - 0.5 * logfact[n + 1] for n in 0:n_max]
    log_Z = logsumexp(2 .* log_f)
    log_f .-= 0.5 * log_Z
    return GutzwillerWavefunction(κ, log_f)
end

# ── Jastrow ───────────────────────────────────────────────────────────────────

struct JastrowWavefunction{T <: Real} <: Wavefunction
    vr::Vector{T}    # fld(L, 2) + 1 real-space coefficients (translational invariance)
end