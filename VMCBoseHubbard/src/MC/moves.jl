# moves.jl

#=
Purpose: calculate the log of the acceptance probability for the Gutzwiller trial state
Input: n (system configuration), from (hop source site), to (hop target site), ψ (wavefunction)
Output: log of the acceptance ratio between two nearest neighbor hop-connected system configurations for the Gutzwiller trial state
Author: Will Mallah
Last Updated: 02/12/26
=#
function log_acceptance_ratio_gutzwiller(
    n::Vector{Int},
    from_site::Int,
    to_site::Int,
    ψ::GutzwillerWavefunction
)
    log_ratio =
        2 * (ψ.log_f[n[from_site]]     - ψ.log_f[n[from_site] + 1]) +
        2 * (ψ.log_f[n[to_site] + 2]   - ψ.log_f[n[to_site] + 1])
        

    return log_ratio
end

#=
Purpose: calculate the log of the acceptance probability for a Jastrow particle hop.
Input: n (system configuration), from_site (hop source site), to_site (hop target site),
       ψ (Jastrow wavefunction), shell_indices (matrix of lattice shell indices)
Output: log of the acceptance ratio between two nearest neighbor hop-connected system configurations for the Jastrow trial state
Author: Will Mallah
Last Updated: 10/03/2026
=#
function log_acceptance_ratio_jastrow(
    n::Vector{Int},
    from_site::Int,
    to_site::Int,
    ψ::JastrowWavefunction,
    shell_indices::Matrix{Int}
)

    Δlogpsi = compute_delta_logpsi_jastrow(n, from_site, to_site, ψ, shell_indices)
    log_ratio = 2.0 * Δlogpsi + log(n[from_site]) - log(n[to_site] + 1)

    return log_ratio
end