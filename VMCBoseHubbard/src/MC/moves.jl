#=
Purpose: determine if a hop between sites is possible
Input: n (vector of integers describing the system state), from (site index of hop source), to (site index of hop destination), n_max (maximum number of particles on a given site)
Output: true if hop is possible, false if hop is not possible
Author: Will Mallah
Last Updated: 01/13/26
To-Do: 
=#
function hop_possible(n::Vector{Int}, from::Int, to::Int, n_max::Int)
    L = length(n)
    return 1 ≤ from ≤ L &&
           1 ≤ to ≤ L &&
           n[from] > 0 &&
           n[to] < n_max
end



#=
Purpose: calculate the acceptance probability
Input: n_old (old state of system), n_new (new state of the system), ψ (wavefunction)
Output: sampling ratio between two system states
Author: Will Mallah
Last Updated: 02/12/26
=#
function acceptance_probability_Gutzwiller(n::Vector{Int},
                                from::Int,
                                to::Int,
                                ψ::Wavefunction)

    log_ratio =
        2 * (ψ.f[n[from]]     - ψ.f[n[from] + 1]) +
        2 * (ψ.f[n[to] + 2]   - ψ.f[n[to] + 1])

    return log_ratio ≥ 0 ? 1.0 : exp(log_ratio)
end

function acceptance_probability_realspace_jastrow(
    n::Vector{Int},
    from_site::Int,
    to_site::Int,
    ψ::Wavefunction
)
    if n[from_site] == 0
        return 0.0
    end

    Δlogpsi = compute_delta_logpsi_realspace(n, from_site, to_site, ψ)
    log_ratio = 2.0 * Δlogpsi + log(n[from_site]) - log(n[to_site] + 1)

    return log_ratio >= 0.0 ? 1.0 : exp(log_ratio)
end