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
function acceptance_probability(n_old::Vector{Int},
                                n_new::Vector{Int},
                                ψ::Wavefunction)

    log_ratio = 0.0

    for i in eachindex(n_old)
        if n_old[i] != n_new[i]
            log_ratio += 2 * (
                ψ.f[n_new[i] + 1] -
                ψ.f[n_old[i] + 1]
            )
        end
    end

    return log_ratio ≥ 0 ? 1.0 : exp(log_ratio)
end
