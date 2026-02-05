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
Purpose: calculate the sampling ratio
Input: n_old (old state of system), n_new (new state of the system), κ (variational parameter), n_max (maximum number of particles on a given site)
Output: sampling ratio between two system states
Author: Will Mallah
Last Updated: 07/04/25
=#
function acceptance_probability(n_old::Vector{Int}, n_new::Vector{Int}, κ::Real)
    ratio = 1.0
    for i in eachindex(n_old)
        if n_old[i] != n_new[i]
            f_old = (1 / sqrt(factorial(n_old[i]))) * exp(-κ * n_old[i]^2 / 2.0)
            f_new = (1 / sqrt(factorial(n_new[i]))) * exp(-κ * n_new[i]^2 / 2.0)
            ratio *= abs2(f_new) / abs2(f_old)
        end
    end

    
    return min(1.0, ratio)
end

function acceptance_probability(n_old::Vector{Int}, n_new::Vector{Int}, ψ::Wavefunction)
    ratio = 1.0
    coeff = exp.(ψ.f)

    for i in eachindex(n_old)
        if n_old[i] != n_new[i]
            f_old = coeff[n_old[i] + 1]
            f_new = coeff[n_new[i] + 1]
            ratio *= abs2(f_new) / abs2(f_old)
        end
    end


    return min(1.0, ratio)
end