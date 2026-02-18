
function signed_logsumexp(logvals, signs)
    m = maximum(logvals)
    s = 0.0
    for (lv, sg) in zip(logvals, signs)
        s += sg * exp(lv - m)
    end
    return m + log(abs(s)), sign(s)
end


#=
Purpose: calculate the local energy
Input: n (vector of integers describing the system state), ψ (wavefunction struct), sys (system struct), n_max (maximum site occupancy)
Output: total local energy (kinetic + potential - chemical), kinetic energy, potential energy
Author: Will Mallah
Last Updated: 01/25/26
=#
function local_energy(n::Vector{Int}, ψ::GutzwillerWavefunction, sys::System, n_max::Int64, grand_canonical::Bool, projective::Bool)
    log_f = ψ.f                     # shared Gutzwiller coefficient vector
    t, U, μ = sys.t, sys.U, sys.μ
    lattice = sys.lattice
    L = length(n)

    log_E_kin_contributions = Float64[]
    signs = Int[]
    E_pot = 0.0

    # Potential energy term
    for i in 1:L
        E_pot += (U / 2) * n[i] * (n[i] - 1)    # no need for f-dependent terms; cancels in ratio
    end

    # Kinetic energy term
    for i in 1:L
        for j in lattice.neighbors[i]
            if j > i
                # hop j → i
                if hop_possible(n, j, i, n_max)
                    # log R = log[ f(ni+1) f(nj-1) / ( f(ni) f(nj) ) ]
                    log_R1 = (log_f[n[i] + 2] + log_f[n[j]]) - (log_f[n[i] + 1] + log_f[n[j] + 1])
                    log_E_kin = 0.5*log((n[i] + 1) * n[j]) + log_R1
                    push!(log_E_kin_contributions, log_E_kin)
                    push!(signs, -1)    # kinetic term is negative
                end

                # hop i → j
                if hop_possible(n, i, j, n_max)
                    # log R = log[ f(nj+1) f(ni-1) / ( f(nj) f(ni) ) ]
                    log_R2 = (log_f[n[j] + 2] + log_f[n[i]]) - (log_f[n[j] + 1] + log_f[n[i] + 1])
                    log_E_kin = 0.5 * log((n[j] + 1) * n[i]) + log_R2
                    push!(log_E_kin_contributions, log_E_kin)
                    push!(signs, -1)    # kinetic term is negative
                end
            end
        end
    end

    log_abs_E, sign_E = signed_logsumexp(log_E_kin_contributions, signs)
    E_kin = sign_E * t * exp(log_abs_E)

    # Chemical potential correction
    N = sum(n)
    
    if grand_canonical && !projective
        return E_kin + E_pot - μ*N, E_kin, E_pot
    else
        return E_kin + E_pot, E_kin, E_pot
    end
end


#=
Purpose: estimate the gradient of the energy for use in gradient descent optimization
Input: results from Monte Carlo integration
Output: energy gradient
Author: Will Mallah
Last Updated: 01/25/26
=#
# function estimate_energy_gradient(result::VMCResults)
#     # Pull list of energies and list of wavefunction derivatives
#     E_loc = result.energies
#     O_k = result.derivative_log_psi

#     # If lists are empty, return NaN
#     if isempty(E_loc) || isempty(O_k)
#         return NaN
#     end

#     # Calculate mean values
#     mean_E  = mean(E_loc)
#     mean_O  = mean(O_k)
#     mean_EO = mean(E_loc .* conj.(O_k))

#     # Return energy gradient (see Sorella "Wave function optimization in the variational Monte Carlo method")
#     return 2 * real(mean_EO - mean_E * conj(mean_O))
# end

function estimate_energy_gradient_and_metric(result::VMCResults)

    E_loc = result.energies
    O_k   = result.derivative_log_psi

    if isempty(E_loc) || isempty(O_k)
        return NaN, NaN
    end

    mean_E  = mean(E_loc)
    mean_O  = mean(O_k)
    mean_EO = mean(E_loc .* conj.(O_k))

    # Gradient
    g = 2 * real(mean_EO - mean_E * conj(mean_O))

    # Metric (covariance of O)
    mean_O2 = mean(abs2.(O_k))
    S = real(mean_O2 - abs2(mean_O))

    return g, S
end
