#=
Purpose: calculate the local energy
Input: n (vector of integers describing the system state), ψ (wavefunction struct), sys (system struct), n_max (maximum site occupancy)
Output: total local energy (kinetic + potential - chemical), kinetic energy, potential energy
Author: Will Mallah
Last Updated: 01/25/26
=#
function local_energy(n::Vector{Int}, ψ::GutzwillerWavefunction, sys::System, n_max)
    log_f = ψ.f                     # shared Gutzwiller coefficient vector
    t, U, μ = sys.t, sys.U, sys.μ
    lattice = sys.lattice
    L = length(n)

    E_kin = 0.0
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
                    R1 = exp(log_R1)
                    E_kin += -t * sqrt((n[i] + 1) * n[j]) * R1
                end

                # hop i → j
                if hop_possible(n, i, j, n_max)
                    # log R = log[ f(nj+1) f(ni-1) / ( f(nj) f(ni) ) ]
                    log_R2 = (log_f[n[j] + 2] + log_f[n[i]]) - (log_f[n[j] + 1] + log_f[n[i] + 1])
                    R2 = exp(log_R2)
                    E_kin += -t * sqrt((n[j] + 1) * n[i]) * R2
                end
            end
        end
    end

    # Chemical potential correction
    N = sum(n)
    
    return E_kin + E_pot - μ*N, E_kin, E_pot
end


#=
Purpose: estimate the gradient of the energy for use in gradient descent optimization
Input: results from Monte Carlo integration
Output: energy gradient
Author: Will Mallah
Last Updated: 01/25/26
=#
function estimate_energy_gradient(result::VMCResults)
    # Pull list of energies and list of wavefunction derivatives
    E_loc = result.energies
    O_k = result.derivative_log_psi

    # If lists are empty, return NaN
    if isempty(E_loc) || isempty(O_k)
        return NaN
    end

    # Calculate mean values
    mean_E  = mean(E_loc)
    mean_O  = mean(O_k)
    mean_EO = mean(E_loc .* conj.(O_k))

    # Return energy gradient (see Sorella "Wave function optimization in the variational Monte Carlo method")
    return 2 * real(mean_EO - mean_E * conj(mean_O))
end