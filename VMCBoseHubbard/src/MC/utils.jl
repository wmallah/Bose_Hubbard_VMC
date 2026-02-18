using Random

#=
Purpose: estimate value for n_max given kappa parameter and cutoff value for decay of Gutzwiller coefficients
Input: kappa (variational parameter), cutoff (value for coefficient deemed insignificant)
Output: particle number where Gutzwiller coefficients are significant
Author: Will Mallah
Last Updated: 07/04/25
=#
function estimate_n_max(κ::Real; cutoff::Real = 1e-6)
    n = 0
    while true
        f_n = (1 / sqrt(float(factorial(big(n))))) * exp(-κ * n^2 / 2)
        if f_n < cutoff
            return n - 1
        end
        n += 1
        if n > 300
            return 300
        end
    end
end


#=
Purpose: run the VMC function for either a 1D or 2D lattice
Input: sys (system struct), κ (variational parameter), n_max (maximum number of particles on a given site), N_total (total number of particles)
Optional Input: num_walkers, num_MC_steps, num_equil_steps (kwargs...)
Output: result from VMC_fixed_particles
Author: Will Mallah
Last Updated: 07/04/25
=#
function run_vmc(sys::System, κ::Real, n_max::Int, N_target::Int; grand_canonical=true, kwargs...)
    lattice = sys.lattice
    if !grand_canonical
        if lattice isa Lattice1D
            return VMC_fixed_particles(sys, κ, n_max, N_target; kwargs...)
        elseif lattice isa Lattice2D
            return VMC_fixed_particles(sys, κ, n_max, N_target; kwargs...)
        else
            error("Unsupported lattice type: $(typeof(lattice))")
        end
    else
        if lattice isa Lattice1D
            return VMC_grand_canonical_adaptive_mu(sys, κ, n_max, N_target, 1.0; kwargs...)
        elseif lattice isa Lattice2D
            return VMC_grand_canonical_adaptive_mu(sys, κ, n_max, N_target, 1.0; kwargs...)
        else
            error("Unsupported lattice type: $(typeof(lattice))")
        end
    end
end