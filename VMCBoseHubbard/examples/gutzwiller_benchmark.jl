using Pkg
Pkg.activate("../")

include("../src/VMCBoseHubbard.jl")
using .VMCBoseHubbard

import ..VMCBoseHubbard: estimate_tau

# -----------------------
# System parameters
# -----------------------

L = 4
N = 2
n_max = N
t = 1.0
κ_init = 1.0
wavefunction_init = GutzwillerWavefunction(κ_init, n_max)

# U_vals = [i for i in 0.0:1.0:10.0]
U_vals = [0.0, 3.0, 3.3, 3.4, 4.0, 6.0]

dim = "1D"

lattice = Lattice1D(L)

dir_base = "../data/VMC/$(dim)/L$(L)_N$(N)/Gutzwiller"
mkpath(dir_base)

results = []

# -----------------------
# Loop over parameters
# -----------------------

for U in U_vals

    println("Optimizing Gutzwiller κ for U = $U")

    sys = System(t, U, N, lattice)

    # -----------------------
    # Optimize κ (SR optimizer)
    # -----------------------

    wavefunction_opt, history = optimize_SR(
        sys,
        wavefunction_init,
        n_max;
        η = 0.05,
        num_walkers = 100,
        num_MC_steps = 5_000,
        num_equil_steps = 1_000,
        block_size = 500,
    )

    κ_opt = wavefunction_opt.κ
    # Set initial gradient descent κ guess as optimal value from previous run
    global κ_init = κ_opt
    println("    Optimal κ = $(round(κ_opt, digits=10))")

    wavefunction_opt = GutzwillerWavefunction(κ_opt, n_max)

    # -----------------------
    # Final high-statistics run
    # -----------------------

    final_result = MC_integration(
        sys,
        wavefunction_opt,
        n_max;
        num_walkers = 200,
        num_MC_steps = 10_000,
        num_equil_steps = 2_000,
        block_size = 40_000
    )

    acceptance_ratio = final_result.acceptance_ratio

    println("Acceptance Ratio: $acceptance_ratio")

    energies = final_result.energies
    τE = estimate_tau(energies)

    println("Estimated autocorrelation time τ = ", τE)
    println("Effective sample size ≈ ", length(energies)/(2τE))

    push!(results, (U = U, κ = κ_opt, result = final_result))
end

# -----------------------
# Save total energies
# -----------------------

open("$(dir_base)/VMC_results.dat", "w") do io

    println(io, "# U   kappa   energy   sem")

    for entry in results
        r = entry.result

        println(io,
            "$(entry.U) " *
            "$(entry.κ) " *
            "$(r.mean_energy) " *
            "$(r.sem_energy)"
        )
    end

end


# -----------------------
# Save kinetic / potential parts
# -----------------------

open("$(dir_base)/VMC_energy_parts.dat", "w") do io

    println(io, "# U   E_kin   E_kin_sem   E_pot   E_pot_sem")

    for entry in results

        r = entry.result

        println(io,
            "$(entry.U) " *
            "$(r.mean_kinetic) $(r.sem_kinetic) " *
            "$(r.mean_potential) $(r.sem_potential)"
        )

    end

end