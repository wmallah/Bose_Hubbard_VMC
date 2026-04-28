using Pkg
Pkg.activate("../")

include("../src/VMCBoseHubbard.jl")
using .VMCBoseHubbard

import ..VMCBoseHubbard: MC_integration_Gutzwiller
import ..VMCBoseHubbard: estimate_tau

# -----------------------
# System parameters
# -----------------------

L = 12
N_target = 12
n_max = N_target
t = 1.0
κ_init = 1.0

U_vals = [i for i in 0.0:1.0:10.0]
μ_vals = zeros(11)

dim = "1D"
grand_canonical = false
projective = false

lattice = Lattice1D(L)
ensemble = !grand_canonical ? "C" : "GC"

dir_base = "../data/$(ensemble)/$(dim)/L$(L)_N$(N_target)"
mkpath(dir_base)

results = []

# -----------------------
# Loop over parameters
# -----------------------

for (U, μ) in zip(U_vals, μ_vals)

    println("Optimizing Gutzwiller κ for U = $U, μ = $μ")

    sys = System(t, U, μ, lattice)

    # -----------------------
    # Optimize κ (SR optimizer)
    # -----------------------

    κ_opt, history = optimize_kappa_SR(
        sys,
        N_target,
        n_max,
        grand_canonical,
        projective;
        κ_init = κ_init,
        η = 0.05,
        num_walkers = 200,
        num_MC_steps = 10_000,
        num_equil_steps = 1_000,
        block_size = 1500,
        z = 1.0
    )

    println("    Optimal κ = $(round(κ_opt, digits=10))")

    global κ_init = κ_opt

    # -----------------------
    # Final high-statistics run
    # -----------------------

    final_result = MC_integration_Gutzwiller(
        sys,
        N_target,
        κ_opt,
        n_max,
        grand_canonical,
        projective;
        num_walkers = 400,
        num_MC_steps = 100_000,
        num_equil_steps = 20_000,
        block_size = 1500
    )

    acceptance_ratio = final_result.acceptance_ratio

    println("Acceptance Ratio: $acceptance_ratio")

    energies = final_result.energies
    τE = estimate_tau(energies)

    println("Estimated autocorrelation time τ = ", τE)
    println("Effective sample size ≈ ", length(energies)/(2τE))

    push!(results, (U = U, κ = κ_opt, result = final_result))

    if grand_canonical
        # -----------------------
        # Save particle-number histogram
        # -----------------------

        hist_file = "$(dir_base)/gutzwiller/PN_hist_U$(U).dat"

        open(hist_file, "w") do io
            println(io, "# N   count")

            for (i, count) in enumerate(final_result.PN)
                if count > 0
                    println(io, "$(i-1) $count")
                end
            end
        end
    end

end

# -----------------------
# Save total energies
# -----------------------

open("$(dir_base)/gutzwiller/VMC_results.dat", "w") do io

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

open("$(dir_base)/gutzwiller/VMC_energy_parts.dat", "w") do io

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