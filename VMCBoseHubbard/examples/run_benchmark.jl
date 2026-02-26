using Pkg
Pkg.activate("../")

include("../src/VMCBoseHubbard.jl")
using .VMCBoseHubbard

import ..VMCBoseHubbard: MC_integration
import ..VMCBoseHubbard: estimate_tau

# -----------------------
# System parameters
# -----------------------
L = 2
N_target = 2
t = 1.0
κ_init = 1.0

# U_vals = [1.0]
# μ_vals = [0.0]

# 12x12 U and μ values
U_vals = [i for i in 0.0:1.0:10.0]
μ_vals = zeros(11)
# μ_vals = [0.6869, 1.1717, 1.4949, 2.0606, 2.7879, 3.1111, 3.2727, 3.4343]

# U and μ values for 2 particles, 4 sites
# U_vals = [0.0, 1.0, 5.0, 10.0]
# μ_vals = [-2.223987815985, -1.996641490650, -0.930098065243, -0.751790527039]

# U and μ values for 2x2
# U_vals = [i for i in 0.0:1.0:10.0]
# μ_vals = [0.0, 0.5, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.2]
# μ_vals = zeros(11)

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

    # Conservative truncation
    n_max = 2

    # -----------------------
    # Optimize κ (MC-error stopping)
    # -----------------------
    κ_opt, history = optimize_kappa(
        sys, N_target, n_max, grand_canonical, !projective;
        κ_init = κ_init,
        η = 0.05,
        num_walkers = 400,
        num_MC_steps = 10_000,
        num_equil_steps = 2_000,
        block_size::Int = 200,
        z::Float64 = 2.0
    )

    println("    Optimal κ = $(round(κ_opt, digits=10))")

    # -----------------------
    # Final high-statistics evaluation
    # -----------------------
    if projective
        final_result = MC_integration(
            sys, N_target, κ_opt, n_max, grand_canonical, projective;
            num_walkers = 400,
            num_MC_steps = 100_000,
            num_equil_steps = 20_000
        )
    else
        final_result = MC_integration(
        sys, N_target, κ_opt, n_max, grand_canonical, !projective;
        num_walkers = 400,
        num_MC_steps = 100_000,
        num_equil_steps = 20_000
        )
    end

    acceptance_ratio = final_result.acceptance_ratio

    println("Acceptance Ratio: $acceptance_ratio")

    energies = final_result.energies
    τE = estimate_tau(energies)
    println("Estimated autocorrelation time τ =", τE)
    println("Effective sample size ≈ ", length(energies)/(2τE))


    push!(results, (U = U, κ = κ_opt, result = final_result))

    # -----------------------
    # Save particle-number histogram
    # -----------------------
    hist_file = "$(dir_base)/PN_hist_U$(U).dat"
    open(hist_file, "w") do io
        println(io, "# N   count")
        for (i, count) in enumerate(final_result.PN)
            if count > 0
                println(io, "$(i - 1) $count")
            end
        end
    end
end

# -----------------------
# Save total energies
# -----------------------
open("$(dir_base)/VMC_results.dat", "w") do io
    println(io, "# U   kappa   energy   sem")
    for entry in results
        r = entry.result
        println(io, "$(entry.U) $(entry.κ) $(r.mean_energy) $(r.sem_energy)")
    end
end

# -----------------------
# Save energy components
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
