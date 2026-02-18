using Pkg
Pkg.activate("../")

include("../src/VMCBoseHubbard.jl")
using .VMCBoseHubbard

import ..VMCBoseHubbard: MC_integration
import ..VMCBoseHubbard: estimate_n_max
# -----------------------
# System parameters
# -----------------------
L = 16
N_target = 16
t = 1.0 * 0.5   # The J we have defined in our Hamiltonian is 2t from Krauth's Hamiltonian definition

U = 7.0                # ← choose interaction
μ = 0.0                # choose chemical potential

dim = "1D"
grand_canonical = false
projective = false

lattice = Lattice1D(L)
ensemble = !grand_canonical ? "C" : "GC"

dir_base = "../data/$(ensemble)/$(dim)/L$(L)_N$(N_target)"
mkpath(dir_base)

# -----------------------
# κ scan grid
# -----------------------
κ_vals = collect(range(2.5, 2.75, length=50))   # adjust as needed

println("Running κ scan for U = $U, μ = $μ")

sys = System(t, U, μ, lattice)

results = []

# -----------------------
# Loop over κ values
# -----------------------
for κ in κ_vals

    n_max = 16 # estimate_n_max(κ)

    println("κ = $(round(κ, digits=6))")

    result = MC_integration(
        sys, N_target, κ, n_max, grand_canonical, !projective;
        num_walkers = 200,
        num_MC_steps = 10_000,
        num_equil_steps = 2_000,
    )

    push!(results, (
        κ = κ,
        mean_energy = result.mean_energy,
        sem_energy = result.sem_energy,
        mean_kinetic = result.mean_kinetic,
        sem_kinetic = result.sem_kinetic,
        mean_potential = result.mean_potential,
        sem_potential = result.sem_potential,
        acc = result.acceptance_ratio
    ))

    acceptance_ratio = result.acceptance_ratio
    println("Acceptance Ratio: $acceptance_ratio")

end

# -----------------------
# Save κ-energy curve
# -----------------------
outfile = "$(dir_base)/E_vs_kappa_U$(U).dat"

open(outfile, "w") do io
    println(io, "# kappa   energy   sem   E_kin   E_kin_sem   E_pot   E_pot_sem   acc_ratio")
    for r in results
        println(io,
            "$(r.κ) " *
            "$(r.mean_energy) $(r.sem_energy) " *
            "$(r.mean_kinetic) $(r.sem_kinetic) " *
            "$(r.mean_potential) $(r.sem_potential) " *
            "$(r.acc)"
        )
    end
end

println("Saved κ-scan data to $outfile")