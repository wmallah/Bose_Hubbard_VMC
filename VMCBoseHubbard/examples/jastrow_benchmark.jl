using Pkg
Pkg.activate("../")

include("../src/VMCBoseHubbard.jl")
using .VMCBoseHubbard

import ..VMCBoseHubbard: MC_integration_Jastrow
import ..VMCBoseHubbard: estimate_tau
import ..VMCBoseHubbard: JastrowParams

function compute_q_grid(L, Nv)
    m = collect(1:Nv)
    return 2π .* m ./ L
end

# -----------------------
# System parameters
# -----------------------
L = 60
N_target = 60
t = 1.0
n_max = N_target

# Initial Jastrow parameters
# vq_init = zeros(L ÷ 2)
vq_init = [3.5024847541488278,3.383476263327771,3.229229526112806,
2.9866547945901933,2.7684143888137296,2.545146148982976,
2.3573277506409385,2.153722151887709,1.9946729151830116,
1.8521619086313694,1.7242674015967503,1.6253829899508971,
1.5339247755799204,1.458178437190528,1.392009697153715,
1.3394279327966896,1.2911998032869836,1.253236370141688,
1.2193395201588166,1.1877892781756658,1.1628372273725993,
1.1388702250162521,1.1232406254805813,1.1083383005460865,
1.095965954592878,1.0862860140717963,1.0810662181552053,
1.0759478016266866,1.070322020907657,0.5333398152971762]

# Parameter scans
# U_vals = [i for i in 0.0:1.0:10.0]
# μ_vals = zeros(11)
U_vals = [2.4, 2.5, 3, 4, 6]
μ_vals = zeros(5)
# U_vals = [4.0]
# μ_vals = [0.0]

dim = "1D"
grand_canonical = false
projective = false

lattice = Lattice1D(L)
ensemble = !grand_canonical ? "C" : "GC"

dir_base = "../data/$(ensemble)/$(dim)/L$(L)_N$(N_target)/jastrow"
mkpath(dir_base)

results = []

# -----------------------
# Loop over parameters
# -----------------------
for (U, μ) in zip(U_vals, μ_vals)

    println("Optimizing Jastrow parameters for U = $U, μ = $μ")

    sys = System(t, U, μ, lattice)

    # Initial parameter object
    params_init = JastrowParams(copy(vq_init))

    # -----------------------
    # Optimize Jastrow parameters (SR)
    # -----------------------
    params_opt, history = optimize_jastrow_SR(
        sys,
        params_init,
        N_target,
        n_max;
        η = 0.05,
        num_walkers = 200,
        num_MC_steps = 3_000,
        num_equil_steps = 500,
        block_size = 1_500,
        z = 1.0
    )

    println("    Optimal vq = ", params_opt.vq)

    # -----------------------
    # Save optimized v_q
    # -----------------------

    vq = params_opt.vq
    Nv = length(vq)

    q = compute_q_grid(L, Nv)

    vq_q2 = 0.5 .* vq .* q.^2

    vq_file = "$(dir_base)/VMC_vq_vs_q_U$(U).dat"

    open(vq_file, "w") do io
        println(io, "# q   v_q   v_q_q2")
        for i in 1:Nv
            println(io, "$(q[i]) $(vq[i]) $(vq_q2[i])")
        end
    end

    println("Saved v_q data to ", vq_file)

    # --------------------------------
    # Use optimized parameters as next guess
    # --------------------------------
    vq_init .= params_opt.vq

    # -----------------------
    # Final high-statistics evaluation
    # -----------------------
    final_result = MC_integration_Jastrow(
        sys,
        N_target,
        params_opt,
        n_max,
        grand_canonical,
        projective;
        num_walkers = 400,
        num_MC_steps = 10_000,
        num_equil_steps = 2_000,
        block_size=1_500
    )

    acceptance_ratio = final_result.acceptance_ratio
    println("Acceptance Ratio: $acceptance_ratio")

    energies = final_result.energies

    τE = estimate_tau(energies)

    println("Estimated autocorrelation time τ =", τE)
    println("Effective sample size ≈ ", length(energies)/(2τE))

    push!(results, (U = U, params = params_opt, result = final_result))

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
# Save energies
# -----------------------
open("$(dir_base)/VMC_results.dat", "w") do io
    println(io, "# U   energy   sem")

    for entry in results
        r = entry.result

        println(io,
            "$(entry.U) $(r.mean_energy) $(r.sem_energy)"
        )
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