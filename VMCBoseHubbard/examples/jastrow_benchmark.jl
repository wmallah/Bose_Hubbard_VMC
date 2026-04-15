using Pkg
Pkg.activate("../")

include("../src/VMCBoseHubbard.jl")
using .VMCBoseHubbard

import ..VMCBoseHubbard: MC_integration_Jastrow
import ..VMCBoseHubbard: estimate_tau
import ..VMCBoseHubbard: JastrowParams
import ..VMCBoseHubbard: optimize_jastrow_SR

function realspace_to_momentum_jastrow(vr::Vector{Float64}, L::Int)
    Rmax = fld(L, 2)
    @assert length(vr) == Rmax + 1

    q_vec = [2π * m / L for m in 0:Rmax]
    vq_vec = zeros(Float64, Rmax + 1)

    if iseven(L)
        for iq in eachindex(q_vec)
            qq = q_vec[iq]

            s = vr[1]  # r = 0

            for r in 1:(Rmax - 1)
                s += 2.0 * vr[r + 1] * cos(qq * r)
            end

            s += vr[Rmax + 1] * cos(qq * Rmax)  # r = L/2
            vq_vec[iq] = s
        end
    else
        for iq in eachindex(q_vec)
            qq = q_vec[iq]

            s = vr[1]  # r = 0

            for r in 1:Rmax
                s += 2.0 * vr[r + 1] * cos(qq * r)
            end

            vq_vec[iq] = s
        end
    end

    return q_vec, vq_vec
end


# -----------------------
# System parameters
# -----------------------
L = 60
N_target = 60
t = 1.0 / 2.0            # NOTE: Capello also follows the "t/2" convention like Krauth did for Gutzwiller
n_max = N_target

# Initial REAL-SPACE Jastrow parameters
vr_init = zeros(fld(L, 2) + 1)

# Parameter scans
U_vals = [2.0, 2.4, 2.5, 3.0, 4.0, 6.0]
# U_vals = 0.0:1.0:10.0
# U_vals = [3.0]
μ_vals = zeros(length(U_vals))

dim = "1D"
grand_canonical = false
projective = false

lattice = Lattice1D(L)
ensemble = !grand_canonical ? "C" : "GC"

dir_base = "../data/$(ensemble)/$(dim)/L$(L)_N$(N_target)/jastrow_realspace"
mkpath(dir_base)

# Summary files written incrementally
results_file = "$(dir_base)/VMC_results.dat"
energy_parts_file = "$(dir_base)/VMC_energy_parts.dat"

open(results_file, "w") do io
    println(io, "# U   energy   sem")
end

open(energy_parts_file, "w") do io
    println(io, "# U   E_kin   E_kin_sem   E_pot   E_pot_sem")
end

results = []

# -----------------------
# Loop over parameters
# -----------------------
for (U, μ) in zip(U_vals, μ_vals)

    println("Optimizing Jastrow parameters for U = $U, μ = $μ")

    sys = System(t, U, μ, lattice)

    params_init = JastrowParams(copy(vr_init))

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
        num_MC_steps = 4_000,
        num_equil_steps = 800,
        block_size = 100,
        z = 1.0
    )

    println("    Optimal vr = ", params_opt.vr)

    # -----------------------
    # Save optimized v(r)
    # -----------------------
    vr = copy(params_opt.vr)
    Rmax = length(vr) - 1

    vr_file = "$(dir_base)/VMC_vr_vs_r_U$(U).dat"
    open(vr_file, "w") do io
        println(io, "# r   v_r")
        for r in 0:Rmax
            println(io, "$(r) $(vr[r + 1])")
        end
    end

    println("Saved v(r) data to ", vr_file)

    # -----------------------
    # Fourier transform v(r) -> v(q)
    # -----------------------
    q, vq = realspace_to_momentum_jastrow(vr, L)
    vq_q2 = vq .* q.^2

    vq_file = "$(dir_base)/VMC_vq_vs_q_U$(U).dat"
    open(vq_file, "w") do io
        println(io, "# q   v_q   v_q_times_q2")
        for i in eachindex(q)
            println(io, "$(q[i]) $(vq[i]) $(vq_q2[i])")
        end
    end

    println("Saved Fourier-transformed v_q data to ", vq_file)

    # -----------------------
    # Optional: save optimization history for this U
    # -----------------------
    history_file = "$(dir_base)/SR_history_U$(U).dat"
    open(history_file, "w") do io
        println(io, "# iter   energy   " * join(["g_$i" for i in 1:length(vr)], " "))
        for (iter, h) in enumerate(history)
            println(io,
                "$(iter) $(h.energy) " * join(string.(h.gradient), " ")
            )
        end
    end

    # --------------------------------
    # Use optimized v(r) as next guess
    # --------------------------------
    vr_init .= params_opt.vr   

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
        num_walkers = 200,
        num_MC_steps = 5_000,
        num_equil_steps = 1_000,
        block_size = 100
    )

    acceptance_ratio = final_result.acceptance_ratio
    println("Acceptance Ratio: $acceptance_ratio")

    energies = final_result.energies
    τE = estimate_tau(energies)

    println("Estimated autocorrelation time τ = ", τE)
    println("Effective sample size ≈ ", length(energies) / (2 * τE))

    push!(results, (U = U, params = params_opt, result = final_result))

    if grand_canonical
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
    # Append summary data immediately
    # -----------------------
    open(results_file, "a") do io
        println(io, "$(U) $(final_result.mean_energy) $(final_result.sem_energy)")
    end

    open(energy_parts_file, "a") do io
        println(io,
            "$(U) " *
            "$(final_result.mean_kinetic) $(final_result.sem_kinetic) " *
            "$(final_result.mean_potential) $(final_result.sem_potential)"
        )
    end

    # Optional: save a one-line run summary per U
    summary_file = "$(dir_base)/run_summary_U$(U).txt"
    open(summary_file, "w") do io
        println(io, "U = $U")
        println(io, "mu = $μ")
        println(io, "acceptance_ratio = $(final_result.acceptance_ratio)")
        println(io, "mean_energy = $(final_result.mean_energy)")
        println(io, "sem_energy = $(final_result.sem_energy)")
        println(io, "mean_kinetic = $(final_result.mean_kinetic)")
        println(io, "sem_kinetic = $(final_result.sem_kinetic)")
        println(io, "mean_potential = $(final_result.mean_potential)")
        println(io, "sem_potential = $(final_result.sem_potential)")
        println(io, "tau_energy = $(τE)")
        println(io, "effective_sample_size = $(length(energies) / (2 * τE))")
    end

    println("Saved all outputs for U = $U")
end