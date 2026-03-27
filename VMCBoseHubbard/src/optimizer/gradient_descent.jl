using Pkg
Pkg.activate("../")

include("../src/VMCBoseHubbard.jl")
using .VMCBoseHubbard

import ..VMCBoseHubbard: MC_integration_Jastrow
import ..VMCBoseHubbard: estimate_tau
import ..VMCBoseHubbard: JastrowParams

function num_pair_modes(L::Int)
    return (L - 1) ÷ 2
end

function has_edge_mode(L::Int)
    return iseven(L)
end

function compute_q_grid(L::Int)
    m = collect(1:num_pair_modes(L))
    return 2π .* m ./ L
end

function initialize_output_file(filepath::String, header::String)
    if !isfile(filepath)
        open(filepath, "w") do io
            println(io, header)
        end
    end
end

function append_energy_result(filepath::String, U, result)
    open(filepath, "a") do io
        println(io, "$(U) $(result.mean_energy) $(result.sem_energy)")
    end
end

function append_energy_parts(filepath::String, U, result)
    open(filepath, "a") do io
        println(io,
            "$(U) " *
            "$(result.mean_kinetic) $(result.sem_kinetic) " *
            "$(result.mean_potential) $(result.sem_potential)"
        )
    end
end

# -----------------------
# System parameters
# -----------------------
L = 60
N_target = 60
t = 1.0
n_max = N_target

# Initial Jastrow parameters
vq_init_full = [3.5024847541488278,3.383476263327771,3.229229526112806,
2.9866547945901933,2.7684143888137296,2.545146148982976,
2.3573277506409385,2.153722151887709,1.9946729151830116,
1.8521619086313694,1.7242674015967503,1.6253829899508971,
1.5339247755799204,1.458178437190528,1.392009697153715,
1.3394279327966896,1.2911998032869836,1.253236370141688,
1.2193395201588166,1.1877892781756658,1.1628372273725993,
1.1388702250162521,1.1232406254805813,1.1083383005460865,
1.095965954592878,1.0862860140717963,1.0810662181552053,
1.0759478016266866,1.070322020907657,0.5333398152971762]

npair = num_pair_modes(L)
vpair_init = copy(vq_init_full[1:npair])
vedge_init = has_edge_mode(L) ? vq_init_full[npair + 1] : nothing

# Parameter scans
U_vals = [2.4, 2.5, 3.0, 4.0, 6.0]
μ_vals = zeros(length(U_vals))

dim = "1D"
grand_canonical = false
projective = false

lattice = Lattice1D(L)
ensemble = !grand_canonical ? "C" : "GC"

dir_base = "../data/$(ensemble)/$(dim)/L$(L)_N$(N_target)/jastrow"
mkpath(dir_base)

results_file = "$(dir_base)/VMC_results.dat"
energy_parts_file = "$(dir_base)/VMC_energy_parts.dat"

initialize_output_file(results_file, "# U   energy   sem")
initialize_output_file(energy_parts_file, "# U   E_kin   E_kin_sem   E_pot   E_pot_sem")

for (U, μ) in zip(U_vals, μ_vals)

    println("Optimizing Jastrow parameters for U = $U, μ = $μ")

    sys = System(t, U, μ, lattice)

    params_init = isnothing(vedge_init) ?
        JastrowParams(copy(vpair_init)) :
        JastrowParams(copy(vpair_init), vedge_init)

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

    println("    Optimal vpair = ", params_opt.vpair)
    if !isnothing(params_opt.vedge)
        println("    Optimal vedge = ", params_opt.vedge)
    end

    # Save paired modes only
    vpair = params_opt.vpair
    q = compute_q_grid(L)
    vq_q2 = vpair .* q.^2

    vq_file = "$(dir_base)/VMC_vq_vs_q_U$(U).dat"
    open(vq_file, "w") do io
        println(io, "# q   v_q   v_q_q2")
        for i in eachindex(vpair)
            println(io, "$(q[i]) $(vpair[i]) $(vq_q2[i])")
        end
    end
    println("Saved paired-mode v_q data to ", vq_file)

    # Save edge mode separately
    if !isnothing(params_opt.vedge)
        vedge_file = "$(dir_base)/VMC_vedge_U$(U).dat"
        open(vedge_file, "w") do io
            println(io, "# q_edge   v_edge")
            println(io, "$(π) $(params_opt.vedge)")
        end
        println("Saved edge-mode data to ", vedge_file)
    end

    # Warm start next U
    vpair_init .= params_opt.vpair
    if has_edge_mode(L) && !isnothing(params_opt.vedge)
        vedge_init = params_opt.vedge
    end

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
        block_size = 1_500
    )

    acceptance_ratio = final_result.acceptance_ratio
    println("Acceptance Ratio: $acceptance_ratio")

    energies = final_result.energies
    τE = estimate_tau(energies)

    println("Estimated autocorrelation time τ = ", τE)
    println("Effective sample size ≈ ", length(energies) / (2τE))

    if grand_canonical
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

    append_energy_result(results_file, U, final_result)
    append_energy_parts(energy_parts_file, U, final_result)

    println("Saved completed results for U = $U")
end