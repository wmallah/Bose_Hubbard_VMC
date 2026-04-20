using Pkg
Pkg.activate("../")

include("../src/VMCBoseHubbard.jl")
using .VMCBoseHubbard

import ..VMCBoseHubbard: MC_integration_Jastrow
import ..VMCBoseHubbard: estimate_tau
import ..VMCBoseHubbard: JastrowParams
import ..VMCBoseHubbard: optimize_jastrow_SR


function realspace_to_momentum_jastrow(vr_in::Vector{Float64}, L::Int)
    Rmax = fld(L, 2)
    @assert length(vr_in) == Rmax + 1

    # Work on a copy so we do not mutate the original optimized parameters
    vr = copy(vr_in)

    # Need to subtract offset to properly do Fourier transform
    vr .-= vr[end]

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


# ============================================================
# Helpers for merge-updating summary files by U
# ============================================================

function normalize_u_key(U)
    return round(Float64(U); digits = 10)
end

function format_u_filename(U)
    return string(normalize_u_key(U))
end

function load_results_dict(filepath::String)
    data = Dict{Float64, Tuple{Float64, Float64}}()

    if !isfile(filepath)
        println("No existing results file found at: ", filepath)
        return data
    end

    println("Loading existing results from: ", filepath)

    open(filepath, "r") do io
        for line in eachline(io)
            s = strip(line)
            if isempty(s) || startswith(s, "#")
                continue
            end

            parts = split(s)
            if length(parts) < 3
                @warn "Skipping malformed line in results file" line = s
                continue
            end

            U   = normalize_u_key(parse(Float64, parts[1]))
            E   = parse(Float64, parts[2])
            sem = parse(Float64, parts[3])

            data[U] = (E, sem)
        end
    end

    println("Loaded existing U values from results file: ", sort(collect(keys(data))))
    return data
end

function load_energy_parts_dict(filepath::String)
    data = Dict{Float64, NTuple{4, Float64}}()

    if !isfile(filepath)
        println("No existing energy-parts file found at: ", filepath)
        return data
    end

    println("Loading existing energy parts from: ", filepath)

    open(filepath, "r") do io
        for line in eachline(io)
            s = strip(line)
            if isempty(s) || startswith(s, "#")
                continue
            end

            parts = split(s)
            if length(parts) < 5
                @warn "Skipping malformed line in energy-parts file" line = s
                continue
            end

            U        = normalize_u_key(parse(Float64, parts[1]))
            Ekin     = parse(Float64, parts[2])
            Ekin_sem = parse(Float64, parts[3])
            Epot     = parse(Float64, parts[4])
            Epot_sem = parse(Float64, parts[5])

            data[U] = (Ekin, Ekin_sem, Epot, Epot_sem)
        end
    end

    println("Loaded existing U values from energy-parts file: ", sort(collect(keys(data))))
    return data
end

function safe_write_results_dict(filepath::String, data::Dict{Float64, Tuple{Float64, Float64}})
    tmpfile = filepath * ".tmp"
    bakfile = filepath * ".bak"

    if isfile(filepath)
        cp(filepath, bakfile; force = true)
    end

    open(tmpfile, "w") do io
        println(io, "# U   energy   sem")
        for U in sort(collect(keys(data)))
            E, sem = data[U]
            println(io, "$(U) $(E) $(sem)")
        end
    end

    mv(tmpfile, filepath; force = true)
end

function safe_write_energy_parts_dict(filepath::String, data::Dict{Float64, NTuple{4, Float64}})
    tmpfile = filepath * ".tmp"
    bakfile = filepath * ".bak"

    if isfile(filepath)
        cp(filepath, bakfile; force = true)
    end

    open(tmpfile, "w") do io
        println(io, "# U   E_kin   E_kin_sem   E_pot   E_pot_sem")
        for U in sort(collect(keys(data)))
            Ekin, Ekin_sem, Epot, Epot_sem = data[U]
            println(io, "$(U) $(Ekin) $(Ekin_sem) $(Epot) $(Epot_sem)")
        end
    end

    mv(tmpfile, filepath; force = true)
end


# -----------------------
# System parameters
# -----------------------
L = 64
N_target = 64
t = 1.0 # / 2.0
n_max = N_target

# Initial REAL-SPACE Jastrow parameters
# vr_init = zeros(fld(L, 2) + 1)
vr_init = [-0.1625881E+01,
    -0.1048148E+01,
    -0.7777309E+00,
    -0.6105601E+00,
    -0.4930722E+00,
    -0.4043681E+00,
    -0.3347432E+00,
    -0.2786680E+00,
    -0.2327923E+00,
    -0.1949050E+00,
    -0.1633171E+00,
    -0.1366961E+00,
    -0.1143394E+00,
    -0.9545265E-01,
    -0.7940817E-01,
    -0.6579527E-01,
    -0.5437115E-01,
    -0.4466728E-01,
    -0.3659925E-01,
    -0.2988174E-01,
    -0.2420967E-01,
    -0.1944907E-01,
    -0.1543752E-01,
    -0.1201605E-01,
    -0.9099452E-02,
    -0.6679389E-02,
    -0.4549107E-02,
    -0.3009113E-02,
    -0.1798847E-02,
    -0.9387513E-03,
    -0.4239784E-03,
    -0.4166991E-04,
    0.0]

# Parameter scans
# U_vals = [2.0, 2.4, 2.5, 3.0, 4.0, 6.0] / t
# U_vals = 0.0:1.0:10.0
U_vals = [3.3578] / t
μ_vals = zeros(length(U_vals))

dim = "1D"
grand_canonical = false
projective = false

lattice = Lattice1D(L)
ensemble = !grand_canonical ? "C" : "GC"

dir_base = "../data/$(ensemble)/$(dim)/L$(L)_N$(N_target)/jastrow_realspace"
mkpath(dir_base)

# Summary files updated by U instead of fully overwritten
results_file = "$(dir_base)/VMC_results.dat"
energy_parts_file = "$(dir_base)/VMC_energy_parts.dat"

println("results_file = ", results_file)
println("energy_parts_file = ", energy_parts_file)

results_dict = load_results_dict(results_file)
energy_parts_dict = load_energy_parts_dict(energy_parts_file)

println("Initial merged results_dict keys = ", sort(collect(keys(results_dict))))
println("Initial merged energy_parts_dict keys = ", sort(collect(keys(energy_parts_dict))))

results = []

# -----------------------
# Loop over parameters
# -----------------------
for (U, μ) in zip(U_vals, μ_vals)

    println("\n==================================================")
    println("Optimizing Jastrow parameters for U = $U, μ = $μ")
    println("==================================================")

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
        num_walkers = 100,
        num_MC_steps = 5_000,
        num_equil_steps = 1_000,
        block_size = 20_000,
        z = 1.0
    )

    # params_opt, history = params_init, []

    println("Optimal vr = ", params_opt.vr)

    # -----------------------
    # Save optimized v(r)
    # -----------------------
    vr = copy(params_opt.vr)
    Rmax = length(vr) - 1

    U_str = format_u_filename(U)

    vr_file = "$(dir_base)/VMC_vr_vs_r_U$(U_str).dat"
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

    vq_file = "$(dir_base)/VMC_vq_vs_q_U$(U_str).dat"
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
    history_file = "$(dir_base)/SR_history_U$(U_str).dat"
    open(history_file, "w") do io
        println(io, "# iter   energy   " * join(["g_$i" for i in 1:length(vr)], " "))
        for (iter, h) in enumerate(history)
            println(io, "$(iter) $(h.energy) " * join(string.(h.gradient), " "))
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
        num_walkers = 400,
        num_MC_steps = 10_000,
        num_equil_steps = 2_000,
        block_size = 40_000
    )

    acceptance_ratio = final_result.acceptance_ratio
    println("Acceptance Ratio: $acceptance_ratio")

    energies = final_result.energies
    τE = estimate_tau(energies)

    println("Estimated autocorrelation time τ = ", τE)
    println("Effective sample size ≈ ", length(energies) / (2 * τE))

    push!(results, (U = U, params = params_opt, result = final_result))

    if grand_canonical
        hist_file = "$(dir_base)/PN_hist_U$(U_str).dat"
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
    # Update summary files for this U only
    # -----------------------
    U_key = normalize_u_key(U)

    println("Updating summary entries for U = ", U_key)

    results_dict[U_key] = (
        final_result.mean_energy,
        final_result.sem_energy
    )

    energy_parts_dict[U_key] = (
        final_result.mean_kinetic,
        final_result.sem_kinetic,
        final_result.mean_potential,
        final_result.sem_potential
    )

    println("Current results_dict keys after update: ", sort(collect(keys(results_dict))))
    println("Current energy_parts_dict keys after update: ", sort(collect(keys(energy_parts_dict))))

    safe_write_results_dict(results_file, results_dict)
    safe_write_energy_parts_dict(energy_parts_file, energy_parts_dict)

    # -----------------------
    # Save a one-line run summary per U
    # -----------------------
    summary_file = "$(dir_base)/run_summary_U$(U_str).txt"
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