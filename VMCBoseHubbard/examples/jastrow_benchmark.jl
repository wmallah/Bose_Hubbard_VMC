using Pkg
Pkg.activate("../")

include("../src/VMCBoseHubbard.jl")
using .VMCBoseHubbard
using FFTW

import .VMCBoseHubbard: MC_integration_Jastrow
import .VMCBoseHubbard: estimate_tau
import .VMCBoseHubbard: JastrowParams
import .VMCBoseHubbard: optimize_jastrow_SR


# ============================================================
# Fourier-transform helpers
# ============================================================

function expand_reduced_jastrow(vr_in::Vector{Float64}, L::Int; includes_R0::Bool)
    Rmax = fld(L, 2)
    vr = copy(vr_in)

    vfull = zeros(Float64, L)

    if includes_R0
        # Convention:
        # vr[1]        -> R = 0
        # vr[2]        -> R = 1
        # ...
        # vr[Rmax + 1] -> R = Rmax

        @assert length(vr) == Rmax + 1

        # R = 0
        vfull[1] = vr[1]

        if iseven(L)
            # R = 1, ..., L/2 - 1
            for R in 1:(Rmax - 1)
                vfull[R + 1] = vr[R + 1]
                vfull[L - R + 1] = vr[R + 1]
            end

            # R = L/2 special point
            vfull[Rmax + 1] = vr[Rmax + 1]
        else
            # R = 1, ..., floor(L/2)
            for R in 1:Rmax
                vfull[R + 1] = vr[R + 1]
                vfull[L - R + 1] = vr[R + 1]
            end
        end

    else
        # Convention:
        # vr[1]    -> R = 1
        # vr[2]    -> R = 2
        # ...
        # vr[Rmax] -> R = Rmax

        @assert length(vr) == Rmax

        if iseven(L)
            # R = 1, ..., L/2 - 1
            for R in 1:(Rmax - 1)
                vfull[R + 1] = vr[R]
                vfull[L - R + 1] = vr[R]
            end

            # R = L/2 special point
            vfull[Rmax + 1] = vr[Rmax]
        else
            # R = 1, ..., floor(L/2)
            for R in 1:Rmax
                vfull[R + 1] = vr[R]
                vfull[L - R + 1] = vr[R]
            end
        end
    end

    return vfull
end


function realspace_to_momentum_fft(vr_in::Vector{Float64}, L::Int; includes_R0::Bool)
    Rmax = fld(L, 2)

    vfull = expand_reduced_jastrow(
        vr_in,
        L;
        includes_R0 = includes_R0
    )

    # FFTW convention:
    # vq_full[m + 1] = sum_R vfull[R + 1] * exp(-2πimR/L)
    vq_full = real(fft(vfull))

    q = [2π * m / L for m in 0:Rmax]
    vq_raw = vq_full[1:(Rmax + 1)]

    return q, vq_raw
end


function jastrow_vq_q2_for_plot(vr_in::Vector{Float64},
                                L::Int;
                                includes_R0::Bool = true,
                                subtract_offset::Bool = true)

    vr = copy(vr_in)

    if subtract_offset
        vr .-= vr[end]
    end

    q_all, vq_raw_all = realspace_to_momentum_fft(
        vr,
        L;
        includes_R0 = includes_R0
    )

    # Drop q = 0
    q = q_all[2:end]

    # Sign convention:
    # The optimized real-space Jastrow values are negative in the convention
    # currently used by the real-space code. For plotting positive v_q q^2,
    # we flip the FFT output here ONCE.
    vq = vq_raw_all[2:end]

    vq_q2 = vq .* q.^2

    return q, vq, vq_q2
end


# ============================================================
# Helpers for merge-updating summary files by U/t
# ============================================================

function normalize_u_key(U_over_t)
    return round(Float64(U_over_t); digits = 10)
end

function format_u_filename(U_over_t)
    return string(normalize_u_key(U_over_t))
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

            U_over_t = normalize_u_key(parse(Float64, parts[1]))
            E        = parse(Float64, parts[2])
            sem      = parse(Float64, parts[3])

            data[U_over_t] = (E, sem)
        end
    end

    println("Loaded existing U/t values from results file: ", sort(collect(keys(data))))
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

            U_over_t = normalize_u_key(parse(Float64, parts[1]))
            Ekin     = parse(Float64, parts[2])
            Ekin_sem = parse(Float64, parts[3])
            Epot     = parse(Float64, parts[4])
            Epot_sem = parse(Float64, parts[5])

            data[U_over_t] = (Ekin, Ekin_sem, Epot, Epot_sem)
        end
    end

    println("Loaded existing U/t values from energy-parts file: ", sort(collect(keys(data))))
    return data
end


function safe_write_results_dict(filepath::String, data::Dict{Float64, Tuple{Float64, Float64}})
    tmpfile = filepath * ".tmp"
    bakfile = filepath * ".bak"

    if isfile(filepath)
        cp(filepath, bakfile; force = true)
    end

    open(tmpfile, "w") do io
        println(io, "# U_over_t   energy   sem")
        for U_over_t in sort(collect(keys(data)))
            E, sem = data[U_over_t]
            println(io, "$(U_over_t) $(E) $(sem)")
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
        println(io, "# U_over_t   E_kin   E_kin_sem   E_pot   E_pot_sem")
        for U_over_t in sort(collect(keys(data)))
            Ekin, Ekin_sem, Epot, Epot_sem = data[U_over_t]
            println(io, "$(U_over_t) $(Ekin) $(Ekin_sem) $(Epot) $(Epot_sem)")
        end
    end

    mv(tmpfile, filepath; force = true)
end


# ============================================================
# Main benchmark
# ============================================================

function run_jastrow_benchmark()

    # -----------------------
    # System parameters
    # -----------------------
    L = 60
    N_target = 60
    n_max = N_target

    # -----------------------
    # Convention control
    # -----------------------
    compare_with_capello = true

    # U_over_t_vals = [2.0, 2.4, 2.5, 3.0, 4.0, 6.0]
    U_over_t_vals = [2.5, 3.0, 4.0, 6.0]


    if compare_with_capello
        # Capello convention:
        #   H = -(t_Capello / 2) K + (U / 2) ∑ᵢ nᵢ(nᵢ - 1)
        #
        # Our code convention:
        #   H = -t_code K + (U / 2) ∑ᵢ nᵢ(nᵢ - 1)
        #
        # Therefore:
        #   t_code = t_Capello / 2
        #
        # U_over_t_vals are interpreted as U / t_Capello.
        t_capello = 1.0
        t_code = t_capello / 2.0
        U_vals = U_over_t_vals .* t_capello

        convention_label = "Capello"
    else
        # Native code convention:
        #   H = -t K + (U / 2) ∑ᵢ nᵢ(nᵢ - 1)
        #
        # U_over_t_vals are interpreted as U / t_code.
        t_code = 1.0
        U_vals = U_over_t_vals .* t

        t_capello = NaN
        convention_label = "code"
    end

    μ_vals = zeros(length(U_over_t_vals))

    # Initial REAL-SPACE Jastrow parameters, including R = 0
    # vr_init = zeros(fld(L, 2) + 1)
    vr_init = [1.7027316860894737, 0.8387568263618391, 0.46810508982724464, 0.2576489514709566, 0.12620697420709975, 0.05131447988471431, 0.0035099467543947965, -0.03505626048074154, -0.06448041856381605, -0.08741093274701328, -0.10676526218097374, -0.12200076551451595, -0.1333735882714788, -0.13796849569377204, -0.14426778943707477, -0.14878223661639542, -0.15900846790830805, -0.1683063599530637, -0.17333107735993109, -0.1739512728233899, -0.16780669284780914, -0.161321735974031, -0.16318798914609078, -0.16278323072981474, -0.15914932078798238, -0.16746431343151214, -0.16050428528256053, -0.1645451916168923, -0.16478619200550473, -0.16083574450381002, -0.16118632620152615]

    dim = "1D"
    grand_canonical = false
    projective = false

    lattice = Lattice1D(L)
    ensemble = !grand_canonical ? "C" : "GC"

    dir_base = "../data/$(ensemble)/$(dim)/L$(L)_N$(N_target)/jastrow_realspace"
    mkpath(dir_base)

    # Summary files indexed by Capello-style U/t, not raw U
    results_file = "$(dir_base)/VMC_results_vs_UoverT.dat"
    energy_parts_file = "$(dir_base)/VMC_energy_parts_vs_UoverT.dat"

    println("results_file = ", results_file)
    println("energy_parts_file = ", energy_parts_file)
    println("Using Capello convention:")
    println("    t_capello = ", t_capello)
    println("    t_code passed to System = ", t_code)
    println("    U/t scan values = ", U_over_t_vals)
    println("    corresponding code U values = ", U_vals)

    results_dict = load_results_dict(results_file)
    energy_parts_dict = load_energy_parts_dict(energy_parts_file)

    println("Initial merged results_dict keys = ", sort(collect(keys(results_dict))))
    println("Initial merged energy_parts_dict keys = ", sort(collect(keys(energy_parts_dict))))

    results = []

    # -----------------------
    # Loop over U/t values
    # -----------------------
    for (U_over_t, U, μ) in zip(U_over_t_vals, U_vals, μ_vals)

        U_str = format_u_filename(U_over_t)
        U_key = normalize_u_key(U_over_t)

        println("\n==================================================")
        println("Optimizing Jastrow parameters")
        println("Convention = $convention_label")
        println("Input label U/t = $U_over_t")

        if compare_with_capello
            println("Capello values: t_Capello = $t_capello, U/t_Capello = $U_over_t")
            println("Code values: t_code = $t_code, U = $U, raw code U/t_code = $(U / t_code)")
        else
            println("Code values: t_code = $t_code, U = $U, raw code U/t_code = $(U / t_code)")
        end

        println("μ = $μ")
        println("==================================================")

        sys = System(t_code, U, μ, lattice)

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
            num_MC_steps = 20_000,
            num_equil_steps = 3_000,
            block_size = 1_000,
            z = 1.0
        )

        # If no optimization desired, use:
        # params_opt, history = params_init, []

        println("Optimal vr = ", params_opt.vr)

        # -----------------------
        # Save optimized v(r)
        # -----------------------
        vr = copy(params_opt.vr)
        vr_tail_subtracted = vr .- vr[end]
        Rmax = length(vr_tail_subtracted) - 1

        vr_file = "$(dir_base)/VMC_vr_vs_r_UoverT$(U_str).dat"
        open(vr_file, "w") do io
            println(io, "# r   v_r_tail_subtracted")
            for r in 0:Rmax
                println(io, "$(r) $(vr_tail_subtracted[r + 1])")
            end
        end

        println("Saved v(r) data to ", vr_file)

        # -----------------------
        # Fourier transform v(r) -> v(q)
        # -----------------------
        q, vq, vq_q2 = jastrow_vq_q2_for_plot(
            vr,
            L;
            includes_R0 = true,
            subtract_offset = true
        )

        vq_file = "$(dir_base)/VMC_vq_vs_q_UoverT$(U_str).dat"
        open(vq_file, "w") do io
            println(io, "# q   v_q   v_q_times_q2")
            for i in eachindex(q)
                println(io, "$(q[i]) $(vq[i]) $(vq_q2[i])")
            end
        end

        println("Saved Fourier-transformed v_q data to ", vq_file)

        # -----------------------
        # Optional: save optimization history for this U/t
        # -----------------------
        history_file = "$(dir_base)/SR_history_UoverT$(U_str).dat"
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
            num_walkers = 100,
            num_MC_steps = 5_000,
            num_equil_steps = 1_000,
            block_size = 20_000
        )

        acceptance_ratio = final_result.acceptance_ratio
        println("Acceptance Ratio: $acceptance_ratio")

        energies = final_result.energies
        τE = estimate_tau(energies)

        println("Estimated autocorrelation time τ = ", τE)
        println("Effective sample size ≈ ", length(energies) / (2 * τE))

        push!(
            results,
            (
                U_over_t = U_over_t,
                U = U,
                t_code = t_code,
                t_capello = t_capello,
                params = params_opt,
                result = final_result
            )
        )

        if grand_canonical
            hist_file = "$(dir_base)/PN_hist_UoverT$(U_str).dat"
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
        # Update summary files for this U/t only
        # -----------------------
        println("Updating summary entries for Capello U/t = ", U_key)

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
        # Save a one-line run summary per U/t
        # -----------------------
        summary_file = "$(dir_base)/run_summary_UoverT$(U_str).txt"
        open(summary_file, "w") do io
            println(io, "compare_with_capello = $compare_with_capello")
            println(io, "convention_label = $convention_label")
            println(io, "input_U_over_t = $U_over_t")
            println(io, "t_code_passed_to_System = $t_code")
            println(io, "U_passed_to_System = $U")
            println(io, "raw_code_U_over_t_code = $(U / t_code)")

            if compare_with_capello
                println(io, "t_capello = $t_capello")
                println(io, "Capello_U_over_t = $U_over_t")
            end

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

        println("Saved all outputs for Capello U/t = $U_over_t")
    end

    return results
end


run_jastrow_benchmark()