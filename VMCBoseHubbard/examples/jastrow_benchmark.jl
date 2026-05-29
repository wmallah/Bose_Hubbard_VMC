using Pkg
Pkg.activate("../")

include("../src/VMCBoseHubbard.jl")
using .VMCBoseHubbard
using FFTW

import .VMCBoseHubbard: estimate_tau
import .VMCBoseHubbard: JastrowWavefunction


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

    # Code convention: save the raw FFT convention.
    # If you want a sign flip for a specific comparison, do that in analysis.
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
    L = 4
    N = 2
    n_max = N

    # User-controlled hopping parameter.
    # The code Hamiltonian is interpreted as:
    #
    #   H = -t * K + (U/2) * sum_i n_i(n_i - 1)
    #
    # Therefore U_over_t_vals below mean U / t in this code convention.
    t = 1.0

    # -----------------------
    # U/t-controlled scan in code convention
    # -----------------------

    # Example scans:
    # U_vals = [0.0:1.0:10.0;]
    U_vals = [0.0, 3.0, 3.3, 3.4, 4.0, 6.0]
    # U_vals = [3.3578]

    U_over_t_vals = U_vals ./ t

    # Initial REAL-SPACE Jastrow parameters, including R = 0
    vr_init = zeros(fld(L, 2) + 1)

    dim = "1D"

    lattice = Lattice1D(L)

    dir_base = "../data/VMC/$(dim)/L$(L)_N$(N)/jastrow_realspace"
    mkpath(dir_base)

    # Summary files indexed by code-convention U/t
    results_file = "$(dir_base)/VMC_results_vs_U_over_t.dat"
    energy_parts_file = "$(dir_base)/VMC_energy_parts_vs_U_over_t.dat"

    println("results_file = ", results_file)
    println("energy_parts_file = ", energy_parts_file)
    println("Using code convention:")
    println("    Hamiltonian hopping parameter t = ", t)
    println("    Hamiltonian interaction parameter U = ", U_vals)
    println("    U/t values = ", U_over_t_vals)

    results_dict = load_results_dict(results_file)
    energy_parts_dict = load_energy_parts_dict(energy_parts_file)

    println("Initial merged results_dict keys = ", sort(collect(keys(results_dict))))
    println("Initial merged energy_parts_dict keys = ", sort(collect(keys(energy_parts_dict))))

    results = []

    # -----------------------
    # Loop over U/t values
    # -----------------------
    for (U_over_t, U) in zip(U_over_t_vals, U_vals)

        U_str = format_u_filename(U_over_t)
        U_key = normalize_u_key(U_over_t)

        println("\n==================================================")
        println("Optimizing Jastrow parameters")
        println("Input code-convention U/t = $U_over_t")
        println("Code values: t = $t, U = $U")
        println("==================================================")

        sys = System(t, U, N, lattice)

        wavefunction_init = JastrowWavefunction(copy(vr_init))

        # -----------------------
        # Optimize Jastrow parameters (SR)
        # -----------------------
        wavefunction_opt, history = optimize_SR(
            sys,
            wavefunction_init,
            n_max;
            η = 0.01,
            num_walkers = 100,
            num_MC_steps = 5_000,
            num_equil_steps = 1_000,
            block_size = 500
        )

        # If no optimization desired, use:
        # wavefunction_opt, history = wavefunction_init, []

        println("Optimal vr = ", wavefunction_opt.vr)

        # -----------------------
        # Save optimized v(r)
        # -----------------------
        vr = copy(wavefunction_opt.vr)
        Rmax = length(vr) - 1

        vr_file = "$(dir_base)/VMC_vr_vs_r_U_over_t$(U_str).dat"
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
        q, vq, vq_q2 = jastrow_vq_q2_for_plot(
            vr,
            L;
            includes_R0 = true,
            subtract_offset = true
        )

        vq_file = "$(dir_base)/VMC_vq_vs_q_U_over_t$(U_str).dat"
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
        history_file = "$(dir_base)/SR_history_U_over_t$(U_str).dat"
        open(history_file, "w") do io
            println(io, "# iter   energy   " * join(["g_$i" for i in 1:length(vr)], " "))
            for (iter, h) in enumerate(history)
                println(io, "$(iter) $(h.energy) " * join(string.(h.gradient), " "))
            end
        end

        # --------------------------------
        # Use optimized v(r) as next guess
        # --------------------------------
        vr_init .= wavefunction_opt.vr

        # -----------------------
        # Final high-statistics evaluation
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
        println("Effective sample size ≈ ", length(energies) / (2 * τE))

        push!(
            results,
            (
                U_over_t = U_over_t,
                U = U,
                t = t,
                wavefunction = wavefunction_opt,
                result = final_result
            )
        )

        # -----------------------
        # Update summary files for this U/t only
        # -----------------------
        println("Updating summary entries for code-convention U/t = ", U_key)

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
        summary_file = "$(dir_base)/run_summary_U_over_t$(U_str).txt"
        open(summary_file, "w") do io
            println(io, "input_U_over_t = $U_over_t")
            println(io, "t = $t")
            println(io, "U = $U")
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

        println("Saved all outputs for code-convention U/t = $U_over_t")
    end

    return results
end


run_jastrow_benchmark()