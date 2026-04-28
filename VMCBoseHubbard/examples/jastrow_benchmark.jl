using Pkg
Pkg.activate("../")

include("../src/VMCBoseHubbard.jl")
using .VMCBoseHubbard
using FFTW

import ..VMCBoseHubbard: MC_integration_Jastrow
import ..VMCBoseHubbard: estimate_tau
import ..VMCBoseHubbard: JastrowParams
import ..VMCBoseHubbard: optimize_jastrow_SR

function reduced_to_full_jastrow(vr_in::Vector{Float64}, L::Int; subtract_offset::Bool=false)
    Rmax = fld(L, 2)
    @assert length(vr_in) == Rmax + 1

    vr = copy(vr_in)

    if subtract_offset
        vr .-= vr[end]
    end

    vfull = zeros(Float64, L)

    # R = 0
    vfull[1] = vr[1]

    # 1 <= R < L/2
    for R in 1:(Rmax - (iseven(L) ? 1 : 0))
        vfull[R + 1] = vr[R + 1]
        vfull[L - R + 1] = vr[R + 1]
    end

    # R = L/2 for even L
    if iseven(L)
        vfull[Rmax + 1] = vr[Rmax + 1]
    end

    return vfull
end

function realspace_to_momentum_jastrow_fft(vr_in::Vector{Float64}, L::Int; subtract_offset::Bool=false)
    vfull = reduced_to_full_jastrow(vr_in, L; subtract_offset=subtract_offset)

    # DFT convention: fq[m+1] = sum_R vfull[R+1] * exp(-2π i m R / L)
    vq_full = real(fft(vfull))

    Rmax = fld(L, 2)
    q_vec = [2π * m / L for m in 0:Rmax]
    vq_vec = vq_full[1:(Rmax + 1)]

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
L = 2
N_target = 2
t = 1.0 # / 2.0
n_max = N_target

# Initial REAL-SPACE Jastrow parameters
vr_init = zeros(fld(L, 2) + 1)
# L = 60, N = 60
# vr_init = [-1.230664048412987, -0.5603479060920281, -0.27660867832731106, -0.1242785567607009, -0.036300220026993776, 0.015542460624502534, 0.04720701326335529, 0.06631426660741839, 0.0769907028776819, 0.08275837621140529, 0.08666069082575337, 0.0885605235609721, 0.09064818664318908, 0.0902966653608874, 0.0909457328011872, 0.09227127580457255, 0.0924555268925304, 0.09315612812079689, 0.09294451392475349, 0.09408782298076765, 0.09507172586719383, 0.09430221366651596, 0.09394029152697604, 0.09293361422192781, 0.09203510919042031, 0.09398848374625114, 0.09466050430029292, 0.09241670543170888, 0.09236268945178183, 0.09283366336796091, 0.09281451965988918]
# L = 16, N = 16
# vr_init = [-0.8965587227480887, -0.32175533626093716, -0.06414963743854304, 0.08031265636802659, 0.16862564901541344, 0.22316137377478304, 0.2566526689716719, 0.27399230706564387, 0.2797190423561562]
# L = 64, N = 64 from Capello
# vr_init = [-0.1625881E+01,
#     -0.1048148E+01,
#     -0.7777309E+00,
#     -0.6105601E+00,
#     -0.4930722E+00,
#     -0.4043681E+00,
#     -0.3347432E+00,
#     -0.2786680E+00,
#     -0.2327923E+00,
#     -0.1949050E+00,
#     -0.1633171E+00,
#     -0.1366961E+00,
#     -0.1143394E+00,
#     -0.9545265E-01,
#     -0.7940817E-01,
#     -0.6579527E-01,
#     -0.5437115E-01,
#     -0.4466728E-01,
#     -0.3659925E-01,
#     -0.2988174E-01,
#     -0.2420967E-01,
#     -0.1944907E-01,
#     -0.1543752E-01,
#     -0.1201605E-01,
#     -0.9099452E-02,
#     -0.6679389E-02,
#     -0.4549107E-02,
#     -0.3009113E-02,
#     -0.1798847E-02,
#     -0.9387513E-03,
#     -0.4239784E-03,
#     -0.4166991E-04,
#     0.0]

# Parameter scans
# U_vals = [2.0, 2.4, 2.5, 3.0, 4.0, 6.0]
# U_vals = [2.5, 3.0, 4.0, 6.0]
# U_vals = 0.0:1.0:10.0
U_vals = [3.3578] # / t
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
        num_walkers = 200,
        num_MC_steps = 8_000,
        num_equil_steps = 2_000,
        block_size = 800,
        z = 1.0
    )
    
    # If no optimization desired, run with this
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
    q, vq = realspace_to_momentum_jastrow_fft(vr, L)
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
        num_MC_steps = 20_000,
        num_equil_steps = 4_000,
        block_size = 80_000
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