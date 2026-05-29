push!(LOAD_PATH, joinpath(dirname(@__FILE__), "src"))

using VMCBoseHubbard
using ArgParse
using Random
using TimerOutputs
using Printf

# ============================================================
# Command line parsing
# ============================================================

function parse_commandline()

    s = ArgParseSettings()

    @add_arg_table s begin

        "--length", "-L"
            help = "Size of 1D lattice"
            arg_type = Int
            required = true
            dest_name = "L"

        "--particle-number", "-N"
            help = "Total number of particles"
            arg_type = Int
            required = true
            dest_name = "N"

        "--interaction", "-U"
            help = "Interaction strength"
            arg_type = Float64
            required = true
            dest_name = "U"

        "--t", "-t"
            help = "Hopping parameter"
            arg_type = Float64
            default = 1.0

        "--trial-state"
            help = "Trial state type (gutzwiller, jastrow)"
            arg_type = String
            required = true

        "--kappa"
            help = "Initial Gutzwiller parameter"
            arg_type = Float64
            default = 1.0

        "--jastrow-potentials"
            help = "Initial Jastrow parameters"
            arg_type = String
            default = ""
            dest_name = "vr_init"

        "--n-max"
            help = "Maximum site occupancy"
            arg_type = Int
            default = -1
            dest_name = "n_max"

        "--eta"
            help = "SR learning rate"
            arg_type = Float64
            default = 0.01

        "--seed"
            help = "Random number generator seed"
            arg_type = Int
            default = 1234

        # ====================================================
        # Optimization parameters
        # ====================================================

        "--opt-num-walkers"
            help = "Optimization walkers"
            arg_type = Int
            default = 100
            dest_name = "opt_num_walkers"

        "--opt-num-MC-steps"
            help = "Optimization MC steps"
            arg_type = Int
            default = 5_000
            dest_name = "opt_num_MC_steps"

        "--opt-num-equil-steps"
            help = "Optimization equilibration steps"
            arg_type = Int
            default = 1_000
            dest_name = "opt_num_equil_steps"

        "--opt-block-size"
            help = "Optimization block size"
            arg_type = Int
            default = 500
            dest_name = "opt_block_size"

        # ====================================================
        # Final MC parameters
        # ====================================================

        "--final-num-walkers"
            help = "Final MC walkers"
            arg_type = Int
            default = 100
            dest_name = "final_num_walkers"

        "--final-num-MC-steps"
            help = "Final MC steps"
            arg_type = Int
            default = 5_000
            dest_name = "final_num_MC_steps"

        "--final-num-equil-steps"
            help = "Final MC equilibration steps"
            arg_type = Int
            default = 1_000
            dest_name = "final_num_equil_steps"

        "--final-block-size"
            help = "Final MC block size"
            arg_type = Int
            default = 500
            dest_name = "final_block_size"

        # ====================================================
        # Output control
        # ====================================================

        "--output-dir"
            help = "Base output directory"
            arg_type = String
            default = "data/VMC"

        "--save-history"
            help = "Save SR optimization history"
            action = :store_true

        "--skip-timing"
            help = "Disable timing output"
            action = :store_true
    end

    args = parse_args(s)

    Random.seed!(args["seed"])

    if args["L"] <= 1
        error("Please input a system size larger than one")
    end

    if args["N"] == 0
        error("Please input a non-zero particle number")
    end

    if args["vr_init"] == ""
        args["vr_init"] = zeros(fld(args["L"], 2) + 1)
    end

    if args["n_max"] == -1
        args["n_max"] = args["N"]
    end

    return args
end


# ============================================================
# Helper formatting
# ============================================================

function format_param(x::Real)
    return string(round(x; digits = 6))
end


# ============================================================
# Parameter file
# ============================================================

function write_parameters_file(
    filepath::String,
    args
)

    open(filepath, "w") do io

        println(io, "# ==================================================")
        println(io, "# System Parameters")
        println(io, "# ==================================================")

        println(io, "L $(args["L"])")
        println(io, "N $(args["N"])")
        println(io, "U $(args["U"])")
        println(io, "t $(args["t"])")
        println(io, "U_over_t $(args["U"] / args["t"])")
        println(io, "n_max $(args["n_max"])")
        println(io, "trial_state $(args["trial-state"])")
        println(io, "seed $(args["seed"])")

        println(io)

        println(io, "# ==================================================")
        println(io, "# Optimization Parameters")
        println(io, "# ==================================================")

        println(io, "eta $(args["eta"])")
        println(io, "opt_num_walkers $(args["opt_num_walkers"])")
        println(io, "opt_num_MC_steps $(args["opt_num_MC_steps"])")
        println(io, "opt_num_equil_steps $(args["opt_num_equil_steps"])")
        println(io, "opt_block_size $(args["opt_block_size"])")

        println(io)

        println(io, "# ==================================================")
        println(io, "# Final MC Parameters")
        println(io, "# ==================================================")

        println(io, "final_num_walkers $(args["final_num_walkers"])")
        println(io, "final_num_MC_steps $(args["final_num_MC_steps"])")
        println(io, "final_num_equil_steps $(args["final_num_equil_steps"])")
        println(io, "final_block_size $(args["final_block_size"])")
    end
end


# ============================================================
# Output helpers
# ============================================================

function write_energy_results(
    filepath::String,
    U_over_t::Float64,
    result
)

    open(filepath, "w") do io

        println(io, "# U_over_t   energy   sem")

        println(
            io,
            "$(U_over_t) " *
            "$(result.mean_energy) " *
            "$(result.sem_energy)"
        )
    end
end


function write_energy_parts(
    filepath::String,
    U_over_t::Float64,
    result
)

    open(filepath, "w") do io

        println(
            io,
            "# U_over_t   E_kin   E_kin_sem   E_pot   E_pot_sem"
        )

        println(
            io,
            "$(U_over_t) " *
            "$(result.mean_kinetic) " *
            "$(result.sem_kinetic) " *
            "$(result.mean_potential) " *
            "$(result.sem_potential)"
        )
    end
end


function write_gutzwiller_parameter(
    filepath::String,
    κ::Float64
)

    open(filepath, "w") do io

        println(io, "# kappa")
        println(io, κ)
    end
end


function write_jastrow_potentials(
    filepath::String,
    vr::Vector{Float64}
)

    Rmax = length(vr) - 1

    open(filepath, "w") do io

        println(io, "# r   v_r")

        for r in 0:Rmax
            println(io, "$(r) $(vr[r + 1])")
        end
    end
end


function write_sr_history(
    filepath::String,
    history
)

    isempty(history) && return

    num_params = length(history[1].gradient)

    open(filepath, "w") do io

        println(
            io,
            "# iter energy " *
            join(["g_$i" for i in 1:num_params], " ")
        )

        for (iter, h) in enumerate(history)

            println(
                io,
                "$(iter) $(h.energy) " *
                join(string.(h.gradient), " ")
            )
        end
    end
end


# ============================================================
# Main
# ============================================================

function main()

    to = TimerOutput()

    args = parse_commandline()

    println("Run with parameters:")

    for (arg, val) in args
        println(@sprintf "  %35s  =>  %15s" arg val)
    end

    if args["skip-timing"]
        disable_timer!(to)
    end

    # ========================================================
    # System parameters
    # ========================================================

    L = args["L"]
    N = args["N"]
    U = args["U"]
    t = args["t"]

    U_over_t = U / t

    trial_state = args["trial-state"]

    # ========================================================
    # Output directory structure
    # ========================================================

    base_dir = args["output-dir"]

    trial_state_dir = joinpath(
        base_dir,
        trial_state
    )

    system_dir = joinpath(
        trial_state_dir,
        "L$(L)_N$(N)"
    )

    interaction_dir = joinpath(
        system_dir,
        "U$(format_param(U))_t$(format_param(t))"
    )

    mkpath(interaction_dir)

    # ========================================================
    # Output file paths
    # ========================================================

    parameters_file = joinpath(
        interaction_dir,
        "parameters.dat"
    )

    energy_file = joinpath(
        interaction_dir,
        "energy.dat"
    )

    energy_parts_file = joinpath(
        interaction_dir,
        "energy_parts.dat"
    )

    history_file = joinpath(
        interaction_dir,
        "sr_history.dat"
    )

    # ========================================================
    # Construct system
    # ========================================================

    lattice = Lattice1D(L)

    sys = System(t, U, N, lattice)

    # ========================================================
    # Construct trial state
    # ========================================================

    if trial_state == "gutzwiller"

        wavefunction_init = GutzwillerWavefunction(
            args["kappa"],
            args["n_max"]
        )

    elseif trial_state == "jastrow"

        wavefunction_init = JastrowWavefunction(
            copy(args["vr_init"])
        )

    else

        error(
            "Unknown --trial-state \"$trial_state\". " *
            "Valid options: \"gutzwiller\", \"jastrow\"."
        )
    end

    # ========================================================
    # Optimization
    # ========================================================

    wavefunction_opt = nothing
    history = nothing

    @timeit to "Gradient Descent" begin

        wavefunction_opt, history = optimize_SR(
            sys,
            wavefunction_init,
            args["n_max"];

            η               = args["eta"],
            num_walkers     = args["opt_num_walkers"],
            num_MC_steps    = args["opt_num_MC_steps"],
            num_equil_steps = args["opt_num_equil_steps"],
            block_size      = args["opt_block_size"]
        )
    end

    # ========================================================
    # Final MC run
    # ========================================================

    final_result = nothing

    @timeit to "High Statistics MC" begin

        final_result = MC_integration(
            sys,
            wavefunction_opt,
            args["n_max"];

            num_walkers     = args["final_num_walkers"],
            num_MC_steps    = args["final_num_MC_steps"],
            num_equil_steps = args["final_num_equil_steps"],
            block_size      = args["final_block_size"]
        )
    end

    # ========================================================
    # Write outputs
    # ========================================================

    @timeit to "Write Results to Disk" begin

        write_parameters_file(
            parameters_file,
            args
        )

        write_energy_results(
            energy_file,
            U_over_t,
            final_result
        )

        write_energy_parts(
            energy_parts_file,
            U_over_t,
            final_result
        )

        if wavefunction_opt isa JastrowWavefunction

            vr_file = joinpath(
                interaction_dir,
                "vr.dat"
            )

            write_jastrow_potentials(
                vr_file,
                wavefunction_opt.vr
            )

        elseif wavefunction_opt isa GutzwillerWavefunction

            kappa_file = joinpath(
                interaction_dir,
                "kappa.dat"
            )

            write_gutzwiller_parameter(
                kappa_file,
                wavefunction_opt.κ
            )
        end

        if args["save-history"]

            write_sr_history(
                history_file,
                history
            )
        end
    end

    # ========================================================
    # Timing summary
    # ========================================================

    show(to)
end


if abspath(PROGRAM_FILE) == @__FILE__

    main()
end