push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using VMCBoseHubbard
using Random
using Statistics
using LinearAlgebra
using Dates

const STUDY_INTERACTIONS = [1.0, 3.3578, 6.0]
const STUDY_TRIAL_STATES = ["gutzwiller", "jastrow"]
const STUDY_ETAS = [0.0025, 0.005, 0.01, 0.02]
const STUDY_ETA_DECAYS = [0.998, 0.98, 0.95, 0.90]
const L = 16
const N = 16
const T = 1.0

function parse_options(args)
    options = Dict{String,Any}(
        "output-dir" => nothing,
        "seed" => 1234,
        "walkers" => 100,
        "mc-steps" => 266_144,
        "equil-steps" => 4_000,
        "opt-walkers" => 100,
        "opt-mc-steps" => 10_000,
        "opt-equil-steps" => 2_000,
        "block-size" => 50_000,
        "max-iters" => 50,
        "etas" => copy(STUDY_ETAS),
        "eta-decays" => copy(STUDY_ETA_DECAYS),
        "autocorrelation-only" => false,
        "updates-only" => false,
        "interaction" => nothing,
        "trial-state" => nothing,
    )

    value_options = Set([
        "--output-dir", "--seed", "--walkers", "--mc-steps",
        "--equil-steps", "--opt-walkers", "--opt-mc-steps",
        "--opt-equil-steps", "--block-size", "--max-iters",
        "--etas", "--eta-decays", "--interaction", "--trial-state",
    ])
    flag_options = Set(["--autocorrelation-only", "--updates-only", "--help", "-h"])

    i = 1
    while i <= length(args)
        option = args[i]
        if option in ("--help", "-h")
            print_study_help()
            exit(0)
        elseif option in flag_options
            options[option == "--autocorrelation-only" ? "autocorrelation-only" : "updates-only"] = true
            i += 1
        elseif option in value_options
            i < length(args) || error("Missing value for $option")
            value = args[i + 1]
            key = option[3:end]
            if key == "output-dir" || key == "trial-state"
                options[key] = value
            elseif key in ("etas", "eta-decays")
                options[key] = parse.(Float64, split(value, ','))
            elseif key == "interaction"
                options[key] = parse(Float64, value)
            elseif key == "seed"
                options[key] = parse(Int, value)
            else
                options[key] = parse(Int, value)
            end
            i += 2
        else
            error("Unknown option: $option")
        end
    end

    options["autocorrelation-only"] && options["updates-only"] &&
        error("Choose at most one of --autocorrelation-only and --updates-only")

    for key in ("walkers", "mc-steps", "opt-walkers", "opt-mc-steps",
                "block-size", "max-iters")
        options[key] > 0 || error("--$key must be positive")
    end
    for key in ("equil-steps", "opt-equil-steps")
        options[key] >= 0 || error("--$key must be non-negative")
    end
    options["equil-steps"] < options["mc-steps"] ||
        error("--equil-steps must be less than --mc-steps")
    options["mc-steps"] - options["equil-steps"] >= 16 ||
        error("At least 16 post-equilibration sweeps are required for blocking analysis")
    options["opt-equil-steps"] < options["opt-mc-steps"] ||
        error("--opt-equil-steps must be less than --opt-mc-steps")
    options["block-size"] <= options["opt-walkers"] * (options["opt-mc-steps"] - options["opt-equil-steps"]) ||
        error("--block-size exceeds the number of optimization samples")
    !isempty(options["etas"]) && all(>(0), options["etas"]) ||
        error("--etas must be a comma-separated list of positive values")
    !isempty(options["eta-decays"]) &&
        all(decay -> 0.0 < decay <= 1.0, options["eta-decays"]) ||
        error("--eta-decays must be a comma-separated list in (0, 1]")
    options["interaction"] === nothing || options["interaction"] in STUDY_INTERACTIONS ||
        error("--interaction must be one of $(join(STUDY_INTERACTIONS, ", "))")
    options["trial-state"] === nothing || options["trial-state"] in STUDY_TRIAL_STATES ||
        error("--trial-state must be gutzwiller or jastrow")

    return options
end

function print_study_help()
    println("""
    Run the 1D L=N=16 default-parameter study at U/t = 1.0, 3.3578, 6.0
    for Gutzwiller and Jastrow trial states.

    Usage: julia --project=VMCBoseHubbard VMCBoseHubbard/examples/default_parameter_study.jl [options]

    Options:
      --output-dir PATH       New output directory (default: timestamped under data/studies)
      --seed INT              Base random seed (default: 1234)
      --walkers INT           Autocorrelation walkers (default: 100)
      --mc-steps INT          Total autocorrelation MC steps (default: 266144; 262144 measured sweeps)
      --equil-steps INT       Autocorrelation equilibration steps (default: 4000)
      --opt-walkers INT       Optimization walkers (default: 100)
      --opt-mc-steps INT      Total MC steps per SR iteration (default: 10000)
      --opt-equil-steps INT   SR equilibration steps (default: 2000)
      --block-size INT        SR samples per block (default: 50000; 500 sweeps)
      --max-iters INT         SR iteration limit (default: 50)
      --etas A,B,...          Initial SR learning rates (default: 0.0025,0.005,0.01,0.02)
      --eta-decays A,B,...    Multiplicative LR decay (default: 0.998,0.98,0.95,0.90)
      --interaction FLOAT     Run one of the three interaction values only
      --trial-state NAME      Run gutzwiller or jastrow only
      --autocorrelation-only  Skip the SR update study
      --updates-only          Skip the autocorrelation study
      -h, --help              Show this help
    """)
end

function selected_cases(options)
    interactions = options["interaction"] === nothing ?
                   STUDY_INTERACTIONS : [options["interaction"]]
    trial_states = options["trial-state"] === nothing ?
                   STUDY_TRIAL_STATES : [options["trial-state"]]
    return [(state, U) for state in trial_states for U in interactions]
end

function initial_wavefunction(state, lattice)
    if state == "gutzwiller"
        return GutzwillerWavefunction(1.0, N)
    end
    return JastrowWavefunction(zeros(length(lattice_shell_distances(lattice))))
end

function case_seed(seed, state, U)
    interaction_index = findfirst(==(U), STUDY_INTERACTIONS)
    state_offset = state == "jastrow" ? 10_000 : 0
    return seed + state_offset + 100 * interaction_index
end

function blocking_rows(energies)
    raw_variance = var(energies)
    rows = NamedTuple[]
    block_means = copy(energies)
    block_sweeps = 1

    while length(block_means) >= 16
        n_blocks = length(block_means)
        tau_int = raw_variance > 0 ?
                  0.5 * block_sweeps * var(block_means) / raw_variance : 0.5
        push!(rows, (
            block_sweeps = block_sweeps,
            n_blocks = n_blocks,
            mean_energy = mean(block_means),
            sem_energy = std(block_means) / sqrt(n_blocks),
            tau_int_sweeps = tau_int,
            effective_sweeps = length(energies) / (2 * tau_int),
        ))

        paired_count = fld(n_blocks, 2)
        paired_count < 8 && break
        block_means = [
            (block_means[2i - 1] + block_means[2i]) / 2
            for i in 1:paired_count
        ]
        block_sweeps *= 2
    end

    return rows
end

function csv_value(value)
    return string(value)
end

function write_autocorrelation_study(path, timeseries_path, options, cases, lattice)
    open(path, "w") do io
        println(io, "trial_state,U_over_t,block_sweeps,n_blocks,mean_energy,sem_energy,tau_int_sweeps,effective_sweeps")
        open(timeseries_path, "w") do series_io
            println(series_io, "trial_state,U_over_t,sweep,energy")

            for (state, U) in cases
                println("Autocorrelation: $state, U/t=$U")
                wf = initial_wavefunction(state, lattice)
                sys = System(T, U, N, lattice)
                Random.seed!(case_seed(options["seed"], state, U))
                result = MC_integration(
                    sys, wf, N;
                    num_walkers = options["walkers"],
                    num_MC_steps = options["mc-steps"],
                    num_equil_steps = options["equil-steps"],
                    block_size = options["walkers"],
                )

                expected_sweeps = options["mc-steps"] - options["equil-steps"]
                length(result.energies) == expected_sweeps ||
                    error("Expected one energy mean per post-equilibration sweep for $state at U/t=$U; got $(length(result.energies)) of $expected_sweeps")

                for (sweep, energy) in enumerate(result.energies)
                    println(series_io, "$state,$U,$sweep,$(csv_value(energy))")
                end
                for row in blocking_rows(result.energies)
                    println(io, join((
                        state, U, row.block_sweeps, row.n_blocks,
                        row.mean_energy, row.sem_energy, row.tau_int_sweeps,
                        row.effective_sweeps,
                    ), ","))
                end
            end
        end
    end
end

function write_update_study(path, options, cases, lattice)
    n_params = length(lattice_shell_distances(lattice))
    open(path, "w") do io
        fields = [
            "trial_state", "U_over_t", "eta_initial", "eta_decay",
            "iteration", "learning_rate",
            "energy", "sem_energy", "gradient_norm", "update_rms",
            "predicted_drop", "resolved_fraction",
        ]
        append!(fields, ["parameter_$i" for i in 1:n_params])
        append!(fields, ["gradient_$i" for i in 1:n_params])
        println(io, join(fields, ","))

        for (state, U) in cases
            sys = System(T, U, N, lattice)
            for eta in options["etas"]
                for eta_decay in options["eta-decays"]
                    println("SR updates: $state, U/t=$U, eta=$eta, decay=$eta_decay")
                    wf_init = initial_wavefunction(state, lattice)
                    Random.seed!(case_seed(options["seed"], state, U))
                    _, history = optimize_SR(
                        sys, wf_init, N;
                        η = eta,
                        η_decay = eta_decay,
                        num_walkers = options["opt-walkers"],
                        num_MC_steps = options["opt-mc-steps"],
                        num_equil_steps = options["opt-equil-steps"],
                        block_size = options["block-size"],
                        max_iters = options["max-iters"],
                    )

                    for (iteration, entry) in enumerate(history)
                        params = entry.wavefunction
                        gradients = entry.gradient
                        row = Any[
                            state, U, eta, eta_decay, iteration,
                            entry.learning_rate, entry.energy,
                            entry.sem_energy, norm(gradients), entry.rms_step,
                            entry.predicted_drop, entry.resolved_fraction,
                        ]
                        append!(row, params)
                        append!(row, fill("", n_params - length(params)))
                        append!(row, gradients)
                        append!(row, fill("", n_params - length(gradients)))
                        println(io, join(csv_value.(row), ","))
                    end
                end
            end
        end
    end
end

function main(args)
    options = parse_options(args)
    output_dir = options["output-dir"] === nothing ?
                 joinpath(@__DIR__, "..", "data", "studies",
                          "default_parameters_$(Dates.format(now(), "yyyymmdd_HHMMSS"))") :
                 abspath(options["output-dir"])
    ispath(output_dir) && error("Output path already exists: $output_dir")
    mkpath(output_dir)

    cases = selected_cases(options)
    lattice = Lattice1D(L)
    println("Study output: $output_dir")
    println("Cases: ", join(["$state at U/t=$U" for (state, U) in cases], "; "))

    if !options["updates-only"]
        write_autocorrelation_study(
            joinpath(output_dir, "autocorrelation_blocking.csv"),
            joinpath(output_dir, "autocorrelation_timeseries.csv"),
            options, cases, lattice,
        )
    end
    if !options["autocorrelation-only"]
        write_update_study(
            joinpath(output_dir, "sr_update_history.csv"),
            options, cases, lattice,
        )
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
