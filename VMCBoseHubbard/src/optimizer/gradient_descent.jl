import ..VMCBoseHubbard: MC_integration
import ..VMCBoseHubbard: estimate_energy_gradient_and_metric

export optimize_kappa

function optimize_kappa(sys::System, N_target::Int, n_max::Int,
                        grand_canonical::Bool, projective::Bool;
                        κ_init::Float64 = 1.0,
                        η::Float64 = 0.05,
                        num_walkers::Int = 200,
                        num_MC_steps::Int = 2000,
                        num_equil_steps::Int = 500,
                        block_size::Int = 200,
                        z::Float64 = 2.0)   # confidence level

    κ = κ_init

    history = Vector{NamedTuple{(:κ,:energy,:gradient,:snr),
               Tuple{Float64,Float64,Float64,Float64}}}()

    λ = 1e-6        # SR regularization
    max_step = 0.2  # safety limiter

    while true

        # --- Monte Carlo ---
        result = MC_integration(
            sys, N_target, κ, n_max, grand_canonical, projective;
            num_walkers = num_walkers,
            num_MC_steps = num_MC_steps,
            num_equil_steps = num_equil_steps,
        )

        E   = result.mean_energy
        err = result.sem_energy

        # --- Statistical gradient + metric ---
        g, SE_g, S = estimate_energy_gradient_and_metric(result;
                                                          block_size=block_size)

        if !isfinite(g) || !isfinite(SE_g) || !isfinite(S)
            @warn "Stopping: non-finite value encountered"
            break
        end

        # Signal to Noise Ratio
        snr = abs(g) / SE_g

        println("κ = $(round(κ, digits=10))  " *
                "E = $(round(E, digits=8)) ± $(round(err, digits=8))  " *
                "g = $(round(g, digits=6))  SNR = $(round(snr, digits=4))")

        push!(history, (κ = κ,
                        energy = E,
                        gradient = g,
                        snr = snr))

        # --- Statistical stopping condition ---
        if abs(g) < z * SE_g
            println("Gradient statistically zero (|g| < $(z)σ). Converged.")
            break
        end

        # --- Natural gradient step (Sochastic Reconfiguration) ---
        Δκ = η * g / (S + λ)

        if abs(Δκ) > max_step
            Δκ = max_step * sign(Δκ)
        end

        κ -= Δκ
        κ = clamp(κ, 1e-12, 10.0)

    end

    return κ, history
end