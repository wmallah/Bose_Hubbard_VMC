import ..VMCBoseHubbard: MC_integration
import ..VMCBoseHubbard: estimate_energy_gradient_and_metric

export optimize_kappa

#=
Purpose: run gradient descent to determine optimal variational parameter value which minimizes the total energy
Input: sys (system struct), N_target (target value for total number of particles), n_max (maximum site occupancy), grand_canonical (true for grand_canonical move proposal, false for canonical move_proposal), projective (true for projective measurements, false for non-projective measurements)
Optional Input: κ_init (initial guess for variational parameter), η (gradient learning rate), num_walkers (number of walkers/configurations), num_MC_steps (number of Monte Carlo steps), num_equil_steps (number of equilibration steps)
Output: optimal variational parameter value and list of all previous variational parameter values
Author: Will Mallah
Last Updated: 01/25/26
=#
function optimize_kappa(sys::System, N_target::Int, n_max::Int, grand_canonical::Bool,
                          projective::Bool;
                          κ_init::Float64 = 1.0,
                          η::Float64 = 0.05,
                          num_walkers::Int = 200,
                          num_MC_steps::Int = 2000,
                          num_equil_steps::Int = 500,)

    # Set varitional parameter value to initial guess and define list to store all variational parameter values
    κ = κ_init
    history = []

    # ---- Initial evaluation ----
    result_old = MC_integration(
        sys, N_target, κ, n_max, grand_canonical, projective;
        num_walkers = num_walkers,
        num_MC_steps = num_MC_steps,
        num_equil_steps = num_equil_steps,
    )

    # Intial energy and gradient
    E_old   = result_old.mean_energy
    err_old = result_old.sem_energy
    grad, S = estimate_energy_gradient_and_metric(result_old)

    # Regularization (very important)
    λ = 1e-6
    κ -= η * grad / (S + λ)

    push!(history, (κ = κ, energy = E_old, gradient = grad))

    # ---- First mandatory update ----
    κ -= η * grad
    κ = clamp(κ, 1e-12, 10.0)

    # Run varitional parameter optimization until change in energy is smaller than Monte Carlo error
    while true
        result_new = MC_integration(
            sys, N_target, κ, n_max, grand_canonical, projective;
            num_walkers = num_walkers,
            num_MC_steps = num_MC_steps,
            num_equil_steps = num_equil_steps,
        )

        # New energy, error for stopping condition, and new gradient for next step
        E_new   = result_new.mean_energy
        err_new = result_new.sem_energy
        grad, S = estimate_energy_gradient_and_metric(result_new)

        # Regularization (very important)
        λ = 1e-6
        κ -= η * grad / (S + λ)

        # Print quick results
        println("κ = $(round(κ, digits=15))  E = $(round(E_new, digits=8)) ± $(round(err_new, digits=8))")

        push!(history, (κ = κ, energy = E_new, gradient = grad))

        # ---- Statistical stopping condition ----
        if abs(E_new - E_old) < sqrt(err_new^2 + err_old^2)
            break
        end

        # ---- Gradient descent update ----
        κ -= η * grad
        κ = clamp(κ, 1e-15, 10.0)

        if abs(η * grad) > 0.5κ
            η *= 0.5
        end

        # Stop and warn if either variational parameter or energy gradient are non-finite
        if !isfinite(κ) || !isfinite(grad)
            @warn "Stopping: non-finite κ or gradient"
            break
        end

        # Set new energy as the old energy for next step in loop
        E_old = E_new
        err_old = err_new

    end

    final_κ = κ

    return final_κ, history
end