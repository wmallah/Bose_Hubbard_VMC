# gradient_descent.jl

# ── Parameter helpers ──────────────────────────────────────────────────────────
#=
flatten_params:   extract the variational parameters as a plain Vector.
unflatten_params: reconstruct a wavefunction from an updated parameter vector.
                  Dispatch on the old wavefunction's type determines which
                  constructor to call. Clamping is applied here for Gutzwiller
                  so the optimizer loop itself stays type-agnostic.
=#

flatten_params(wf::GutzwillerWavefunction) = [wf.κ]
flatten_params(wf::JastrowWavefunction)    = copy(wf.vr)

function unflatten_params(v::Vector{<:Real}, ::GutzwillerWavefunction, n_max::Int)
    κ = clamp(v[1], 1e-12, 10.0)
    return GutzwillerWavefunction(κ, n_max)
end

function unflatten_params(v::Vector{<:Real}, ::JastrowWavefunction, ::Int)
    return JastrowWavefunction(v)
end


# ── SNR helper ─────────────────────────────────────────────────────────────────
#=
Returns the signal-to-noise ratio |g_i| / σ(g_i) for each gradient component.
A component is considered statistically resolved if snr[i] > z_grad.
=#
function snr_vector(g::Vector{Float64}, SE_g::Vector{Float64};
                    zero_tol::Float64 = 1e-16)
    snr = similar(g)
    for i in eachindex(g)
        if SE_g[i] > zero_tol
            snr[i] = abs(g[i]) / SE_g[i]
        elseif abs(g[i]) <= zero_tol
            snr[i] = 0.0
        else
            snr[i] = Inf
        end
    end
    return snr
end

function max_resolved_gradient_components(num_params::Int, max_fraction::Float64)
    num_params >= 1 || throw(ArgumentError("num_params must be positive"))
    0.0 <= max_fraction <= 1.0 ||
        throw(ArgumentError("max_fraction must be between 0 and 1"))

    allowed = floor(Int, max_fraction * num_params)
    return num_params > 1 && max_fraction > 0.0 ? max(1, allowed) : allowed
end

function optimization_summary(history::AbstractVector, max_iters::Int; skipped::Bool = false)
    max_iters >= 1 || throw(ArgumentError("max_iters must be positive"))

    if skipped
        return (;
            status = "skipped",
            iterations = 0,
            max_iters,
            converged = false,
            energy = missing,
            sem_energy = missing,
            energy_span = missing,
            energy_resolution = missing,
            energy_flat = false,
            resolved_gradients = missing,
            allowed_resolved_gradients = missing,
            gradient_unresolved = false,
            step_small = false,
            qualifying_iterations = 0,
            required_hits = missing,
        )
    elseif isempty(history)
        return (;
            status = "stopped_before_valid_iteration",
            iterations = 0,
            max_iters,
            converged = false,
            energy = missing,
            sem_energy = missing,
            energy_span = missing,
            energy_resolution = missing,
            energy_flat = false,
            resolved_gradients = missing,
            allowed_resolved_gradients = missing,
            gradient_unresolved = false,
            step_small = false,
            qualifying_iterations = 0,
            required_hits = missing,
        )
    end

    latest = last(history)
    status = latest.converged ? "converged" :
             length(history) >= max_iters ? "max_iterations_reached" :
             "stopped_early"
    return (;
        status,
        iterations = length(history),
        max_iters,
        converged = latest.converged,
        energy = latest.energy,
        sem_energy = latest.sem_energy,
        energy_span = latest.energy_span,
        energy_resolution = latest.energy_resolution,
        energy_flat = latest.energy_flat,
        resolved_gradients = latest.num_resolved,
        allowed_resolved_gradients = latest.max_resolved_components,
        gradient_unresolved = latest.gradient_unresolved,
        step_small = latest.step_small,
        qualifying_iterations = latest.qualifying_iterations,
        required_hits = latest.required_hits,
    )
end


# ── Stochastic Reconfiguration optimizer ──────────────────────────────────────
#=
Minimizes the variational energy using the Stochastic Reconfiguration (SR)
update rule:
    Δv = η · (S + λI)⁻¹ · g

where g is the energy gradient and S is the quantum geometric tensor (metric).
Works for any Wavefunction subtype via flatten_params / unflatten_params dispatch.

Convergence is assessed over a rolling window. Criteria:
    1. Only a small fraction of gradient components are statistically resolved.
    2. The energy is flat over a rolling window AND the applied RMS parameter
         step is small.

The gradient fraction avoids an all-components test whose false-positive rate
gets worse as the number of parameters grows. The energy window uses the
reported Monte Carlo uncertainties, while the step criterion uses the actual
possibly clipped update.

kwargs
------
η              : learning rate
η_decay        : multiplicative learning-rate decay applied after each update
λ              : diagonal regularization of S
max_step       : hard clip on ||Δv||; useful for Gutzwiller to avoid large
                 steps early in optimization (set to Inf to disable)
z_grad         : SNR threshold for a gradient component to be "resolved"
max_resolved_fraction : largest allowed resolved-gradient fraction
z_energy        : multiplier on the combined SEM energy plateau threshold
step_atol       : absolute RMS applied-step tolerance
step_rtol       : relative RMS applied-step tolerance
energy_window   : number of recent energies used for the plateau test
required_hits   : qualifying iterations required within the energy window
min_iters      : minimum iterations before convergence is checked
max_iters      : hard iteration cap
=#
function optimize_SR(sys::System,
                     wf::Wavefunction,
                     n_max::Int;
                     η               ::Float64 = 0.05,
                     η_decay         ::Float64 = 0.998,
                     λ               ::Float64 = 1e-3,
                     max_step        ::Float64 = Inf,
                     num_walkers     ::Int     = 200,
                     num_MC_steps    ::Int     = 30000,
                     num_equil_steps ::Int     = 5000,
                     block_size      ::Int     = 200,
                     z_grad          ::Float64 = 3.0,
                     max_resolved_fraction ::Float64 = 0.05,
                     z_energy        ::Float64 = 1.0,
                     step_atol       ::Float64 = 1e-5,
                     step_rtol       ::Float64 = 1e-3,
                     energy_window   ::Int     = 5,
                     required_hits   ::Int     = 4,
                     min_iters       ::Int     = 10,
                     max_iters       ::Int     = 200)

    history = NamedTuple[]
    energy_history = Float64[]
    energy_error_history = Float64[]
    quality_history = Bool[]
    prev_E = nothing
    prev_err = nothing
    converged_now = false

    energy_window >= 1 || throw(ArgumentError("energy_window must be positive"))
    1 <= required_hits <= energy_window ||
        throw(ArgumentError("required_hits must be between 1 and energy_window"))
    max_iters >= 1 || throw(ArgumentError("max_iters must be positive"))
    0.0 <= max_resolved_fraction <= 1.0 ||
        throw(ArgumentError("max_resolved_fraction must be between 0 and 1"))
    0.0 < η_decay <= 1.0 ||
        throw(ArgumentError("η_decay must be greater than 0 and at most 1"))

    for iter in 1:max_iters

        # ── Monte Carlo ───────────────────────────────────────────────────────
        result = MC_integration(sys, wf, n_max;
                                num_walkers     = num_walkers,
                                num_MC_steps    = num_MC_steps,
                                num_equil_steps = num_equil_steps,
                                block_size      = block_size)

        E    = result.mean_energy
        err  = result.sem_energy
        g    = result.gradient
        SE_g = result.gradient_standard_error
        S    = result.metric

        if any(!isfinite, g) || any(!isfinite, SE_g) || any(!isfinite, S)
            @warn "Stopping at iteration $iter: non-finite gradient or metric."
            break
        end

        # ── Natural gradient step ─────────────────────────────────────────────
        direction = (S + λ * I) \ g
        params    = flatten_params(wf)
        Δv        = η .* direction

        # Optional hard clip on step size
        if isfinite(max_step) && norm(Δv) > max_step
            Δv .*= max_step / norm(Δv)
        end

        # ── Convergence diagnostics ───────────────────────────────────────────
        snr            = snr_vector(g, SE_g; zero_tol = 1e-16)
        num_resolved   = count(x -> x > z_grad, snr)
        resolved_fraction = num_resolved / length(g)
        max_resolved_components =
            max_resolved_gradient_components(length(g), max_resolved_fraction)
        predicted_drop = dot(g, Δv)
        rms_step       = norm(Δv) / sqrt(length(Δv))

        if predicted_drop < 0.0
            @warn "SR step is not a descent direction at iteration $iter." predicted_drop
            predicted_drop = 0.0
        end

        # ── Progress output ───────────────────────────────────────────────────
        println("── Iteration $iter ─────────────────────────────")
        println("  Energy           = $(round(E, digits=8)) ± $(round(err, digits=8))")
        println("  Gradient norm    = $(round(norm(g), digits=6))")
        println("  Max SNR          = $(round(maximum(snr), digits=4))")
        println("  Resolved comps   = $num_resolved / $(length(g))")
        println("  Allowed resolved = $max_resolved_components / $(length(g))")
        println("  Applied RMS step = $(round(rms_step, digits=8))")

        if prev_E !== nothing
            ΔE     = abs(E - prev_E)
            ΔE_err = sqrt(err^2 + prev_err^2)
            println("  |ΔE|             = $(round(ΔE, digits=8))")
            println("  σ(ΔE)           = $(round(ΔE_err, digits=8))")
        end

        # ── Convergence test ──────────────────────────────────────────────────
        push!(energy_history, E)
        push!(energy_error_history, err)
        first_window = max(1, length(energy_history) - energy_window + 1)
        recent_energies = @view energy_history[first_window:end]
        recent_errors = @view energy_error_history[first_window:end]
        energy_span = maximum(recent_energies) - minimum(recent_energies)
        energy_resolution = z_energy * sqrt(sum(recent_errors .^ 2))
        energy_flat = length(recent_energies) == energy_window &&
                      energy_span <= energy_resolution
        gradient_unresolved = num_resolved <= max_resolved_components
        step_small = rms_step <= step_atol + step_rtol *
                     max(norm(params) / sqrt(length(params)), 1.0)
        push!(quality_history, gradient_unresolved && step_small)
        recent_quality = @view quality_history[first_window:end]
        converged_now = iter >= min_iters && energy_flat &&
                        count(recent_quality) >= required_hits
        qualifying_iterations = count(recent_quality)

        push!(history, (
            wavefunction   = params,
            learning_rate  = η,
            energy         = E,
            sem_energy     = err,
            gradient       = copy(g),
            snr            = copy(snr),
            predicted_drop = predicted_drop,
            rms_step       = rms_step,
            resolved_fraction = resolved_fraction,
            energy_span    = energy_span,
            energy_resolution = energy_resolution,
            energy_flat    = energy_flat,
            num_resolved   = num_resolved,
            max_resolved_components = max_resolved_components,
            gradient_unresolved = gradient_unresolved,
            step_small     = step_small,
            qualifying_iterations = qualifying_iterations,
            required_hits  = required_hits,
            converged      = converged_now,
        ))

        if converged_now
            println("Converged at iteration $iter.")
            println("  qualifying iterations = ", qualifying_iterations)
            println("  energy_flat        = ", energy_flat)
            println("  step_small        = ", step_small)
            break
        end

        # ── Parameter update ──────────────────────────────────────────────────
        wf       = unflatten_params(flatten_params(wf) .- Δv, wf, n_max)
        prev_E   = E
        prev_err = err
        η       *= η_decay
    end

    if !converged_now && length(history) == max_iters
        diagnostics = optimization_summary(history, max_iters)
        @warn "Reached max_iters=$max_iters before satisfying the convergence criteria." diagnostics
    end

    return wf, history
end