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
    return JastrowWavefunction(copy(v))
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


# ── Stochastic Reconfiguration optimizer ──────────────────────────────────────
#=
Minimizes the variational energy using the Stochastic Reconfiguration (SR)
update rule:
    Δv = η · (S + λI)⁻¹ · g

where g is the energy gradient and S is the quantum geometric tensor (metric).
Works for any Wavefunction subtype via flatten_params / unflatten_params dispatch.

Convergence is assessed using a patience counter: the optimizer must satisfy
one of the convergence criteria for `patience` consecutive iterations before
stopping. Criteria:
  1. All gradient components are statistically indistinguishable from zero.
  2. The predicted energy drop is smaller than the energy resolution AND the
     RMS parameter step is smaller than step_tol.

The reference learning rate η_ref is used only for convergence diagnostics
(predicted drop and RMS step), decoupling the stopping condition from the
choice of η.

kwargs
------
η              : learning rate
λ              : diagonal regularisation of S (Tikhonov)
max_step       : hard clip on ||Δv||; useful for Gutzwiller to avoid large
                 steps early in optimisation (set to Inf to disable)
z_grad         : SNR threshold for a gradient component to be "resolved"
z_energy       : multiplier on energy SEM for predicted-drop convergence test
step_tol       : RMS step threshold (reference η) for step-size convergence test
min_iters      : minimum iterations before convergence is checked
patience       : consecutive convergence hits required to stop
max_iters      : hard iteration cap
=#
function optimize_SR(sys::System,
                     wf::Wavefunction,
                     n_max::Int;
                     η               ::Float64 = 0.05,
                     λ               ::Float64 = 1e-3,
                     max_step        ::Float64 = Inf,
                     num_walkers     ::Int     = 200,
                     num_MC_steps    ::Int     = 30000,
                     num_equil_steps ::Int     = 5000,
                     block_size      ::Int     = 200,
                     z_grad          ::Float64 = 1.0,
                     z_energy        ::Float64 = 1.0,
                     step_tol        ::Float64 = 1e-4,
                     min_iters       ::Int     = 5,
                     patience        ::Int     = 2,
                     max_iters       ::Int     = 200)

    history          = NamedTuple[]
    prev_E           = nothing
    prev_err         = nothing
    convergence_hits = 0
    η_ref            = 0.05     # fixed reference η for convergence diagnostics only

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
        Δv        = η     .* direction
        Δv_ref    = η_ref .* direction

        # Optional hard clip on step size
        if isfinite(max_step) && norm(Δv) > max_step
            Δv .*= max_step / norm(Δv)
        end

        # ── Convergence diagnostics ───────────────────────────────────────────
        snr            = snr_vector(g, SE_g; zero_tol = 1e-16)
        num_resolved   = count(x -> x > z_grad, snr)
        predicted_drop = dot(g, Δv_ref)
        rms_step       = norm(Δv_ref) / sqrt(length(Δv_ref))

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

        if prev_E !== nothing
            ΔE     = abs(E - prev_E)
            ΔE_err = sqrt(err^2 + prev_err^2)
            println("  |ΔE|             = $(round(ΔE, digits=8))")
            println("  σ(ΔE)           = $(round(ΔE_err, digits=8))")
        end

        push!(history, (
            wavefunction   = flatten_params(wf),
            energy         = E,
            sem_energy     = err,
            gradient       = copy(g),
            snr            = copy(snr),
            predicted_drop = predicted_drop,
            rms_step       = rms_step
        ))

        # ── Convergence test ──────────────────────────────────────────────────
        gradient_zero     = num_resolved == 0
        energy_unresolved = predicted_drop <= z_energy * err
        step_small        = rms_step <= step_tol

        converged_now = iter >= min_iters &&
                        (gradient_zero || (energy_unresolved && step_small))

        if converged_now
            convergence_hits += 1
            println("  Convergence candidate: $convergence_hits / $patience")
        else
            convergence_hits = 0
        end

        if convergence_hits >= patience
            println("Converged at iteration $iter.")
            println("  gradient_zero     = ", gradient_zero)
            println("  energy_unresolved = ", energy_unresolved)
            println("  step_small        = ", step_small)
            wf = unflatten_params(flatten_params(wf) .- Δv, wf, n_max)
            break
        end

        # ── Parameter update ──────────────────────────────────────────────────
        wf       = unflatten_params(flatten_params(wf) .- Δv, wf, n_max)
        prev_E   = E
        prev_err = err
        η       *= 0.998
    end

    return wf, history
end