using LinearAlgebra

import ..VMCBoseHubbard: MC_integration_Gutzwiller
import ..VMCBoseHubbard: MC_integration_Jastrow
import ..VMCBoseHubbard: estimate_energy_gradient_and_metric

export optimize_kappa_SR, optimize_jastrow_SR

function optimize_kappa_SR(sys::System,
                           N_target::Int,
                           n_max::Int,
                           grand_canonical::Bool,
                           projective::Bool;
                           κ_init::Float64 = 1.0,
                           η::Float64 = 0.05,
                           num_walkers::Int = 200,
                           num_MC_steps::Int = 30000,
                           num_equil_steps::Int = 5000,
                           block_size::Int = 200,
                           z::Float64 = 1.0)

    κ = κ_init

    history = Vector{NamedTuple{(:κ,:energy,:gradient,:snr),
               Tuple{Float64,Float64,Float64,Float64}}}()

    λ = 1e-3
    max_step = 0.2

    while true

        ############################################################
        # Monte Carlo sampling
        ############################################################

        result = MC_integration_Gutzwiller(
            sys,
            N_target,
            κ,
            n_max,
            grand_canonical,
            projective;
            num_walkers=num_walkers,
            num_MC_steps=num_MC_steps,
            num_equil_steps=num_equil_steps,
            block_size=block_size
        )

        E   = result.mean_energy
        err = result.sem_energy

        ############################################################
        # Gradient and metric from MC
        ############################################################

        g_vec  = result.gradient
        SE_vec = result.gradient_standard_error
        S_mat  = result.metric

        g   = g_vec[1]
        SEg = SE_vec[1]
        S   = S_mat[1,1]

        if !isfinite(g) || !isfinite(SEg) || !isfinite(S)
            @warn "Stopping: non-finite values encountered"
            break
        end

        ############################################################
        # Signal-to-noise ratio
        ############################################################

        snr = abs(g) / SEg

        println("κ = $(round(κ,digits=10))  " *
                "E = $(round(E,digits=8)) ± $(round(err,digits=8))  " *
                "g = $(round(g,digits=6))  " *
                "SNR = $(round(snr,digits=4))")

        push!(history,
              (κ = κ,
               energy = E,
               gradient = g,
               snr = snr))

        ############################################################
        # Statistical convergence test
        ############################################################

        if abs(g) < z * SEg
            println("Gradient statistically zero (|g| < $(z)σ). Converged.")
            break
        end

        ############################################################
        # Natural gradient step (SR)
        ############################################################

        Δκ = η * g / (S + λ)

        if abs(Δκ) > max_step
            Δκ = max_step * sign(Δκ)
        end

        ############################################################
        # Parameter update
        ############################################################

        κ -= Δκ
        κ = clamp(κ, 1e-12, 10.0)

        ############################################################
        # Mild learning rate decay
        ############################################################

        η *= 0.998

    end

    return κ, history
end


function flatten_params(params::JastrowParams)
    return copy(params.vr)
end

function unflatten_params(v::Vector{T}, L::Int) where {T<:Real}
    @assert length(v) == fld(L, 2) + 1
    return JastrowParams(copy(v))
end


function optimize_jastrow_SR(sys::System,
                             params::JastrowParams,
                             N_target::Int,
                             n_max::Int;
                             η::Float64 = 0.05,
                             num_walkers::Int = 200,
                             num_MC_steps::Int = 30000,
                             num_equil_steps::Int = 5000,
                             block_size::Int = 200,
                             z::Float64 = 1.0)

    history = []

    λ = 1e-3
    max_step = 0.2

    L = length(sys.lattice.neighbors)

    while true

        ############################################################
        # Monte Carlo sampling
        ############################################################

        result = MC_integration_Jastrow(
            sys,
            N_target,
            params,
            n_max,
            false,
            false;
            num_walkers = num_walkers,
            num_MC_steps = num_MC_steps,
            num_equil_steps = num_equil_steps,
            block_size = block_size
        )

        E = result.mean_energy
        err = result.sem_energy

        ############################################################
        # Estimate gradient and SR metric
        ############################################################

        g = result.gradient
        SE_g = result.gradient_standard_error
        S = result.metric

        if any(!isfinite, g) || any(!isfinite, SE_g) || any(!isfinite, S)
            @warn "Stopping: non-finite values encountered"
            break
        end

        ############################################################
        # Signal-to-noise ratios
        ############################################################

        snr = similar(g)
        for i in eachindex(g)
            snr[i] = SE_g[i] > 0 ? abs(g[i]) / SE_g[i] : Inf
        end

        println("Energy = $(round(E, digits=8)) ± $(round(err, digits=8))")
        println("Gradient norm = ", norm(g))
        println("Max SNR = ", maximum(snr))

        push!(history,
              (params = flatten_params(params),
               energy = E,
               gradient = copy(g),
               snr = copy(snr)))

        ############################################################
        # Statistical convergence test
        ############################################################

        if all(abs.(g) .< z .* SE_g)
            println("All gradient components statistically zero. Converged.")
            break
        end

        ############################################################
        # Natural gradient step (SR)
        ############################################################

        Δv = η * ((S + λ * I) \ g)

        step_norm = norm(Δv)
        if step_norm > max_step
            Δv *= max_step / step_norm
        end

        ############################################################
        # Parameter update
        ############################################################

        v_old = flatten_params(params)
        v_new = v_old .- Δv
        params = unflatten_params(v_new, L)

        η *= 0.998

    end

    return params, history
end