using Random, Statistics
using ProgressMeter
using FFTW



Random.seed!(1)

#=
Purpose: store information about the VMC results
Input: mean total energy, STD total mean energy, mean kinetic energy, STD of mean kinetic energy, mean potential energy, STD mean potential energy, acceptance ratio, vector of energies, number of failed moves, total particle number distribution
Author: Will Mallah
Last Updated: 01/25/26
=#
struct VMCResults
    mean_energy::Float64
    sem_energy::Float64
    mean_kinetic::Float64
    sem_kinetic::Float64
    mean_potential::Float64
    sem_potential::Float64
    gradient::Vector{Float64}
    gradient_standard_error::Vector{Float64}
    metric::Matrix{Float64}
    num_samples::Int64
    acceptance_ratio::Float64
    energies::Vector{Float64}
    num_failed_moves::Int
    PN::Vector{Int}
end


#=
Prupose: ensure walkers (system configurations) have physical entries
Input: n (walker/system configurationa as a vector), n_max (maximum site occupancy)
Output: true if walker is allowed, false if walker has error
Author: Will Mallah
Last Updated: 01/25/26
=#
function check_and_warn_walker(n::Vector{Int}, n_max::Int)
    if any(isnan, n)
        @warn "Walker contains NaN: $n"
        return false
    elseif any(isinf, n)
        @warn "Walker contains Inf: $n"
        return false
    elseif any(x -> x < 0 || x > n_max, n)
        @warn "Walker has out-of-bounds occupation: $n"
        return false
    end
    return true
end


#=
Purpose: perform Monte Carlo to integrate the local energy
Input: sys (system struct), N_target (target value for total number of particles), κ (variational parameter), n_max (maximum site occupancy), grand_canonical (true for grand_canonical move proposal, false for canonical move_proposal), projective (true for projective measurements, false for non-projective measurements)
Optional Input: num_walkers (number of walkers/configurations), num_MC_steps (number of Monte Carlo steps), num_equil_steps (number of equilibration steps)
Output: struct of variational Monte Carlo results (see struct defined above)
Author: Will Mallah
Last Updated: 01/25/26
=#
function MC_integration_Gutzwiller(sys::System,
                                   N_target::Int,
                                   κ::Real,
                                   n_max::Int,
                                   grand_canonical,
                                   projective;
                                   num_walkers::Int = 200,
                                   num_MC_steps::Int = 30000,
                                   num_equil_steps::Int = 5000,
                                   block_size::Int = 200)

    ensemble = !grand_canonical ? "Canonical" : "Grand Canonical"

    L = length(sys.lattice.neighbors)
    μ = sys.μ

    ############################################################
    # Initialize walkers
    ############################################################

    function ground_state_like_configuration()
        if N_target > L * n_max
            error("Impossible: N > L * n_max")
        end

        n = fill(div(N_target, L), L)
        remainder = N_target % L

        for i in 1:remainder
            n[i] += 1
        end

        return n
    end

    walkers = [ground_state_like_configuration() for _ in 1:num_walkers]

    wf = generate_coefficients(κ, n_max)

    ############################################################
    # Measurement storage
    ############################################################

    PN = zeros(Int, L * n_max)

    block_sum_E = 0.0
    block_sum_T = 0.0
    block_sum_V = 0.0
    block_count = 0

    block_means_E = Float64[]
    block_means_T = Float64[]
    block_means_V = Float64[]

    derivative_logpsi_blocks = Float64[]

    sum_O  = 0.0
    sum_OO = 0.0
    sum_EO = 0.0

    block_sum_g = 0.0
    block_gradients = Float64[]

    num_samples = 0

    num_accepted_moves = 0
    num_failed_moves = 0

    ############################################################
    # Monte Carlo loop
    ############################################################

    @showprogress enabled=true "Running " * ensemble * " VMC..." for step in 1:num_MC_steps

        for i in 1:num_walkers

            n = walkers[i]

            from = rand(1:L)
            while n[from] == 0
                from = rand(1:L)
            end

            to = rand(sys.lattice.neighbors[from])
            while n[to] == n_max
                to = rand(sys.lattice.neighbors[from])
            end

            if from != to

                # compute ratio using current configuration
                ratio = acceptance_probability_Gutzwiller(n, from, to, wf)

                if isfinite(ratio) && rand() < ratio

                    # apply the move
                    n[from] -= 1
                    n[to]   += 1

                    num_accepted_moves += 1

                else
                    num_failed_moves += 1
                end

            else
                num_failed_moves += 1
            end

            ####################################################
            # Particle number histogram
            ####################################################

            N_now = sum(n)

            if N_now + 1 <= length(PN)
                PN[N_now + 1] += 1
            end

            ####################################################
            # Measurements
            ####################################################

            if step >= num_equil_steps

                if !projective || (projective && N_now == N_target)

                    E, T, V = local_energy(n, wf, sys, n_max, grand_canonical, projective)

                    if isfinite(E)

                        block_sum_E += E
                        block_sum_T += T
                        block_sum_V += V
                        block_count += 1

                        O = -0.5 * sum(n .^ 2)

                        g_sample = 2 * E * O
                        block_sum_g += g_sample

                        sum_O  += O
                        sum_OO += O * O
                        sum_EO += E * O

                        if block_count == block_size

                            push!(block_means_E, block_sum_E / block_size)
                            push!(block_means_T, block_sum_T / block_size)
                            push!(block_means_V, block_sum_V / block_size)

                            push!(block_gradients, block_sum_g / block_size)

                            block_sum_E = 0.0
                            block_sum_T = 0.0
                            block_sum_V = 0.0
                            block_sum_g = 0.0

                            block_count = 0
                        end

                        num_samples += 1
                    end
                end
            end
        end
    end

    ############################################################
    # Final statistics
    ############################################################

    acceptance_ratio = num_accepted_moves / (num_accepted_moves + num_failed_moves)

    if isempty(block_means_E)
        @warn "No valid energy samples collected!"

        return VMCResults(Inf, Inf, Inf, Inf, Inf, Inf,
                          Float64[], Float64[], zeros(1,1),
                          0, acceptance_ratio, Float64[],
                          num_failed_moves, PN)
    end

    n_blocks = length(block_means_E)

    E_mean = mean(block_means_E)
    E_error = std(block_means_E) / sqrt(n_blocks)

    O_mean  = sum_O  / num_samples
    OO_mean = sum_OO / num_samples
    EO_mean = sum_EO / num_samples

    g = [2 * (EO_mean - E_mean * O_mean)]

    S = [OO_mean - O_mean^2;;]

    g_blocks = reshape(block_gradients, :, 1)
    g_blocks .-= mean(g_blocks, dims=1)

    SE_g = vec(std(g_blocks, dims=1)) ./ sqrt(size(g_blocks,1))

    return VMCResults(
        E_mean, E_error,
        mean(block_means_T), std(block_means_T) / sqrt(length(block_means_T)),
        mean(block_means_V), std(block_means_V) / sqrt(length(block_means_V)),
        g, SE_g, S,
        num_samples,
        acceptance_ratio,
        block_means_E,
        num_failed_moves,
        PN
    )
end

#=
Purpose: store all information for our walkers
Input: n (walker/system configuration in real space),
       logpsi (cached log of Jastrow part of wavefunction),
       N (total particle number)
Author: Will Mallah
Last Updated: 04/03/2026
=#
mutable struct Walker
    n::Vector{Int}
    logpsi::Float64
    N::Int
end


#=
Purpose: initialize a walker in real space
Input: n (walker/system configuration in real space),
       params (real-space Jastrow coefficients)
Output: Walker struct
Last Updated: 04/03/2026
=#
function initialize_walker(n::Vector{Int}, params::JastrowParams)
    logpsi = compute_logpsi_realspace(n, params)
    return Walker(copy(n), logpsi, sum(n))
end


function MC_integration_Jastrow(sys::System,
                                N_target::Int,
                                params::JastrowParams,
                                n_max::Int,
                                grand_canonical,
                                projective;
                                num_walkers::Int = 200,
                                num_MC_steps::Int = 30000,
                                num_equil_steps::Int = 5000,
                                block_size::Int = 200)

    ensemble = !grand_canonical ? "Canonical" : "Grand Canonical"

    L = length(sys.lattice.neighbors)
    μ = sys.μ  # kept in case you use it later for grand canonical or projected logic

    ############################################################
    # Ground-state-like walker generator
    ############################################################

    function ground_state_like_configuration()
        if N_target > L * n_max
            error("Impossible: N > L * n_max")
        end

        n = fill(div(N_target, L), L)
        remainder = N_target % L

        for i in 1:remainder
            n[i] += 1
        end

        return n
    end

    ############################################################
    # Initialize walkers
    ############################################################

    walkers = [initialize_walker(ground_state_like_configuration(), params)
               for _ in 1:num_walkers]

    ############################################################
    # Measurement storage
    ############################################################

    # Histogram total particle number
    PN = zeros(Int, L * n_max + 1)

    # Block accumulators
    block_sum_E = 0.0
    block_sum_T = 0.0
    block_sum_V = 0.0
    block_count = 0

    block_means_E = Float64[]
    block_means_T = Float64[]
    block_means_V = Float64[]
    total_N = Float64[]

    # Gradient / SR storage
    Nv = length(params.vr)

    sum_O = zeros(Float64, Nv)
    sum_OO = zeros(Float64, Nv, Nv)
    sum_EO = zeros(Float64, Nv)

    block_sum_g = zeros(Float64, Nv)
    block_gradients = Vector{Vector{Float64}}()

    num_samples = 0
    num_completed_blocks = 0

    num_accepted_moves = 0
    num_failed_moves = 0

    ############################################################
    # Monte Carlo loop
    ############################################################

    @showprogress enabled=true "Running " * ensemble * " VMC..." for step in 1:num_MC_steps
        for w in walkers
            n = w.n

            ####################################################
            # Move proposal
            ####################################################

            if grand_canonical
                error("Grand canonical moves not implemented yet for the real-space Jastrow branch.")
            else
                # choose source site with at least one boson
                from = rand(1:L)
                while n[from] == 0
                    from = rand(1:L)
                end

                # choose neighboring target not already at n_max
                to = rand(sys.lattice.neighbors[from])
                while n[to] == n_max
                    to = rand(sys.lattice.neighbors[from])
                end

                # compute acceptance from the ORIGINAL configuration
                Δlogpsi = compute_delta_logpsi_realspace(n, from, to, params)
                log_ratio = 2.0 * Δlogpsi + log(n[from]) - log(n[to] + 1)

                if isfinite(log_ratio) && (log_ratio >= 0.0 || log(rand()) < log_ratio)
                    n[from] -= 1
                    n[to]   += 1

                    w.logpsi += Δlogpsi
                    w.N = N_target

                    num_accepted_moves += 1
                else
                    num_failed_moves += 1
                end

                N_now = N_target
            end

            ####################################################
            # Measurements
            ####################################################

            if N_now + 1 <= length(PN)
                PN[N_now + 1] += 1
            end

            if step >= num_equil_steps
                if !projective || (projective && N_now == N_target)

                    E, T, V = local_energy_jastrow(w.n, sys, params)

                    if isfinite(E)
                        block_sum_E += E
                        block_sum_T += T
                        block_sum_V += V
                        block_count += 1

                        O = logpsi_derivatives_realspace(w.n)

                        # Store sample accumulators for gradient / SR
                        sum_O  .+= O
                        sum_OO .+= O * O'
                        sum_EO .+= E .* O
                        push!(total_N, N_now)

                        # block gradient estimate
                        g_sample = 2.0 .* (E .* O)
                        block_sum_g .+= g_sample

                        num_samples += 1

                        if block_count == block_size
                            push!(block_means_E, block_sum_E / block_size)
                            push!(block_means_T, block_sum_T / block_size)
                            push!(block_means_V, block_sum_V / block_size)
                            push!(block_gradients, block_sum_g ./ block_size)

                            block_sum_E = 0.0
                            block_sum_T = 0.0
                            block_sum_V = 0.0
                            block_sum_g .= 0.0
                            block_count = 0

                            num_completed_blocks += 1
                        end
                    end
                end
            end
        end
    end

    ############################################################
    # Final statistics
    ############################################################

    total_attempts = num_accepted_moves + num_failed_moves
    acceptance_ratio = total_attempts > 0 ? num_accepted_moves / total_attempts : 0.0

    if isempty(block_means_E)
        @warn "No valid energy samples collected!"

        return VMCResults(
            Inf, Inf,
            Inf, Inf,
            Inf, Inf,
            Float64[], Float64[], zeros(Float64, 0, 0),
            num_samples,
            acceptance_ratio,
            Float64[],
            num_failed_moves,
            Int[]
        )
    end

    E_mean = mean(block_means_E)
    E_error = std(block_means_E) / sqrt(length(block_means_E))

    T_mean = mean(block_means_T)
    T_error = std(block_means_T) / sqrt(length(block_means_T))

    V_mean = mean(block_means_V)
    V_error = std(block_means_V) / sqrt(length(block_means_V))

    # Use num_samples here, not number of completed blocks
    O_mean  = sum_O  ./ num_samples
    OO_mean = sum_OO ./ num_samples
    EO_mean = sum_EO ./ num_samples

    g = 2.0 .* (EO_mean .- E_mean .* O_mean)
    S = OO_mean .- O_mean * O_mean'

    if isempty(block_gradients)
        SE_g = fill(Inf, Nv)
    else
        g_blocks = hcat(block_gradients...)'
        g_blocks .-= mean(g_blocks, dims=1)
        SE_g = vec(std(g_blocks, dims=1)) ./ sqrt(size(g_blocks, 1))
    end

    return VMCResults(
        E_mean, E_error,
        T_mean, T_error,
        V_mean, V_error,
        g, SE_g, S,
        num_samples,
        acceptance_ratio,
        block_means_E,
        num_failed_moves,
        PN
    )
end