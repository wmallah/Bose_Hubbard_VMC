using Random, Statistics
using ProgressMeter
using FFTW



Random.seed!(0)

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

    PN = zeros(Int, L * n_max + 1)

    # Current block accumulators
    block_sum_E  = 0.0
    block_sum_T  = 0.0
    block_sum_V  = 0.0
    block_sum_O  = zeros(Float64, length(params.vr))
    block_sum_OO = zeros(Float64, length(params.vr), length(params.vr))
    block_sum_EO = zeros(Float64, length(params.vr))
    block_count  = 0

    # Per-block means (used for SEMs and blocked gradient error bars)
    block_means_E  = Float64[]
    block_means_T  = Float64[]
    block_means_V  = Float64[]
    block_means_O  = Vector{Vector{Float64}}()
    block_means_EO = Vector{Vector{Float64}}()

    # Global sums over COMPLETED BLOCKS ONLY
    Nv = length(params.vr)

    used_sum_E  = 0.0
    used_sum_T  = 0.0
    used_sum_V  = 0.0
    used_sum_O  = zeros(Float64, Nv)
    used_sum_OO = zeros(Float64, Nv, Nv)
    used_sum_EO = zeros(Float64, Nv)

    num_samples_used = 0
    num_samples_total = 0

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
                from = rand(1:L)
                while n[from] == 0
                    from = rand(1:L)
                end

                to = rand(sys.lattice.neighbors[from])
                while n[to] == n_max
                    to = rand(sys.lattice.neighbors[from])
                end

                Δlogpsi = compute_delta_logpsi_realspace(n, from, to, params)
                log_ratio = acceptance_probability_realspace_jastrow(n, from, to, params)

                if isfinite(log_ratio) && (log(rand()) < log_ratio)
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

            if step > num_equil_steps
                if !projective || (projective && N_now == N_target)

                    E, T, V = local_energy_jastrow(w.n, sys, n_max, params)

                    if isfinite(E)
                        O = logpsi_derivatives_realspace(w.n)

                        block_sum_E  += E
                        block_sum_T  += T
                        block_sum_V  += V
                        block_sum_O  .+= O
                        block_sum_OO .+= O * O'
                        block_sum_EO .+= E .* O
                        block_count  += 1

                        num_samples_total += 1

                        if block_count == block_size
                            # Block means
                            E_block  = block_sum_E / block_size
                            T_block  = block_sum_T / block_size
                            V_block  = block_sum_V / block_size
                            O_block  = block_sum_O ./ block_size
                            EO_block = block_sum_EO ./ block_size

                            push!(block_means_E, E_block)
                            push!(block_means_T, T_block)
                            push!(block_means_V, V_block)
                            push!(block_means_O, copy(O_block))
                            push!(block_means_EO, copy(EO_block))

                            # Add this completed block to the global used sums
                            used_sum_E  += block_sum_E
                            used_sum_T  += block_sum_T
                            used_sum_V  += block_sum_V
                            used_sum_O  .+= block_sum_O
                            used_sum_OO .+= block_sum_OO
                            used_sum_EO .+= block_sum_EO

                            num_samples_used += block_size

                            # Reset block accumulators
                            block_sum_E  = 0.0
                            block_sum_T  = 0.0
                            block_sum_V  = 0.0
                            block_sum_O .= 0.0
                            block_sum_OO .= 0.0
                            block_sum_EO .= 0.0
                            block_count  = 0
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

    n_blocks = length(block_means_E)

    if n_blocks == 0 || num_samples_used == 0
        @warn "No valid completed blocks collected!"

        return VMCResults(
            Inf, Inf,
            Inf, Inf,
            Inf, Inf,
            Float64[], Float64[], zeros(Float64, 0, 0),
            0,
            acceptance_ratio,
            Float64[],
            num_failed_moves,
            Int[]
        )
    end

    if block_count > 0
        @warn "Discarding incomplete final block of $block_count samples for consistent blocked statistics."
    end

    if n_blocks < 5
        @warn "Only $n_blocks completed blocks collected; gradient error bars may be unreliable."
    end

    # Point estimates from the SAME completed-block sample set
    E_mean  = used_sum_E  / num_samples_used
    T_mean  = used_sum_T  / num_samples_used
    V_mean  = used_sum_V  / num_samples_used
    O_mean  = used_sum_O  / num_samples_used
    OO_mean = used_sum_OO / num_samples_used
    EO_mean = used_sum_EO / num_samples_used

    # Energy SEMs from block means
    if n_blocks > 1
        E_error = std(block_means_E) / sqrt(n_blocks)
        T_error = std(block_means_T) / sqrt(n_blocks)
        V_error = std(block_means_V) / sqrt(n_blocks)
    else
        E_error = Inf
        T_error = Inf
        V_error = Inf
    end

    # SR gradient / metric
    g = 2.0 .* (EO_mean .- E_mean .* O_mean)
    S = OO_mean .- O_mean * O_mean'
    S = 0.5 .* (S + S')

    # Blocked standard error for the SAME estimator family:
    # X_i = 2 * (E_i O_i - E_mean O_i - O_mean E_i + E_mean O_mean)
    # So the corresponding block mean is:
    # X_b = 2 * (EO_b - E_mean O_b - O_mean E_b + E_mean O_mean)
    if n_blocks > 1
        g_blocks = Matrix{Float64}(undef, n_blocks, Nv)

        for b in 1:n_blocks
            E_b  = block_means_E[b]
            O_b  = block_means_O[b]
            EO_b = block_means_EO[b]

            g_blocks[b, :] .= 2.0 .* (EO_b .- E_mean .* O_b .- E_b .* O_mean .+ E_mean .* O_mean)
        end

        SE_g = vec(std(g_blocks, dims=1)) ./ sqrt(n_blocks)
    else
        SE_g = fill(Inf, Nv)
    end

    return VMCResults(
        E_mean, E_error,
        T_mean, T_error,
        V_mean, V_error,
        g, SE_g, S,
        num_samples_used,
        acceptance_ratio,
        block_means_E,
        num_failed_moves,
        PN
    )
end