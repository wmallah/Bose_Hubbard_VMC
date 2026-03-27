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
                ratio = acceptance_probability(n, from, to, wf)

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
Input: n (walker/system configuration in real space), nq (walker/system configuration in momentum space), logpsi (log of our wavefunction)
Author: Will Mallah
Last Updated: 03/04/2026
=#
mutable struct Walker
    n::Vector{Int}               # real-space occupations
    nq::Vector{ComplexF64}       # Fourier density modes
    logpsi::Float64              # cached log wavefunction
    N::Int64                     # total particle number
end



#=
Purpose: initialize the walkers, which now contain n (walker/system configuration in real space), nq (walker/system configuration in momentum space), logpsi (log of our wavefunction)
Input: n (walker/system configuration in real sapce), params (set of Jastrow coefficients), L (system size)
Output: Walker struct
Last Updated: 03/05/2026
=#
function initialize_walker(n::Vector{Int}, params::JastrowParams, L::Int, N::Int)
    @assert length(n) == L
    @assert sum(n) == N

    nq = fft(Float64.(n))
    logpsi = compute_logpsi(nq, params, L)

    return Walker(copy(n), nq, logpsi, N)
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

    # Select wether system is canonical or grand canonical according to the grand_canonical boolean variable (false for canonical, true for grand canonical)
    ensemble = !grand_canonical ? "Canonical" : "Grand Canonical"

    # Determine the lattice size from the length of the lattice neighbors generated in the sys struct
    L = length(sys.lattice.neighbors)
    # Retrieve the value for the chemcial potential from the sys struct
    μ = sys.μ

    ############################################################
    # Ground-state-like walker generator
    ############################################################

    # Intialize the system configurations/walkers as true unit filling plus remainder
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

    walkers = [initialize_walker(ground_state_like_configuration(), params, L, N_target)
               for _ in 1:num_walkers]

    ############################################################
    # Measurement storage
    ############################################################

    # For histograming total particle number in grand canonical
    PN = zeros(Int, L * n_max)

    # Blocking sums and vector initialization
    num_completed_blocks = 0
    num_samples = 0

    block_sum_E = 0.0
    block_sum_T = 0.0
    block_sum_V = 0.0
    block_count = 0

    block_means_E = Float64[]
    block_means_T = Float64[]
    block_means_V = Float64[]
    total_N = Float64[]

    npair = num_pair_modes(L)
    Nv = npair + (has_edge_mode(L) ? 1 : 0)

    sum_O = zeros(Float64, Nv)
    sum_OO = zeros(Float64, Nv, Nv)
    sum_EO = zeros(Float64, Nv)

    block_sum_g = zeros(Float64, Nv)
    block_gradients = Vector{Vector{Float64}}()

    num_samples = 0
    
    num_accepted_moves = 0
    num_failed_moves = 0

    ############################################################
    # Precompute Fourier phase factors
    ############################################################

    # [HAVE OUTDATED FUNCTION FOR THIS IN "utils.jl"]

    # Generate possible momentum values
    qvals = [2π*(k-1)/L for k in 1:L]

    # Initialize phase matrix
    phase = Matrix{ComplexF64}(undef, L, L)

    # Fill phase matrix
    for k in 1:L
        for r in 1:L
            phase[k,r] = exp(-im * qvals[k] * (r-1))
        end
    end

    ############################################################
    # Monte Carlo loop
    ############################################################

    @showprogress enabled=true "Running " * ensemble * " VMC..." for step in 1:num_MC_steps
        # Loop through system configurations/walkers to update
        for w in walkers
            # Temporary variable to hold current walker
            n = w.n
            # Temporary variable to hold Fourier configuration modes
            nq = w.nq

            ####################################################
            # Move proposal
            ####################################################

            if grand_canonical
                error("Grand canonical moves not implemented yet")
                # Select random site in the lattice
                site = rand(1:L)

                # Add or remove a particle on random site with 50/50 probability (add/remove, respectively)
                if rand() < 0.5
                    n[site] += 1
                else
                    n[site] -= 1
                end

                # Expensive to sum walker/configuration every time. Change w.N in walker struct instead after move proposal [FIX FOR GRAND CANONICAL]
                N_now = sum(n)
            else
                # Select random site to be source of move
                from = rand(1:L)
                while n[from] == 0
                    from = rand(1:L)
                end

                # Select random neighbor (target site) of that previously chosen random site
                to = rand(sys.lattice.neighbors[from])
                while n[to] == n_max
                    to = rand(sys.lattice.neighbors[from])
                end

                # If the source and target sites aren't the same site (they shouldn't be from how the neighbor matrix is generated), attempt to move a single particle
                if from != to
                    n[from] -= 1
                    n[to] += 1
                end

                # Total particle number should not change in canonical ensemble
                N_now = N_target
            end

            ####################################################
            # Compute Δlogψ
            ####################################################

            Δlogψ = compute_delta_logpsi(nq, from, to, phase, params, L)

            log_ratio = 2 * Δlogψ

            ####################################################
            # Metropolis step
            ####################################################

            if isfinite(log_ratio) && log(rand()) < log_ratio
                # Update Fourier components and logpsi if move accepted
                @inbounds for k in eachindex(nq)
                    nq[k] += phase[k,to] - phase[k,from]
                end
                w.logpsi += Δlogψ

                num_accepted_moves += 1
            else
                # Return to originial walker/configuration if move not accepted
                n[from] += 1
                n[to]   -= 1
                num_failed_moves += 1
            end

            ####################################################
            # Measurements
            ####################################################

            # Histogram total number of particles
            if N_now + 1 <= length(PN)
                PN[N_now + 1] += 1
            end

            # Only make measurements after equilibration
            if step >= num_equil_steps

                if !projective || (projective && N_now == N_target)

                    E, T, V = local_energy_jastrow(L, sys, w, params, phase)

                    if isfinite(E)
                        block_sum_E += E
                        block_sum_T += T
                        block_sum_V += V
                        block_count += 1

                        O = logpsi_derivatives(w.nq, L)

                        sum_O  .+= O
                        sum_OO .+= O * O'
                        sum_EO .+= E .* O
                        push!(total_N, N_now)

                        num_samples += 1

                        # accumulate raw contribution for this block
                        block_sum_g .+= 2 .* (E .* O)

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

    # Compute acceptance ratio
    n_attempts = num_accepted_moves + num_failed_moves
    acceptance_ratio = n_attempts > 0 ? num_accepted_moves / n_attempts : 0.0

    # If no valid blocked samples were collected, warn and return
    if num_completed_blocks == 0 || num_samples == 0

        @warn "No valid blocked samples collected!"

        return VMCResults(
            Inf, Inf,
            Inf, Inf,
            Inf, Inf,
            Float64[], Float64[], zeros(Float64, 0, 0),
            num_samples,
            acceptance_ratio,
            Float64[],
            num_failed_moves,
            PN
        )

    end


    # Block means
    E_mean = mean(block_means_E)
    E_error = std(block_means_E) / sqrt(num_completed_blocks)

    O_mean  = sum_O  ./ num_samples
    OO_mean = sum_OO ./ num_samples
    EO_mean = sum_EO ./ num_samples

    # Gradient and metric from block means
    g = 2 .* (EO_mean .- E_mean .* O_mean)
    S = OO_mean .- O_mean * O_mean'

    # Standard error of the gradient
    g_blocks = hcat(block_gradients...)'
    g_blocks .-= mean(g_blocks, dims=1)
    SE_g = vec(std(g_blocks, dims=1)) ./ sqrt(num_completed_blocks)

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