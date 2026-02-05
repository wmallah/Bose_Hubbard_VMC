using Random, Statistics
using ProgressMeter

Random.seed!(1234)

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
    acceptance_ratio::Float64
    energies::Vector{Float64}
    derivative_log_psi::Vector{Float64}
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
function MC_integration(sys::System, N_target::Int, κ::Real, n_max::Int, grand_canonical, projective;
                              num_walkers::Int = 200,
                              num_MC_steps::Int = 30000,
                              num_equil_steps::Int = 5000)

    ensemble = !grand_canonical ? "Canonical" : "Grand Canonical"

    # Extract the system size from the number of rows in the adjacency matrix
    L = length(sys.lattice.neighbors)

    # Function that takes in the system size and target number of particles and returns a random array for the system configuration
    function random_walker(L::Int, N::Int)
        # idx = randperm(L)[1:N]
        w = zeros(Int, L)
        for i in eachindex(w)
            w[i] = 1
        end
        return w
    end

    # Generate an array of random walkers (system configurations) using list comprehension
    walkers = [random_walker(L, N_target) for _ in 1:num_walkers]    

    # Generate the coefficients for the Gutzwiller wavefunction
    wf = generate_coefficients(κ, n_max)
    # println(exp.(wf.f))
    # println(sum(exp.(2 .* wf.f)) ≈ 1.0)

    # Track total number of particles and number of accepted/failed moves
    PN = zeros(Int, 2000)
    num_accepted_moves, num_failed_moves = 0, 0

    # Create empty arrays for all the measurements we want to track
    energies, derivative_log_psi, kinetic, potential, total_N = Float64[], Float64[], Float64[], Float64[], Float64[]

    # Begin Monte Carlo Loop outer loop (number of steps in our simulation)
    @showprogress enabled=true "Running " * ensemble * " VMC..." for step in 1:num_MC_steps
        # Begin Monte Carlo inner loop (number of walkers/configurations)
        for i in 1:num_walkers
            # Initialize the old and new sets of configurations
            n_old = walkers[i]
            n_new = copy(n_old)

            # Randomly select a site of the current configuration
            site = rand(1:L)

            # Either grand canonical (adding/removing particles) or canonical (hopping particles) move proposal
            if grand_canonical
                # Add or remove a particle on this site with 50/50 probability
                if rand() < 0.5
                    n_new[site] += 1
                else
                    n_new[site] -= 1
                end
            else
                # Hop particle from random site to random neigbor site
                from = site
                to = rand(sys.lattice.neighbors[from])

                n_new[from] -= 1
                n_new[to] += 1

                # If move is unphysical, count failed move and continue to walker loop
                if n_new[to] > n_max
                    num_failed_moves += 1
                    continue
                end
            end

            # Reject proposed move if unphysical and continue to walker loop
            if n_new[site] > n_max || n_new[site] < 0
                num_failed_moves += 1
                continue
            else
                # Single-site Gutzwiller log acceptance ratio:
                ratio = acceptance_probability(n_old, n_new, wf)
                # ratio = acceptance_probability(n_old, n_new, κ)

                # Accept move based on Metropolis-Hastings
                if isfinite(ratio) && rand() < ratio
                    walkers[i] = n_new
                    num_accepted_moves += 1
                else
                    num_failed_moves += 1
                    continue
                end
            end
            
            # Histogram the total number of particles from the configuration
            N_now = sum(walkers[i])
            if N_now + 1 <= length(PN)
                PN[N_now + 1] += 1
            end

            # Check to make sure the walkers have physically correct entries
            if check_and_warn_walker(walkers[i], n_max)
                # Only make measurements after equilibration and with the target number of particles in the system
                if step >= num_equil_steps
                    # If we are not projecting (non-projective grand canonical or canonical), measure. If we are projecting (projective grand canonical), only measure if the number of particles is our target number of particles
                    if !projective || (projective && N_now == N_target)
                        # Measure the total local energy as well as the kinetic and potential energies separately
                        E, T, V = local_energy(n_new, wf, sys, n_max)

                        # If the energy energy is finite, push to the respective vectors
                        if isfinite(E)
                            push!(energies, E)
                            push!(kinetic, T)
                            push!(potential, V)
                            push!(total_N, N_now)

                            # Calculate and track this value (derivative of log psi) for the gradient in our Gradient Descent optimization method
                            val = -0.5 * sum(walkers[i] .^ 2)
                            if !isfinite(val)
                                @warn "Non-finite derivative_log_psi value: $val"
                            else
                                push!(derivative_log_psi, val)
                            end
                        else
                            @warn "Non-finite local energy detected: E = $E"
                            continue
                        end
                    end
                end
            else
                @warn "Invalid walker skipped"
            end
        end
    end

    # Calculate the acceptance ratio to check if the simulation is accepting or rejecting most proposed moves
    acceptance_ratio = num_accepted_moves / (num_accepted_moves + num_failed_moves)

    # Warn if no valid energy samples were collected
    if isempty(energies)
        @warn "No valid energy samples collected!"
        return VMCResults(Inf, Inf, Inf, Inf, Inf, Inf, 0.0, Float64[], Float64[], num_failed_moves, Int[])
    end

    return VMCResults(
        mean(energies), std(energies) / sqrt(length(energies)),
        mean(kinetic), std(kinetic) / sqrt(length(kinetic)),
        mean(potential), std(potential) / sqrt(length(potential)),
        acceptance_ratio, energies, derivative_log_psi, num_failed_moves, PN
    )
end