using Random

export estimate_tau, blocking_error, compute_delta_logpsi

#=
Purpose: estimate value for n_max given kappa parameter and cutoff value for decay of Gutzwiller coefficients
Input: kappa (variational parameter), cutoff (value for coefficient deemed insignificant)
Output: particle number where Gutzwiller coefficients are significant
Author: Will Mallah
Last Updated: 07/04/25
=#
function estimate_n_max(κ::Real; cutoff::Real = 1e-6)
    n = 0
    while true
        f_n = (1 / sqrt(float(factorial(big(n))))) * exp(-κ * n^2 / 2)
        if f_n < cutoff
            return n - 1
        end
        n += 1
        if n > 300
            return 300
        end
    end
end


#=
Purpose: run the VMC function for either a 1D or 2D lattice
Input: sys (system struct), κ (variational parameter), n_max (maximum number of particles on a given site), N_total (total number of particles)
Optional Input: num_walkers, num_MC_steps, num_equil_steps (kwargs...)
Output: result from VMC_fixed_particles
Author: Will Mallah
Last Updated: 07/04/25
=#
function run_vmc(sys::System, κ::Real, n_max::Int, N_target::Int; grand_canonical=true, kwargs...)
    lattice = sys.lattice
    if !grand_canonical
        if lattice isa Lattice1D
            return VMC_fixed_particles(sys, κ, n_max, N_target; kwargs...)
        elseif lattice isa Lattice2D
            return VMC_fixed_particles(sys, κ, n_max, N_target; kwargs...)
        else
            error("Unsupported lattice type: $(typeof(lattice))")
        end
    else
        if lattice isa Lattice1D
            return VMC_grand_canonical_adaptive_mu(sys, κ, n_max, N_target, 1.0; kwargs...)
        elseif lattice isa Lattice2D
            return VMC_grand_canonical_adaptive_mu(sys, κ, n_max, N_target, 1.0; kwargs...)
        else
            error("Unsupported lattice type: $(typeof(lattice))")
        end
    end
end

#=
Purpose: calculate the blocking error, which takes autocorrelation time into account
Input: data in the form of a vector
Optional Input: block_size (number of consecutive MC samples per block)
Output: standard error for given data
Author: Will Mallah
Last Updated: 02/22/26
Notes: We partition the time series into contiguous chunks, because correlations exist between nearby samples. This function automatically incorporates autocorrelation as long as block_size > autocorrelation_time
=#
function blocking_error(data::Vector{Float64}; block_size::Int=100)

    # Divide total data into blocks evenly
    N = length(data)
    n_blocks = div(N, block_size)

    #=
    You need multiple blocks to estimate the variance of block means.
    Fewer than ~5 blocks → variance estimate becomes unstable.
    This prevents nonsense errors.
    =#
    if n_blocks < 5
        error("Not enough blocks for reliable blocking estimate.")
    end

    #=
    Truncate to full blocks
    If N is not an exact multiple of block_size:
        We discard the remainder
    This avoids partial blocks that bias variance
    =#
    truncated = data[1:(n_blocks * block_size)]

    #=
    Reshape into blocks
    Each column is one block
    Each column contains block_size consecutive MC samples
    =#
    blocks = reshape(truncated, block_size, n_blocks)

    #=
    Compute block means
    These are averages over correlated chunks
        Key idea:
        If block_size > autocorrelation time, then these averages are approximately independent random variables
        That’s the entire trick of blocking
    =#
    block_means = vec(mean(blocks, dims=1))

    #=
    If block_size ≫ τ:
    Correlations inside blocks remain.
    Correlations between blocks vanish.
    So block means behave like independent samples.
    Then classical statistics applies.
    =#
    # Variance of block means
    var_blocks = var(block_means)

    # Standard Error (SE)
    SE = sqrt(var_blocks / n_blocks)

    return SE
end


# τ_int = 1/2 + Σ_{t=1}^{∞} ρ(t)
function estimate_tau(data::Vector{Float64}; max_lag::Int=1000)

    N = length(data)
    μ = mean(data)
    σ2 = var(data)

    τ = 0.5     # intiial value at t=0

    # Cannot send sum to infinity in practice
    for t in 1:min(max_lag, N-1)
        c = 0.0
        for i in 1:(N-t)
            c += (data[i] - μ)*(data[i+t] - μ)
        end
        c /= (N - t)

        ρ = c / σ2

        # Stop summing once ρ becomes negative (prevents noise from blowing up τ artificially)
        if ρ <= 0
            break
        end

        τ += ρ
    end

    return τ
end

function compute_logpsi(nq, params, L)

    vq = params.vq
    logpsi = 0.0

    for m in 1:(L ÷ 2)
        k = m + 1
        logpsi -= vq[m] * abs2(nq[k]) / (2L)
    end

    return logpsi
end

function compute_delta_logpsi(nq, from, to, phase, params, L)

    vq = params.vq

    log_R = 0.0

    halfL = L ÷ 2

    @inbounds for m in 1:halfL

        k = m + 1

        Δnq = phase[k,to] - phase[k,from]

        a = nq[k]
        b = Δnq

        log_R += vq[m] * (2 * real(conj(a) * b) + abs2(b))

    end

    return -log_R / (2L)

end


function build_phase_table(L)

    phase = Matrix{ComplexF64}(undef, L, L)

    for site in 1:L
        for k in 1:L
            q = 2π*(k-1)/L
            phase[site,k] = exp(im*q*(site-1))
        end
    end

    return phase

end


function update_nq!(nq, i, j, phase)

    for k in eachindex(nq)
        nq[k] += phase[j,k] - phase[i,k]
    end

end