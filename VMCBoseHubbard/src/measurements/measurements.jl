
function signed_logsumexp(logvals, signs)
    m = maximum(logvals)
    s = 0.0
    for (lv, sg) in zip(logvals, signs)
        s += sg * exp(lv - m)
    end
    return m + log(abs(s)), sign(s)
end


#=
Purpose: calculate the local energy
Input: n (vector of integers describing the system state), ψ (wavefunction struct), sys (system struct), n_max (maximum site occupancy)
Output: total local energy (kinetic + potential - chemical), kinetic energy, potential energy
Author: Will Mallah
Last Updated: 01/25/26
=#
function local_energy(n::Vector{Int}, ψ::GutzwillerWavefunction, sys::System, n_max::Int64, grand_canonical::Bool, projective::Bool)
    log_f = ψ.f                     # shared Gutzwiller coefficient vector
    t, U, μ = sys.t, sys.U, sys.μ
    lattice = sys.lattice
    L = length(n)

    log_E_kin_contributions = Float64[]
    signs = Int[]
    E_pot = 0.0

    # Potential energy term
    for i in 1:L
        E_pot += (U / 2) * n[i] * (n[i] - 1)    # no need for f-dependent terms; cancels in ratio
    end

    # Kinetic energy term
    for i in 1:L
        for j in lattice.neighbors[i]
            if j > i
                # hop j → i
                if hop_possible(n, j, i, n_max)
                    # log R = log[ f(ni+1) f(nj-1) / ( f(ni) f(nj) ) ]
                    log_R1 = (log_f[n[i] + 2] + log_f[n[j]]) - (log_f[n[i] + 1] + log_f[n[j] + 1])
                    log_E_kin = 0.5*log((n[i] + 1) * n[j]) + log_R1
                    push!(log_E_kin_contributions, log_E_kin)
                    push!(signs, -1)    # kinetic term is negative
                end

                # hop i → j
                if hop_possible(n, i, j, n_max)
                    # log R = log[ f(nj+1) f(ni-1) / ( f(nj) f(ni) ) ]
                    log_R2 = (log_f[n[j] + 2] + log_f[n[i]]) - (log_f[n[j] + 1] + log_f[n[i] + 1])
                    log_E_kin = 0.5 * log((n[j] + 1) * n[i]) + log_R2
                    push!(log_E_kin_contributions, log_E_kin)
                    push!(signs, -1)    # kinetic term is negative
                end
            end
        end
    end

    log_abs_E, sign_E = signed_logsumexp(log_E_kin_contributions, signs)
    E_kin = sign_E * t * exp(log_abs_E)

    # Chemical potential correction
    N = sum(n)
    
    if grand_canonical && !projective
        return E_kin + E_pot - μ*N, E_kin, E_pot
    else
        return E_kin + E_pot, E_kin, E_pot
    end
end


#=
Purpose: estimate the gradient of the energy and metric for use in natural gradient descent optimization
Input: result from Monte Carlo integration and block_size for autocorrelation blocking approximation
Output: energy gradient, standard error of energy gradient, and metric
Author: Will Mallah
Last Updated: 02/22/26
=#
function estimate_energy_gradient_and_metric(result::VMCResults;
                                             block_size::Int = 200)

    E = result.energies
    O = result.derivative_log_psi

    if isempty(E) || isempty(O)
        return NaN, NaN, NaN
    end

    N = length(E)

    if N != length(O)
        error("Energy and derivative arrays must have same length.")
    end

    # ---------- Means ----------
    mean_E = mean(E)
    mean_O = mean(O)

    # ---------- Gradient samples ----------
    # X_i = 2 (O_i E_i - <O> E_i)
    X = 2 .* (O .* E .- mean_O .* E)

    g = mean(X)

    # ---------- SR metric ----------
    S = mean(O.^2) - mean_O^2

    # ---------- Blocking error for gradient ----------
    n_blocks = div(N, block_size)

    # The minimum number of blocks is arbitrary at the moment, but the more blocks you have, the smaller the error
    if n_blocks < 5
        error("Not enough blocks for reliable gradient error estimate.")
    end

    # Truncate some data so that blocks are same size
    X_trunc = X[1:(n_blocks * block_size)]
    # Reshape data so each column is one block and each row is an entry in that block
    blocks = reshape(X_trunc, block_size, n_blocks)

    # Generate vector that contains mean of each respective block as its elements
    block_means = vec(mean(blocks, dims=1))
    # Take the variance of these mean values
    var_blocks = var(block_means)

    # Calculate the Standard Error of the Gradient (SEG)
    SE_g = sqrt(var_blocks / n_blocks)

    return g, SE_g, S
end


function local_energy_jastrow(L, sys, walker::Walker, params::JastrowParams, phase)
    t, U = sys.t, sys.U
    lattice = sys.lattice
    n = walker.n
    nq = walker.nq

    E_pot = 0.0
    log_E_kin_contributions = Float64[]
    signs = Int[]

    npair = num_pair_modes(L)

    # -------------------------
    # Interaction energy
    # -------------------------
    @inbounds for i in 1:L
        ni = n[i]
        E_pot += (U / 2) * ni * (ni - 1)
    end

    # -------------------------
    # Kinetic energy
    # -------------------------
    @inbounds for i in 1:L
        for j in lattice.neighbors[i]

            # avoid double counting bonds
            if j > i

                # -------------------------
                # hop j -> i
                # -------------------------
                if n[j] > 0
                    Δnq = similar(nq)

                    for k in eachindex(nq)
                        Δnq[k] = phase[k, i] - phase[k, j]
                    end

                    log_R = 0.0

                    # paired modes: q = 2πm/L, m = 1, ..., floor((L-1)/2)
                    for m in 1:npair
                        k = m + 1
                        old = abs2(nq[k])
                        new = abs2(nq[k] + Δnq[k])
                        log_R -= params.vpair[m] * (new - old) / L
                    end

                    # edge/Nyquist mode only for even L
                    if has_edge_mode(L)
                        kedge = edge_mode_index(L)
                        old = abs2(nq[kedge])
                        new = abs2(nq[kedge] + Δnq[kedge])
                        log_R -= something(params.vedge, 0.0) * (new - old) / (2L)
                    end

                    log_E_kin = 0.5 * log((n[i] + 1) * n[j]) + log_R

                    push!(log_E_kin_contributions, log_E_kin)
                    push!(signs, -1)
                end

                # -------------------------
                # hop i -> j
                # -------------------------
                if n[i] > 0
                    Δnq = similar(nq)

                    for k in eachindex(nq)
                        Δnq[k] = phase[k, j] - phase[k, i]
                    end

                    log_R = 0.0

                    # paired modes
                    for m in 1:npair
                        k = m + 1
                        old = abs2(nq[k])
                        new = abs2(nq[k] + Δnq[k])
                        log_R -= params.vpair[m] * (new - old) / L
                    end

                    # edge/Nyquist mode
                    if has_edge_mode(L)
                        kedge = edge_mode_index(L)
                        old = abs2(nq[kedge])
                        new = abs2(nq[kedge] + Δnq[kedge])
                        log_R -= something(params.vedge, 0.0) * (new - old) / (2L)
                    end

                    log_E_kin = 0.5 * log((n[j] + 1) * n[i]) + log_R

                    push!(log_E_kin_contributions, log_E_kin)
                    push!(signs, -1)
                end
            end
        end
    end

    if isempty(log_E_kin_contributions)
        E_kin = 0.0
    else
        log_abs_E, sign_E = signed_logsumexp(log_E_kin_contributions, signs)
        E_kin = sign_E * t * exp(log_abs_E)
    end

    log_abs_E, sign_E = signed_logsumexp(log_E_kin_contributions, signs)
    E_kin = sign_E * t * exp(log_abs_E)

    return E_kin + E_pot, E_kin, E_pot
end


function logpsi_derivatives(nq, L)
    npair = num_pair_modes(L)
    has_edge = has_edge_mode(L)

    Nv = npair + (has_edge ? 1 : 0)
    O = zeros(Float64, Nv)

    # Paired modes
    @inbounds for m in 1:npair
        k = m + 1
        O[m] = -abs2(nq[k]) / L
    end

    # Edge mode
    if has_edge
        kedge = edge_mode_index(L)
        O[npair + 1] = -abs2(nq[kedge]) / (2L)
    end

    return O
end