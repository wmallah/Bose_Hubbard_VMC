
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

function local_potential_energy(n::Vector{Int}, U::Float64)
    Epot = 0.0
    for ni in n
        Epot += 0.5 * U * ni * (ni - 1)
    end
    return Epot
end


function local_kinetic_energy_jastrow(
    n::Vector{Int},
    t::Float64,
    n_max::Int,
    ψ::Wavefunction,
    lattice
)
    L = length(n)
    Ekin = 0.0

    for i in 1:L
        for j in lattice.neighbors[i]
            if j > i
                # hop j -> i gives a_i^† a_j
                if n[j] > 0 && n[i] < n_max
                    Δlogpsi = compute_delta_logpsi_realspace(n, j, i, ψ)
                    Ekin -= t * n[j] * exp(Δlogpsi)
                end

                # hop i -> j gives a_j^† a_i
                if n[i] > 0 && n[j] < n_max
                    Δlogpsi = compute_delta_logpsi_realspace(n, i, j, ψ)
                    Ekin -= t * n[i] * exp(Δlogpsi)
                end
            end
        end
    end

    return Ekin
end


function local_energy_jastrow(
    n::Vector{Int},
    sys::System,
    n_max::Int,
    ψ::Wavefunction
)
    t, U = sys.t, sys.U
    lattice = sys.lattice

    Epot = local_potential_energy(n, U)
    Ekin = local_kinetic_energy_jastrow(n, t, n_max, ψ, lattice)
    return Ekin + Epot, Ekin, Epot
end


function logpsi_derivatives_realspace(n::Vector{Int})
    L = length(n)
    Rmax = fld(L, 2)
    O = zeros(Float64, Rmax + 1)

    for idx in 1:(Rmax + 1)
        R = idx - 1

        # prefactor matches the symmetric Jastrow convention
        # logψ = -∑_R prefactor(R) * v_R * ∑_i n_i n_{i+R}
        #
        # R = 0 gets 1/2 from the usual symmetric density-density form.
        # For even L, R = L/2 also gets 1/2 because each opposite-site pair
        # appears twice in ∑_i n_i n_{i+R}.
        prefactor = 1.0
        if R == 0
            prefactor = 0.5
        elseif iseven(L) && R == Rmax
            prefactor= 0.5
        end

        SR = 0.0
        for i in 1:L
            j = mod1(i + R, L)
            SR += n[i] * n[j]
        end

        O[idx] = -prefactor * SR         # CHANGED FROM O[idx] = prefactor * SR
    end

    return O
end


function compute_logpsi_realspace(n::Vector{Int}, ψ::Wavefunction)
    vr = ψ.vr
    L = length(n)
    Rmax = fld(L, 2)

    @assert length(vr) == Rmax + 1

    logpsi = 0.0

    for idx in 1:(Rmax + 1)
        R = idx - 1

        # Symmetric real-space Jastrow convention:
        # logψ = -∑_R c_R v_R ∑_i n_i n_{i+R}
        #
        # R = 0 gets c_R = 1/2 from the usual symmetric density-density form.
        # For even L, R = L/2 also gets c_R = 1/2 because opposite-site
        # pairs are counted twice in ∑_i n_i n_{i+R}.
        prefactor = 1.0
        if R == 0
            prefactor = 0.5
        elseif iseven(L) && R == Rmax
            prefactor= 0.5
        end

        # The "weights" in the sum we are computing are the Jastrow potentials
        weight = prefactor * vr[idx]

        SR = 0.0
        for i in 1:L
            j = mod1(i + R, L)
            SR += n[i] * n[j]
        end

        logpsi -= weight * SR               # CHANGED FROM logpsi += weight * SR
    end

    return logpsi
end


function compute_delta_logpsi_realspace(
    n::Vector{Int},
    from_site::Int,
    to_site::Int,
    ψ::Wavefunction
)
    # Extract Jastrow potentials from wavefunction struct
    vr = ψ.vr
    # Extract the system size from the length of the configuration vector
    L = length(n)
    # Define the maximum difference between two sites on our 1D periodic lattice
    Rmax = fld(L, 2)

    # Assert quantities to ensure physical laws
    @assert 1 <= from_site <= L
    @assert 1 <= to_site <= L
    @assert from_site != to_site
    @assert n[from_site] > 0
    # Short-hand notation
    a = from_site
    b = to_site

    # Initialize the quantitiy we want to compute
    Δlogpsi = 0.0

    # Sum over site distances using Julia indexing (idx = 1 --> R = 0, idx = Rmax + 1 --> R = Rmax)
    for idx in 1:(Rmax + 1)
        R = idx - 1
        
        # Symmetric real-space Jastrow convention:
        # logψ = -∑_R c_R v_R ∑_i n_i n_{i+R}
        #
        # R = 0 gets c_R = 1/2 from the usual symmetric density-density form.
        # For even L, R = L/2 also gets c_R = 1/2 because opposite-site
        # pairs are counted twice in ∑_i n_i n_{i+R}.
        prefactor = 1.0
        if R == 0
            prefactor = 0.5
        elseif iseven(L) && R == Rmax
            prefactor= 0.5
        end

        # The "weights" in the sum we are computing are the Jastrow potentials
        weight = prefactor * vr[idx]

        # 
        affected_i = unique((
            a,
            b,
            mod1(a - R, L),
            mod1(b - R, L),
        ))

        # Initialize old and new operator values
        old_local = 0.0
        new_local = 0.0

        # Only sum over sites which are affected by the hopping move (won't cancel exactly in the sum)
        for i in affected_i
            # Distance from i site
            j = mod1(i + R, L)

            # Asssign old operator values
            ni_old = n[i]
            nj_old = n[j]

            # Assign new operators from hopping move
            ni_new = ni_old + (i == b ? 1 : 0) - (i == a ? 1 : 0)
            nj_new = nj_old + (j == b ? 1 : 0) - (j == a ? 1 : 0)

            # Sum non-cancelling terms
            old_local += ni_old * nj_old
            new_local += ni_new * nj_new
        end

        # Sum Jastrow part of ratio
        ΔSR = (new_local - old_local)
        Δlogpsi -= weight * ΔSR                 # CHANGED FROM Δlogpsi += weight * ΔSR
    end

    return Δlogpsi
end