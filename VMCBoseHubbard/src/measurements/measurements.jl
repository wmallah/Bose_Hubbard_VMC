# measurements.jl

"""
Purpose: calculate local potential energy (trial state independent)
Input: n (vector of integers describing the system configuration, U (interaction parameter)
Output: local potential energy for given configuration
Author: Will Mallah
Last Updated: 06/09/2026
"""
function local_potential_energy(n::Vector{Int}, U::Float64)
    Epot = 0.0
    for ni in n
        Epot += 0.5 * U * ni * (ni - 1)
    end
    return Epot
end


"""
Purpose: calculate the density-density correlation averaged over each nonzero distance shell.
Input: n (system configuration), lattice (periodic lattice)
Output: vector of shell-averaged density-density correlations, excluding the on-site shell
Author: Will Mallah
Last Updated: 10/03/2026
Notes: The 1D normalization matches the previous average over sites at each separation.
"""
function local_density_density_correlation(n::Vector{Int}, lattice::AbstractLattice)
    shell_indices = lattice_shell_indices(lattice)
    shell_pair_counts = _nonzero_shell_pair_counts(shell_indices)
    return _local_density_density_correlation(n, shell_indices, shell_pair_counts)
end

"""
Purpose: count ordered site pairs in each nonzero distance shell.
Input: shell_indices (matrix of lattice shell indices)
Output: vector of ordered-pair counts, excluding the on-site shell
Author: Will Mallah
Last Updated: 10/03/2026
"""
function _nonzero_shell_pair_counts(shell_indices::Matrix{Int})
    shell_pair_counts = zeros(Int, maximum(shell_indices) - 1)
    for shell_index in shell_indices
        if shell_index > 1
            shell_pair_counts[shell_index - 1] += 1
        end
    end
    return shell_pair_counts
end

"""
Purpose: calculate density-density correlations from precomputed shell data.
Input: n (system configuration), shell_indices (pair-to-shell matrix),
       shell_pair_counts (ordered-pair count for each nonzero shell)
Output: vector of shell-averaged density-density correlations
Author: Will Mallah
Last Updated: 10/03/2026
"""
function _local_density_density_correlation(
    n::Vector{Int},
    shell_indices::Matrix{Int},
    shell_pair_counts::Vector{Int}
)
    M = length(n)
    size(shell_indices) == (M, M) ||
        throw(DimensionMismatch("configuration size must match lattice site count"))
    length(shell_pair_counts) == maximum(shell_indices) - 1 ||
        throw(DimensionMismatch("pair counts must match nonzero lattice distance shells"))

    correlations = zeros(Float64, length(shell_pair_counts))
    for i in 1:M, j in 1:M
        shell_index = shell_indices[i, j]
        if shell_index > 1
            correlations[shell_index - 1] += n[i] * n[j]
        end
    end

    return correlations ./ shell_pair_counts
end


# ── Gutzwiller ────────────────────────────────────────────────────────────────
"""
Purpose: calculate local kinetic energy for the Gutzwiller trial state
Input: n (system configuration), t (hopping parameter), n_max (maximum site occupancy),
       ψ (Jastrow wavefunction), lattice, shell_indices (matrix of lattice shell indices)
Output: local kinetic energy of given configuration for Gutzwiller trial state
Author: Will Mallah
Last Updated: 06/09/2026
"""
function local_kinetic_energy_gutzwiller(
    n::Vector{Int},
    t::Float64,
    n_max::Int,
    ψ::GutzwillerWavefunction,
    lattice
)
    log_f = ψ.log_f
    M = length(n)
    E_kin = 0.0

    for i in 1:M
        for j in lattice.neighbors[i]
            if j > i
                # hop j → i
                if n[j] > 0 && n[i] < n_max
                    log_R = (log_f[n[i] + 2] + log_f[n[j]]) - (log_f[n[i] + 1] + log_f[n[j] + 1])
                    E_kin -= t * sqrt((n[i] + 1) * n[j]) * exp(log_R)
                end

                # hop i → j
                if n[i] > 0 && n[j] < n_max
                    log_R = (log_f[n[j] + 2] + log_f[n[i]]) - (log_f[n[j] + 1] + log_f[n[i] + 1])
                    E_kin -= t * sqrt((n[j] + 1) * n[i]) * exp(log_R)
                end
            end
        end
    end

    return E_kin
end


"""
Purpose: calculate total local energy for the Gutzwiller trial state
Input: n (system configuration), sys (system struct), n_max (maximum site occupancy),
       ψ (Jastrow wavefunction), shell_indices (optional precomputed shell-index matrix)
Output: total local energy of given configuration for Gutzwiller trial state
Author: Will Mallah
Last Updated: 06/09/2026
"""
function local_energy_gutzwiller(n::Vector{Int}, ψ::GutzwillerWavefunction, sys::System, n_max::Int64)
    t, U = sys.t, sys.U
    lattice = sys.lattice

    E_pot = local_potential_energy(n, U)
    E_kin = local_kinetic_energy_gutzwiller(n, t, n_max, ψ, lattice)
    return E_kin + E_pot, E_kin, E_pot
end


"""
Purpose: calculate the derivative of the Gutzwiller log-wavefunction with respect to κ.
Input: n (vector of integers describing the system configuration)
Output: derivative of the log-wavefunction with respect to κ
Author: Will Mallah
Last Updated: 10/03/2026
"""
function logpsi_derivative_gutzwiller(n::Vector{Int})
    return -0.5 * sum(n .^ 2)
end


# ── Jastrow ───────────────────────────────────────────────────────────────────
"""
Purpose: calculate local kinetic energy for the Jastrow trial state.
Input: n (system configuration), t (hopping parameter), n_max (maximum site occupancy),
       ψ (Jastrow wavefunction), lattice, shell_indices (matrix of lattice shell indices)
Output: local kinetic energy of given configuration for Jastrow trial state
Author: Will Mallah
Last Updated: 06/09/2026
"""
function local_kinetic_energy_jastrow(
    n::Vector{Int},
    t::Float64,
    n_max::Int,
    ψ::JastrowWavefunction,
    lattice::AbstractLattice,
    shell_indices::Matrix{Int}
)
    M = length(n)
    Ekin = 0.0

    for i in 1:M
        for j in lattice.neighbors[i]
            if j > i
                # hop j -> i gives a_i^† a_j. The Jastrow ratio here excludes
                # the condensate state's 1/sqrt(prod(n_i!)) factor; combining
                # that factor with the bosonic matrix element leaves n[j].
                if n[j] > 0 && n[i] < n_max
                    Δlogpsi = compute_delta_logpsi_jastrow(n, j, i, ψ, shell_indices)
                    Ekin -= t * n[j] * exp(Δlogpsi)
                end

                # hop i -> j gives a_j^† a_i
                if n[i] > 0 && n[j] < n_max
                    Δlogpsi = compute_delta_logpsi_jastrow(n, i, j, ψ, shell_indices)
                    Ekin -= t * n[i] * exp(Δlogpsi)
                end
            end
        end
    end

    return Ekin
end


"""
Purpose: calculate total local energy for the Jastrow trial state
Input: n (vector of integers describing the system configuration, ψ (wavefunction struct), sys (system struct), n_max (maximum site occupancy)
Output: total local energy of given configuration for Jastrow trial state
Author: Will Mallah
Last Updated: 06/09/2026
"""
function local_energy_jastrow(
    n::Vector{Int},
    sys::System,
    n_max::Int,
    ψ::JastrowWavefunction,
    shell_indices::Matrix{Int} = lattice_shell_indices(sys.lattice)
)
    t, U = sys.t, sys.U
    lattice = sys.lattice

    Epot = local_potential_energy(n, U)
    Ekin = local_kinetic_energy_jastrow(n, t, n_max, ψ, lattice, shell_indices)
    return Ekin + Epot, Ekin, Epot
end


"""
Purpose: return the derivatives of the real-space Jastrow log-wavefunction with respect to its shell potentials.
Input: n (vector of integers describing the system configuration), lattice (lattice struct)
Output: vector of derivatives of the log-wavefunction with respect to each shell potential
Author: Will Mallah
Last Updated: 10/03/2026
Notes: Uses symmetric pair counting, so each ordered site pair contributes half to its distance shell.
"""
function logpsi_derivatives_jastrow(n::Vector{Int}, lattice::AbstractLattice)
    return _logpsi_derivatives_jastrow(n, lattice_shell_indices(lattice))
end


"""
Purpose: calculate Jastrow log-wavefunction derivatives using precomputed shell indices.
Input: n (vector of integers describing the system configuration), shell_indices (matrix of lattice shell indices)
Output: vector of derivatives of the log-wavefunction with respect to each shell potential
Author: Will Mallah
Last Updated: 10/03/2026
Notes: Each shell derivative sums -n[i] * n[j] / 2 over all ordered site pairs in that shell.
"""
function _logpsi_derivatives_jastrow(n::Vector{Int}, shell_indices::Matrix{Int})
    M = length(n)
    size(shell_indices) == (M, M) ||
        throw(DimensionMismatch("configuration size must match lattice site count"))

    O = zeros(Float64, maximum(shell_indices))
    for i in 1:M, j in 1:M
        O[shell_indices[i, j]] -= 0.5 * n[i] * n[j]
    end

    return O
end


"""
Purpose: calculate the change in the Jastrow log-wavefunction for a single particle hop.
Input: n (system configuration), from_site (hop source site), to_site (hop target site),
       ψ (Jastrow wavefunction), shell_indices (matrix of lattice shell indices)
Output: change in the Jastrow log-wavefunction
Author: Will Mallah
Last Updated: 10/03/2026
Notes: Uses symmetric pair counting and only sums contributions involving either hopped site.
"""
function compute_delta_logpsi_jastrow(
    n::Vector{Int},
    from_site::Int,
    to_site::Int,
    ψ::JastrowWavefunction,
    shell_indices::Matrix{Int}
)
    M = length(n)
    size(shell_indices) == (M, M) ||
        throw(DimensionMismatch("configuration size must match lattice site count"))
    length(ψ.vr) == maximum(shell_indices) ||
        throw(DimensionMismatch("Jastrow parameter count must match the lattice distance-shell count"))
    checkbounds(Bool, n, from_site) || throw(BoundsError(n, from_site))
    checkbounds(Bool, n, to_site) || throw(BoundsError(n, to_site))
    @assert from_site != to_site
    @assert n[from_site] > 0
    vr = ψ.vr
    Δlogpsi = 0.0

    for j in 1:M
        Δlogpsi -=
            (vr[shell_indices[to_site, j]] - vr[shell_indices[from_site, j]]) * n[j]
    end
    Δlogpsi -= 0.5 * (
        vr[shell_indices[from_site, from_site]] +
        vr[shell_indices[to_site, to_site]] -
        2 * vr[shell_indices[from_site, to_site]]
    )

    return Δlogpsi
end