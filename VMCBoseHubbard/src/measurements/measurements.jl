# measurements.jl

#=
Purpose: calculate local potential energy (trial state independent)
Input: n (vector of integers describing the system configuration, U (interaction parameter)
Output: local potential energy for given configuration
Author: Will Mallah
Last Updated: 06/09/2026
=#
function local_potential_energy(n::Vector{Int}, U::Float64)
    Epot = 0.0
    for ni in n
        Epot += 0.5 * U * ni * (ni - 1)
    end
    return Epot
end


#=
Purpose: calculate particle correlation function for a given configuration
Input: n (vector of integers describing the system configuration
Output: particle correlation function for given configuration
Author: Will Mallah
Last Updated: 07/01/2026
=#
function local_density_density_correlation(n::Vector{Int})
    L = length(n)
    Rmax = fld(L, 2)
    C = zeros(Float64, Rmax)

    for d in 1:Rmax
        for i in 1:L
            j = mod1(i + d, L)
            C[d] += n[i] * n[j]
        end
    end

    return C / L
end


# ── Gutzwiller ────────────────────────────────────────────────────────────────
#=
Purpose: calculate local kinetic energy for the Gutzwiller trial state
Input: n (vector of integers describing the system configuration, t (hopping parameter), n_max (maximum site occupancy), ψ (wavefunction struct), lattice (lattice struct)
Output: local kinetic energy of given configuration for Gutzwiller trial state
Author: Will Mallah
Last Updated: 06/09/2026
=#
function local_kinetic_energy_gutzwiller(
    n::Vector{Int},
    t::Float64,
    n_max::Int,
    ψ::GutzwillerWavefunction,
    lattice
)
    log_f = ψ.log_f
    L = length(n)
    E_kin = 0.0

    for i in 1:L
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


#=
Purpose: calculate total local energy for the Gutzwiller trial state
Input: n (vector of integers describing the system configuration, ψ (wavefunction struct), sys (system struct), n_max (maximum site occupancy)
Output: total local energy of given configuration for Gutzwiller trial state
Author: Will Mallah
Last Updated: 06/09/2026
=#
function local_energy_gutzwiller(n::Vector{Int}, ψ::GutzwillerWavefunction, sys::System, n_max::Int64)
    t, U = sys.t, sys.U
    lattice = sys.lattice

    E_pot = local_potential_energy(n, U)
    E_kin = local_kinetic_energy_gutzwiller(n, t, n_max, ψ, lattice)
    return E_kin + E_pot, E_kin, E_pot
end


# ── Jastrow ───────────────────────────────────────────────────────────────────
#=
Purpose: calculate local kinetic energy for the Jastrow trial state
Input: n (vector of integers describing the system configuration, t (hopping parameter), n_max (maximum site occupancy), ψ (wavefunction struct), lattice (lattice struct)
Output: local kinetic energy of given configuration for Jastrow trial state
Author: Will Mallah
Last Updated: 06/09/2026
=#
function local_kinetic_energy_jastrow(
    n::Vector{Int},
    t::Float64,
    n_max::Int,
    ψ::JastrowWavefunction,
    lattice
)
    L = length(n)
    Ekin = 0.0

    for i in 1:L
        for j in lattice.neighbors[i]
            if j > i
                # hop j -> i gives a_i^† a_j. The Jastrow ratio here excludes
                # the condensate state's 1/sqrt(prod(n_i!)) factor; combining
                # that factor with the bosonic matrix element leaves n[j].
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


#=
Purpose: calculate total local energy for the Jastrow trial state
Input: n (vector of integers describing the system configuration, ψ (wavefunction struct), sys (system struct), n_max (maximum site occupancy)
Output: total local energy of given configuration for Jastrow trial state
Author: Will Mallah
Last Updated: 06/09/2026
=#
function local_energy_jastrow(
    n::Vector{Int},
    sys::System,
    n_max::Int,
    ψ::JastrowWavefunction
)
    t, U = sys.t, sys.U
    lattice = sys.lattice

    Epot = local_potential_energy(n, U)
    Ekin = local_kinetic_energy_jastrow(n, t, n_max, ψ, lattice)
    return Ekin + Epot, Ekin, Epot
end


#=
Purpose: construct and store Gutzwiller coefficients from Gutzwiller variational parameter (κ)
Input: κ (Gutzwiller variational parameter), n_max (maximum site occupancy), logfact (pre-generated factorial values)
Output: GutzwillerWavefunction struct
Author: Will Mallah
Last Updated: 06/09/2026
=#
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

        O[idx] = -prefactor * SR
    end

    return O
end


#=
Purpose: compute the change in the log of the Jastrow exponetial piece of the wavefunction
Input: n (system configuration), from_site (hop source site), to_site (hop target site), ψ (wavefunction)
Output: change in the log of the Jastrow exponetial piece of the wavefunction
Author: Will Mallah
Last Updated: 06/09/2026
=#
function compute_delta_logpsi_realspace(
    n::Vector{Int},
    from_site::Int,
    to_site::Int,
    ψ::JastrowWavefunction
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
        Δlogpsi -= weight * ΔSR
    end

    return Δlogpsi
end