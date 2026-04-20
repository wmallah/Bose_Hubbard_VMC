using Pkg
Pkg.activate("../")

include("../src/VMCBoseHubbard.jl")
using .VMCBoseHubbard

import ..VMCBoseHubbard: MC_integration_Jastrow
import ..VMCBoseHubbard: estimate_tau
import ..VMCBoseHubbard: JastrowParams
import ..VMCBoseHubbard: compute_logpsi_realspace
import ..VMCBoseHubbard: compute_delta_logpsi_realspace
import ..VMCBoseHubbard: acceptance_probability_realspace_jastrow


function test_delta_logpsi_random_nn(num_tests::Int = 1000; max_occupancy::Int = 3)
    for L in 4:16
        Rmax = fld(L, 2)

        for test in 1:num_tests
            vr = randn(Rmax + 1)
            ψ = JastrowParams(copy(vr))

            n = rand(0:max_occupancy, L)
            if sum(n) == 0
                n[rand(1:L)] = 1
            end

            occupied_sites = findall(x -> x > 0, n)
            from_site = rand(occupied_sites)

            # nearest neighbors on a 1D ring
            nn_sites = [mod1(from_site - 1, L), mod1(from_site + 1, L)]
            to_site = rand(nn_sites)

            n_new = copy(n)
            n_new[from_site] -= 1
            n_new[to_site] += 1

            Δ_direct = compute_logpsi_realspace(n_new, ψ) - compute_logpsi_realspace(n, ψ)
            Δ_fast   = compute_delta_logpsi_realspace(n, from_site, to_site, ψ)

            if !isapprox(Δ_direct, Δ_fast; atol=1e-12, rtol=1e-12)
                println("Mismatch found!")
                println("L = ", L)
                println("n = ", n)
                println("vr = ", vr)
                println("from_site = ", from_site, ", to_site = ", to_site)
                println("n_new = ", n_new)
                println("Δ_direct = ", Δ_direct)
                println("Δ_fast   = ", Δ_fast)
                println("difference = ", abs(Δ_direct - Δ_fast))
                return false
            end
        end
    end

    println("All randomized nearest-neighbor delta-logpsi tests passed.")
    return true
end

# test_delta_logpsi_random_nn()


using SpecialFunctions: loggamma

function compute_logabspsi_realspace(n::Vector{Int}, vr::Vector{Float64})
    logpsi_j = compute_logpsi_realspace(n, vr)

    logphi0 = 0.0
    for ni in n
        logphi0 -= 0.5 * loggamma(ni + 1)
    end

    return logpsi_j + logphi0
end


function test_acceptance_ratio_random(num_tests::Int=1000)
    for L in (4, 5, 6, 7, 8)
        Rmax = fld(L, 2)

        for test in 1:num_tests
            vr = randn(Rmax)

            n = rand(0:3, L)
            if sum(n) == 0
                n[rand(1:L)] = 1
            end

            occupied_sites = findall(>(0), n)
            from_site = rand(occupied_sites)
            to_site = rand(setdiff(1:L, [from_site]))

            n_new = copy(n)
            n_new[from_site] -= 1
            n_new[to_site] += 1

            # direct ratio from full wavefunction
            logabs_old = compute_logabspsi_realspace(n, vr)
            logabs_new = compute_logabspsi_realspace(n_new, vr)
            ratio_direct = exp(2.0 * (logabs_new - logabs_old))

            # fast ratio from Δlogpsi + condensate factor
            Δlogpsi = compute_delta_logpsi_realspace(n, from_site, to_site, vr)
            ratio_fast = exp(2.0 * Δlogpsi) * (n[from_site] / (n[to_site] + 1))

            if !isapprox(ratio_direct, ratio_fast; atol=1e-12, rtol=1e-12)
                println("Mismatch found!")
                println("L = ", L)
                println("n = ", n)
                println("vr = ", vr)
                println("from_site = ", from_site, ", to_site = ", to_site)
                println("ratio_direct = ", ratio_direct)
                println("ratio_fast   = ", ratio_fast)
                println("difference = ", abs(ratio_direct - ratio_fast))
                return false
            end
        end
    end

    println("All randomized acceptance-ratio tests passed.")
    return true
end


# test_acceptance_ratio_random()


using SpecialFunctions: loggamma

# ------------------------------------------------------------------
# log-amplitudes for the two candidate conventions
# ------------------------------------------------------------------

function logamp_pure_jastrow(n::Vector{Int}, ψ::JastrowParams)
    return compute_logpsi_realspace(n, ψ)
end

function logamp_bosonic_jastrow(n::Vector{Int}, ψ::JastrowParams)
    logpsi = compute_logpsi_realspace(n, ψ)
    logfact = 0.0
    for ni in n
        logfact += loggamma(ni + 1)   # log(ni!)
    end
    return logpsi - 0.5 * logfact
end

# ------------------------------------------------------------------
# exact log |psi_new / psi_old|^2 from a chosen amplitude convention
# ------------------------------------------------------------------

function exact_logprob_ratio_from_amplitude(
    n::Vector{Int},
    from_site::Int,
    to_site::Int,
    ψ::JastrowParams;
    convention::Symbol = :pure
)
    n_new = copy(n)
    n_new[from_site] -= 1
    n_new[to_site] += 1

    logamp_old =
        convention == :pure    ? logamp_pure_jastrow(n, ψ) :
        convention == :bosonic ? logamp_bosonic_jastrow(n, ψ) :
        error("Unknown convention: $convention")

    logamp_new =
        convention == :pure    ? logamp_pure_jastrow(n_new, ψ) :
        convention == :bosonic ? logamp_bosonic_jastrow(n_new, ψ) :
        error("Unknown convention: $convention")

    return 2.0 * (logamp_new - logamp_old)
end

# ------------------------------------------------------------------
# exact kinetic matrix element for one hop from_site -> to_site
# ------------------------------------------------------------------

function exact_kinetic_hop_contribution(
    n::Vector{Int},
    from_site::Int,
    to_site::Int,
    t::Float64,
    ψ::JastrowParams;
    convention::Symbol = :pure
)
    @assert n[from_site] > 0

    n_new = copy(n)
    n_new[from_site] -= 1
    n_new[to_site] += 1

    logamp_old =
        convention == :pure    ? logamp_pure_jastrow(n, ψ) :
        convention == :bosonic ? logamp_bosonic_jastrow(n, ψ) :
        error("Unknown convention: $convention")

    logamp_new =
        convention == :pure    ? logamp_pure_jastrow(n_new, ψ) :
        convention == :bosonic ? logamp_bosonic_jastrow(n_new, ψ) :
        error("Unknown convention: $convention")

    ratio = exp(logamp_new - logamp_old)

    return -t * sqrt((n[to_site] + 1) * n[from_site]) * ratio
end

# ------------------------------------------------------------------
# main randomized test
# ------------------------------------------------------------------

function test_jastrow_convention_random(
    num_tests::Int = 1000;
    max_occupancy::Int = 3,
    t::Float64 = 1.0,
    atol::Float64 = 1e-12,
    rtol::Float64 = 1e-12,
    verbose_fail::Bool = true
)
    max_accept_err_pure = 0.0
    max_accept_err_bos  = 0.0
    max_kin_err_pure    = 0.0
    max_kin_err_bos     = 0.0

    for L in 4:16
        Rmax = fld(L, 2)

        for test in 1:num_tests
            vr = randn(Rmax + 1)
            ψ = JastrowParams(copy(vr))

            n = rand(0:max_occupancy, L)
            if sum(n) == 0
                n[rand(1:L)] = 1
            end

            occupied_sites = findall(x -> x > 0, n)
            from_site = rand(occupied_sites)

            # nearest-neighbor hop on a 1D ring
            nn_sites = [mod1(from_site - 1, L), mod1(from_site + 1, L)]
            to_site = rand(nn_sites)

            # -----------------------------
            # acceptance / probability ratio
            # -----------------------------
            logratio_code = acceptance_probability_realspace_jastrow(n, from_site, to_site, ψ)

            logratio_pure = exact_logprob_ratio_from_amplitude(
                n, from_site, to_site, ψ; convention=:pure
            )

            logratio_bos = exact_logprob_ratio_from_amplitude(
                n, from_site, to_site, ψ; convention=:bosonic
            )

            err_accept_pure = abs(logratio_code - logratio_pure)
            err_accept_bos  = abs(logratio_code - logratio_bos)

            max_accept_err_pure = max(max_accept_err_pure, err_accept_pure)
            max_accept_err_bos  = max(max_accept_err_bos,  err_accept_bos)

            # -----------------------------
            # kinetic contribution for this hop
            # code convention:
            #   -t * n[from_site] * exp(Δlogpsi)
            # -----------------------------
            Δlogpsi = compute_delta_logpsi_realspace(n, from_site, to_site, ψ)
            kin_code = -t * n[from_site] * exp(Δlogpsi)

            kin_pure = exact_kinetic_hop_contribution(
                n, from_site, to_site, t, ψ; convention=:pure
            )

            kin_bos = exact_kinetic_hop_contribution(
                n, from_site, to_site, t, ψ; convention=:bosonic
            )

            err_kin_pure = abs(kin_code - kin_pure)
            err_kin_bos  = abs(kin_code - kin_bos)

            max_kin_err_pure = max(max_kin_err_pure, err_kin_pure)
            max_kin_err_bos  = max(max_kin_err_bos,  err_kin_bos)

            # Optional hard fail if code matches neither convention closely
            # This is intentionally loose compared to the summary printout.
            good_accept = isapprox(logratio_code, logratio_pure; atol=atol, rtol=rtol) ||
                          isapprox(logratio_code, logratio_bos;  atol=atol, rtol=rtol)

            good_kin = isapprox(kin_code, kin_pure; atol=atol, rtol=rtol) ||
                       isapprox(kin_code, kin_bos;  atol=atol, rtol=rtol)

            if !(good_accept && good_kin)
                if verbose_fail
                    println("Potential mismatch found!")
                    println("L = ", L)
                    println("n = ", n)
                    println("vr = ", vr)
                    println("from_site = ", from_site, ", to_site = ", to_site)
                    println()
                    println("Acceptance log-ratio:")
                    println("  code    = ", logratio_code)
                    println("  pure    = ", logratio_pure)
                    println("  bosonic = ", logratio_bos)
                    println()
                    println("Single-hop kinetic contribution:")
                    println("  code    = ", kin_code)
                    println("  pure    = ", kin_pure)
                    println("  bosonic = ", kin_bos)
                end
                return false
            end
        end
    end

    println("All randomized Jastrow convention tests passed.")
    println()
    println("Maximum acceptance-log-ratio errors:")
    println("  pure    = ", max_accept_err_pure)
    println("  bosonic = ", max_accept_err_bos)
    println()
    println("Maximum single-hop kinetic errors:")
    println("  pure    = ", max_kin_err_pure)
    println("  bosonic = ", max_kin_err_bos)

    if max_accept_err_bos < max_accept_err_pure && max_kin_err_bos < max_kin_err_pure
        println()
        println("Conclusion: your current acceptance + kinetic formulas match the BOSONIC-factorial convention.")
    elseif max_accept_err_pure < max_accept_err_bos && max_kin_err_pure < max_kin_err_bos
        println()
        println("Conclusion: your current acceptance + kinetic formulas match the PURE-Jastrow convention.")
    else
        println()
        println("Conclusion: mixed result. Acceptance and kinetic pieces may not be using the same convention.")
    end

    return true
end

test_jastrow_convention_random()