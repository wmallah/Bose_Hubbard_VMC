using Pkg
Pkg.activate("../")

include("../src/VMCBoseHubbard.jl")
using .VMCBoseHubbard

import ..VMCBoseHubbard: MC_integration_Jastrow
import ..VMCBoseHubbard: estimate_tau
import ..VMCBoseHubbard: JastrowParams
import ..VMCBoseHubbard: compute_logpsi_realspace
import ..VMCBoseHubbard: compute_delta_logpsi_realspace


function test_delta_logpsi_random(num_tests::Int=1000)
    for L in (4, 5, 6, 7, 8)
        Rmax = fld(L, 2)

        for test in 1:num_tests
            vr = randn(Rmax)

            # random occupation configuration with at least one boson somewhere
            n = rand(0:3, L)
            if sum(n) == 0
                n[rand(1:L)] = 1
            end

            # choose a legal from_site
            occupied_sites = findall(>(0), n)
            from_site = rand(occupied_sites)

            # choose a different to_site
            to_site = rand(setdiff(1:L, [from_site]))

            n_new = copy(n)
            n_new[from_site] -= 1
            n_new[to_site] += 1

            Δ_direct = compute_logpsi_realspace(n_new, vr) - compute_logpsi_realspace(n, vr)
            Δ_fast   = compute_delta_logpsi_realspace(n, from_site, to_site, vr)

            if !isapprox(Δ_direct, Δ_fast; atol=1e-12, rtol=1e-12)
                println("Mismatch found!")
                println("L = ", L)
                println("n = ", n)
                println("vr = ", vr)
                println("from_site = ", from_site, ", to_site = ", to_site)
                println("Δ_direct = ", Δ_direct)
                println("Δ_fast   = ", Δ_fast)
                println("difference = ", abs(Δ_direct - Δ_fast))
                return false
            end
        end
    end

    println("All randomized delta-logpsi tests passed.")
    return true
end

# test_delta_logpsi_random()


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


test_acceptance_ratio_random()