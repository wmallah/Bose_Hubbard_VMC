ENV["GKSwstype"] = "100"
using Pkg, Plots
gr()
Pkg.activate("../")

include("../src/VMCBoseHubbard.jl")
using .VMCBoseHubbard

import ..VMCBoseHubbard: MC_integration

function run_mu_point(
    sys, N_target, κ, n_max;
    num_walkers = 200,
    num_MC_steps = 5_000,
    num_equil_steps = 1_000
)
    result = MC_integration(
        sys, N_target, κ, n_max, true, false;
        num_walkers = num_walkers,
        num_MC_steps = num_MC_steps,
        num_equil_steps = num_equil_steps
    )

    PN = result.PN
    total = sum(PN)

    meanN = sum((i-1) * PN[i] for i in eachindex(PN)) / total
    N_peak = argmax(PN) - 1

    return (
        PN = PN,
        meanN = meanN,
        N_peak = N_peak,
        result = result
    )
end


function scan_mu(
    U;
    μ_vals,
    κ = 1.0,
    num_walkers = 200,
    num_MC_steps = 5_000,
    num_equil_steps = 1_000
)
    lattice = Lattice1D(L)
    n_max = 12

    data = Dict{Float64, Any}()

    for μ in μ_vals
        sys = System(t, U, μ, lattice)

        obs = run_mu_point(
            sys, N_target, κ, n_max;
            num_walkers = num_walkers,
            num_MC_steps = num_MC_steps,
            num_equil_steps = num_equil_steps
        )

        println(
            "μ = $(round(μ, digits=4)) | " *
            "⟨N⟩ = $(round(obs.meanN, digits=3)) | " *
            "N_peak = $(obs.N_peak)"
        )

        data[μ] = obs
    end

    return data
end


L = 12
N_target = 12
U = 8.0
t = 1.0

μ_vals = range(-4.0, 4.0; length = 100)

scan_data = scan_mu(
    U;
    μ_vals = μ_vals,
    κ = 3.519438908375297
)

mkpath("../analysis_scripts/figures/mu_statistics/U_$(U)")


meanNs = [scan_data[μ].meanN for μ in μ_vals]

p = plot(
    μ_vals,
    meanNs;
    xlabel = "μ",
    ylabel = "⟨N⟩",
    marker = :circle,
    lw = 2,
    label = "⟨N⟩",
    title = "Mean particle number vs μ"
)

savefig(p, "../analysis_scripts/figures/mu_statistics/U_$(U)/meanN_vs_mu.png")

hline!([N_target], ls = :dash, label = "N_target")


Npeaks = [scan_data[μ].N_peak for μ in μ_vals]

p = plot(
    μ_vals,
    Npeaks;
    xlabel = "μ",
    ylabel = "N_peak",
    marker = :square,
    lw = 2,
    label = "argmax P_N",
    title = "Dominant particle number vs μ"
)

savefig(p, "../analysis_scripts/figures/mu_statistics/U_$(U)/N_peak_vs_mu.png")

hline!([N_target], ls = :dash, label = "N_target")


# Build PN matrix: rows = N, columns = μ
N_max = length(first(values(scan_data)).PN) - 1

PN_matrix = zeros(Float64, N_max + 1, length(μ_vals))

for (j, μ) in enumerate(μ_vals)
    PN = scan_data[μ].PN
    PN_matrix[:, j] .= PN ./ sum(PN)   # normalize
end

h = heatmap(
    μ_vals,
    0:N_max,
    PN_matrix;
    xlabel = "μ",
    ylabel = "N",
    colorbar_title = "P_N",
    title = "Particle-number distribution P_N(μ)",
)

savefig(h, "../analysis_scripts/figures/mu_statistics/U_$(U)/heatmap.png")