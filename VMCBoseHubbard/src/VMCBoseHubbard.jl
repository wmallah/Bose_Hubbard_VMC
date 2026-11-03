module VMCBoseHubbard

# ── Dependencies ──────────────────────────────────────────────
using Random, Statistics
using LinearAlgebra
using ProgressMeter
using SpecialFunctions, LogExpFunctions

# ── Source files ──────────────────────────────────────────────
include("lattice/lattice.jl")
include("system/system.jl")
include("wavefunction/wavefunction.jl")
include("MC/moves.jl")
include("MC/MC_integration.jl")
include("measurements/measurements.jl")
include("optimizer/gradient_descent.jl")

# ── Public API ────────────────────────────────────────────────
# Lattice
export Lattice1D, Lattice2D, lattice_shell_distances, lattice_shell_indices

# System
export System

# Wavefunction
export GutzwillerWavefunction, JastrowWavefunction

# MC
export MC_integration
export VMCResults

# Optimizer
export optimize_SR

end