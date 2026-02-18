# VMCBoseHubbard.jl

module VMCBoseHubbard

# Include source files — they all define functions/types in *this* module
include("lattice/lattice.jl")
include("system/system.jl")
include("wavefunction/wavefunction.jl")
include("MC/utils.jl")
include("MC/moves.jl")
include("MC/MC_integration.jl")
include("measurements/measurements.jl")
include("optimizer/gradient_descent.jl")

# Export user-facing names directly
export Lattice1D, Lattice2D
export System
export GutzwillerWavefunction
export estimate_n_max
export estimate_energy_gradient_and_metric
export optimize_kappa

end
