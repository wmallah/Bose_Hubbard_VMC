using Pkg
Pkg.activate("../")

include("../src/VMCBoseHubbard.jl")
using .VMCBoseHubbard

import ..VMCBoseHubbard: MC_integration_Jastrow
import ..VMCBoseHubbard: optimize_jastrow_SR
import ..VMCBoseHubbard: JastrowParams

L = 6
lattice = Lattice1D(L)
sys = System(1.0, 2.0, 0.0, lattice)

params = JastrowParams(zeros(fld(L,2)))

params_opt, history = optimize_jastrow_SR(
    sys,
    params,
    6,
    6;
    num_walkers = 20,
    num_MC_steps = 200,
    num_equil_steps = 50,
    block_size = 20
)