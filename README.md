# Documentation

## General Desciption
This code currently estimates the ground state energy (zero temperature) for the one-dimensional Bose-Hubbard model in the canonical ensemble given system parameters (L, N, n_max, U, t). The ground state energy is estimated via Monte Carlo simulations for integration and Gradient Descent with Stochastic Reconfiguration for parameter optimization. Currently, there is an option to run the code with either the Gutzwiller or Jastrow trial wavefunction.

For multiple interaction runs, please see "gutzwiller_run_benchmark.jl" and "jastrow_run_benchmark.jl" under "Examples".

## Installation

Installing this package requires Julia 1.11.7 or later. To install, clone this repository, run julia from the root directory of the repository, and then run the following command in the Julia REPL:

```julia 
] activate .
] instantiate
```

## Dependencies
- Random
- Statistics
- LinearAlgebra
- ProgressMeter
- SpecialFunctions
- LogExpFunctions

To install the dependencies run the following command in a Julia REPL:
 
```julia 
] add Random Statistics LinearAlgebra ProgressBars SpecialFunctions LogExpFunctions ArgParse Printf TimerOutputs
```

## Usage

In order to get a quick idea of the options, which the code accepts:
```bash
julia pigsfli.jl --help
```

### Command Lines Options
```bash
  -L, --length L:                                       Size of 1D lattice (type: Int64)

  -N, --particle-number N:                              Total number of particles (type: Int64)

  -U, --interaction U:                                  Interaction strength (type: Float64)

  -t, --t T:                                            Hopping parameter (type: Float64, default: 1.0)

  --trial-state TRIAL-STATE:                            Trial state type (gutzwiller, jastrow)

  --kappa KAPPA:                                        Initial Gutzwiller parameter (type: Float64, default: 1.0)

  --jastrow-potentials VR_INIT:                         Initial Jastrow parameters (default: "")

  --n-max N_MAX:                                        Maximum site occupancy (type: Int64, default: -1)

  --eta ETA:                                            SR learning rate (type: Float64, default: 0.01)

  --seed SEED:                                          Random number generator seed (type: Int64, default: 1234)

  --opt-num-walkers OPT_NUM_WALKERS:                    Optimization walkers (type: Int64, default:100)

  --opt-num-MC-steps OPT_NUM_MC_STEPS:                  Optimization MC steps (type: Int64, default: 5000)

  --opt-num-equil-steps OPT_NUM_EQUIL_STEPS:            Optimization equilibration steps (type: Int64, default: 1000)

  --opt-block-size OPT_BLOCK_SIZE:                      Optimization block size (type: Int64, default: 500)

  --final-num-walkers FINAL_NUM_WALKERS:                Final MC walkers (type: Int64, default: 100)

  --final-num-MC-steps FINAL_NUM_MC_STEPS:              Final MC steps (type: Int64, default: 5000)

  --final-num-equil-steps FINAL_NUM_EQUIL_STEPS:        Final MC equilibration steps (type: Int64, default: 1000)

  --final-block-size FINAL_BLOCK_SIZE:                  Final MC block size (type: Int64, default: 500)

  --output-dir OUTPUT-DIR:                              Base output directory (default: "data/VMC")

  --save-history:                                       Save SR optimization history

  --skip-timing:                                        Disable timing output
  
  -h, --help                                            show this help message and exit
```