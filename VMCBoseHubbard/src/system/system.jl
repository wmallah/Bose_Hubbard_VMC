#=
Purpose: store system parameters and other system information 
Input: t (hopping amplitude), U (interaction strength), N (total number of particles), lattice
Author: Will Mallah
Last Updated: 05/22/2026
=#
struct System{T <: Real}
    t::T
    U::T
    N::Int
    lattice::AbstractLattice
end