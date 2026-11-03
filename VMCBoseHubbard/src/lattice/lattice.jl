abstract type AbstractLattice end

# ─── Lattice Types ─────────────────────────────────────────

"""
Purpose: store information about 1D lattice
Input: lattice size, list of neighbors
Author: Will Mallah
Last Updated: 07/04/25
"""
struct Lattice1D <: AbstractLattice
    L::Int
    neighbors::Vector{Vector{Int}}
end


"""
Purpose: generate lattice for 1D system
Input: size of lattice (number of sites), boolean value defaulted to true for periodic boundary conditions
Output: struct containing lattice information
Author: Will Mallah
Last Updated: 07/04/25
To-Do: implement open boundary conditions
"""
function Lattice1D(L::Int; periodic::Bool=true)
    if periodic
        neighbors = [Int[] for _ in 1:L]
        for i in 1:(L-1)
            push!(neighbors[i], i+1)
            push!(neighbors[i+1], i)
        end
        if periodic && L > 2        # shouldn't need this periodic boolean condition here
            push!(neighbors[1], L)
            push!(neighbors[L], 1)
        end
        return Lattice1D(L, neighbors)
    else
       error("Open boundary conditions for Lattice1D are not implemented yet.")
    end
end


"""
Purpose: store information about 2D lattice
Input: Lx, Ly, list of neighbors
Author: Will Mallah
Last Updated: 07/04/25
"""
struct Lattice2D <: AbstractLattice
    Lx::Int
    Ly::Int
    neighbors::Vector{Vector{Int}}
end


"""
Purpose: generate lattice for 2D system
Input: Lx, Ly, boolean value defaulted to true for periodic boundary conditions (periodic)
Output: struct containing lattice information
Author: Will Mallah
Last Updated: 10/02/2026
"""
function Lattice2D(Lx::Int, Ly::Int; periodic::Bool=true)
    if periodic
        neighbors = [Int[] for _ in 1:(Lx * Ly)]

        # mod1 acts to wrap coordinates around the lattice for periodic boundary conditions
        # site(x, y) returns the 1D index corresponding to the 2D coordinates (x, y): i.e., follow left-to-right, top-to-bottom order with 1 indexing
        site(x, y) = (mod1(x, Lx) - 1) + (mod1(y, Ly) - 1) * Lx + 1

        # Loop over all sites and determine their neighbors based on periodic boundary conditions
        for y in 1:Ly, x in 1:Lx
            i = site(x, y)

            # Determine the neighbors for the current site (i) in the right, left, up, and down directions
            for (dx, dy) in ((1, 0), (-1, 0), (0, 1), (0, -1))  # right, left, up, down
                j = site(x + dx, y + dy)
                if j ∉ neighbors[i]
                    push!(neighbors[i], j)
                end
                if i ∉ neighbors[j]
                    push!(neighbors[j], i)
                end
            end
        end
    else
        error("Open boundary conditions for Lattice2D are not implemented yet.")
    end

    return Lattice2D(Lx, Ly, neighbors)
end

_lattice_site_count(lattice::Lattice1D) = lattice.L
_lattice_site_count(lattice::Lattice2D) = lattice.Lx * lattice.Ly


"""
Purpose: compute the squared periodic distance between two sites on a lattice
Input: lattice (Lattice1D), i (site index), j (site index)
Output: squared periodic distance between sites i and j
Author: Will Mallah
Last Updated: 10/02/2026
"""
function _periodic_distance_squared(lattice::Lattice1D, i::Int, j::Int)
    Δx = abs(i - j)
    return min(Δx, lattice.L - Δx)^2
end


"""
Purpose: compute the squared periodic distance between two sites on a 2D lattice
Input: lattice (Lattice2D), i (site index), j (site index)
Output: squared periodic distance between sites i and j
Author: Will Mallah
Last Updated: 10/02/2026
"""
function _periodic_distance_squared(lattice::Lattice2D, i::Int, j::Int)
    x_i = mod1(i, lattice.Lx)
    y_i = fld(i - 1, lattice.Lx) + 1
    x_j = mod1(j, lattice.Lx)
    y_j = fld(j - 1, lattice.Lx) + 1

    Δx = abs(x_i - x_j)
    Δy = abs(y_i - y_j)
    dx = min(Δx, lattice.Lx - Δx)
    dy = min(Δy, lattice.Ly - Δy)
    return dx^2 + dy^2
end

_lattice_shell_squared_distances(lattice::Lattice1D) =
    [distance^2 for distance in 0:fld(lattice.L, 2)]


"""
Purpose: compute the squared periodic distances for all unique distance shells on a 2D lattice
Input: lattice (Lattice2D)
Output: sorted vector of unique squared periodic distances
Author: Will Mallah
Last Updated: 10/02/2026
"""
function _lattice_shell_squared_distances(lattice::Lattice2D)
    distances_squared = [
        dx^2 + dy^2
        for dx in 0:fld(lattice.Lx, 2)
        for dy in 0:fld(lattice.Ly, 2)
    ]
    return sort!(unique!(distances_squared))
end


"""
Purpose: compute the sorted distinct periodic Euclidean distances between lattice sites
Input: lattice (AbstractLattice)
Output: sorted vector of distinct periodic Euclidean distances
Author: Will Mallah
Last Updated: 10/02/2026
Notes:
- The on-site shell, at distance zero, is included first.
"""
function lattice_shell_distances(lattice::AbstractLattice)
    return sqrt.(Float64.(_lattice_shell_squared_distances(lattice)))
end


"""
Purpose: compute the distance-shell indices for all pairs of sites on a lattice
Input: lattice (AbstractLattice)
Output: matrix of distance-shell indices, where the on-site shell has index 1
Author: Will Mallah
Last Updated: 10/02/2026
"""
function lattice_shell_indices(lattice::AbstractLattice)
    M = _lattice_site_count(lattice)
    shell_squared_distances = _lattice_shell_squared_distances(lattice)
    shell_indices_by_distance = Dict(
        distance_squared => index
        for (index, distance_squared) in enumerate(shell_squared_distances)
    )

    indices = Matrix{Int}(undef, M, M)
    for i in 1:M, j in 1:M
        distance_squared = _periodic_distance_squared(lattice, i, j)
        indices[i, j] = shell_indices_by_distance[distance_squared]
    end
    return indices
end