using Test
using VMCBoseHubbard

@testset "configuration history output" begin
    lattice = Lattice1D(4)
    system = System(1.0, 1.0, 4, lattice)
    n_max = 4
    wavefunctions = (
        GutzwillerWavefunction(1.0, n_max),
        JastrowWavefunction(zeros(length(lattice_shell_distances(lattice)))),
    )

    for wavefunction in wavefunctions
        history = IOBuffer()
        MC_integration(
            system,
            wavefunction,
            n_max;
            num_walkers = 2,
            num_MC_steps = 4,
            num_equil_steps = 0,
            block_size = 1,
            configuration_history = history,
            configuration_metadata = Dict(
                "dimension" => "1",
                "N" => "4",
                "U_over_t" => "1.0",
            ),
        )

        lines = split(chomp(String(take!(history))), '\n')
        @test Set(lines[1:3]) == Set([
            "# dimension=1",
            "# N=4",
            "# U_over_t=1.0",
        ])
        @test lines[4] == "step,site_1,site_2,site_3,site_4"
        rows = [parse.(Int, split(line, ',')) for line in lines[5:end]]
        @test [row[1] for row in rows] == collect(0:4)
        @test all(sum(row[2:end]) == system.N for row in rows)
        @test all(all(>=(0), row[2:end]) for row in rows)
    end
end
