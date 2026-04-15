using Plots

function realspace_to_momentum_jastrow(vr::Vector{Float64}, L::Int)
    Rmax = fld(L, 2)
    @assert length(vr) == Rmax + 1

    q = [2π * m / L for m in 0:Rmax]
    vq = zeros(Float64, Rmax + 1)

    if iseven(L)
        for iq in eachindex(q)
            qq = q[iq]

            s = vr[1]   # R = 0 term

            for r in 1:(Rmax - 1)
                s += 2.0 * vr[r + 1] * cos(qq * r)
            end

            s += vr[Rmax + 1] * cos(qq * Rmax)

            vq[iq] = s
        end
    else
        for iq in eachindex(q)
            qq = q[iq]

            s = vr[1]   # R = 0 term

            for r in 1:Rmax
                s += 2.0 * vr[r + 1] * cos(qq * r)
            end

            vq[iq] = s
        end
    end

    return q, vq
end

L = 64
Rmax = fld(L, 2)
vr = [exp(-0.2 * r) for r in 0:Rmax]

q, vq = realspace_to_momentum_jastrow(vr, L)

p = plot(
    q,
    vq,
    marker = :circle,
    linestyle = :solid,
    xlabel = "q",
    ylabel = "v_q",
    title = "Test transform: v_q versus q for L = $L",
    label = "Fourier transform of test v(r)",
    markersize = 4
)

savefig(p, "test_realspace_to_momentum_jastrow.png")
println("Saved plot to test_realspace_to_momentum_jastrow.png")