BLAS.set_num_threads(16)
"""
    find_closest_position(val, vec)

A helper function to find the index of an element in `vec` that is closest to `val`.
"""
find_closest_position(val::Real, vec::Vector{<:Real}) = argmin(abs.(vec .- val))

"""
    area_calculation(fn, X_i)

Calculates the area under a curve defined by discrete points (`fn` vs `X_i`) using the trapezoidal rule.
"""
function area_calculation(fn::Vector{<:Real}, X_i::Vector{<:Real})
    # Calculates the sum of the areas of the trapezoids formed by adjacent points
    return sum(0.5 * (fn[i] + fn[i+1]) * (X_i[i+1] - X_i[i]) for i = 1:(length(X_i)-1))
end

"""
    Smooth_load(grid, Nom_load, s, load_start, load_end)

Creates a smooth, distributed load vector over a specified grid.

The function uses the difference of two hyperbolic tangent (tanh) functions to create a
smooth "rectangular" pulse. The total area under this curve is then scaled to match the 
`Nom_load`, conserving the overall load.

# Arguments
- `grid::Vector{<:Real}`: The vector of grid points where the load is evaluated.
- `Nom_load::Real`: The total integrated force the distributed load should exert (e.g., -100 N).
- `s::Real`: A parameter controlling the steepness of the load's start and end.
- `load_start::Real`: The coordinate where the load begins.
- `load_end::Real`: The coordinate where the load ends.

# Returns
- `Vector{Float64}`: A vector representing the distributed load `q(x)` at each grid point.
"""
function Smooth_load(
    grid::Vector{<:Real},
    Nom_load::Real,
    s::Real,
    load_start::Real,
    load_end::Real,
)

    # Find grid indices closest to the specified load start and end points
    start_idx = find_closest_position(load_start, grid)
    end_idx = find_closest_position(load_end, grid)

    # 1. Generate a "unit" load profile using tanh functions
    unit_load = [
        0.5 * (tanh(s * (x - grid[start_idx])) - tanh(s * (x - grid[end_idx]))) for
        x in grid
    ]

    # 2. Calculate the area under the unit profile to find the scaling ratio
    area = area_calculation(unit_load, grid)

    # 3. Scale the profile to ensure the total force is conserved
    ratio = Nom_load / area

    return unit_load .* ratio
end
