
# Optimized control points from the main script
const point_x = [100.0, 233.33333333333331, 366.66666666666663, 500.0]
const point_y = [1675.242799799218, 3432.759356004631, 6616.367492740979, 7023.098744157708]
const point_delta = [1.6228037921129532, 3.354560823532906, 3.192809860731041, 4.479124379524981]

# Internal helper function to evaluate the polynomial
function _lagrange_polynomial_eval(x, px, py)
    sum(j -> py[j] * prod(i -> i==j ? 1.0 : (x - px[i]) / (px[j] - px[i]), eachindex(px)), eachindex(px))
end

# --- Public API Functions ---

"""
    get_s(N::Real) -> Float64

Calculates the optimal steepness parameter `s` for `N` grid points.
"""
get_s(N::Real) = _lagrange_polynomial_eval(N, point_x, point_y)

"""
    get_delta(N::Integer) -> Int

Calculates the optimal integer `δ` for `N` grid points.
The result is rounded and clamped to the valid range `1 <= δ < 0.4 * N`.
"""
function get_delta(N::Integer)
    val = _lagrange_polynomial_eval(N, point_x, point_delta)
    # Clamp the value to the valid range from the optimization constraints
    clamp(round(Int, val), 1, floor(Int, 0.4 * N) - 1)
end

