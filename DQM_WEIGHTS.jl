using LinearAlgebra

"""
    DQM_weights(N, a, b)

Generates the Differential Quadrature Method (DQM) weighting matrix and its corresponding grid points.

This function uses a Chebyshev-Gauss-Lobatto (CGL) point distribution, which clusters points
near the boundaries to ensure numerical stability for polynomial-based methods. It then
calculates the first-order derivative weighting matrix for that specific grid.

# Arguments
- `N::Int`: The number of grid points.
- `a::Real`: The starting coordinate of the domain (e.g., 0).
- `b::Real`: The ending coordinate of the domain (e.g., L).

# Returns
- `Tuple{Matrix{Float64}, Vector{Float64}}`: A tuple containing:
    1.  The `(N x N)` DQM weighting matrix for the first derivative.
    2.  A vector of `N` grid point coordinates.
"""
function DQM_weights(N::Int, a::Real, b::Real)

    # 1. Generate Chebyshev-Gauss-Lobatto (CGL) grid points using high precision
    X_i_big = [
        BigFloat(a) + 0.5 * (1 - cos((i - 1) / (N - 1) * pi)) * (BigFloat(b) - BigFloat(a)) for i = 1:N
    ]

    # 2. Pre-calculate the M_i values for efficiency
    # M_i is the product of (x_i - x_k) for all k != i.
    M_vals = [prod(X_i_big[i] - X_i_big[k] for k = 1:N if i != k) for i = 1:N]
    
    # 3. Calculate the weighting matrix (originally a_ij + a_ii)
    D_matrix = zeros(Float64, N, N)
    for i = 1:N
        row_sum = 0.0
        for j = 1:N
            if i != j
                # Off-diagonal elements
                D_matrix[i, j] = M_vals[i] / (M_vals[j] * (X_i_big[i] - X_i_big[j]))
                row_sum += D_matrix[i, j]
            end
        end
        # Diagonal elements are the negative sum of the other elements in the row
        D_matrix[i, i] = -row_sum
    end

    return Float64.(D_matrix), Float64.(X_i_big)
end
