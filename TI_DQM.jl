using LinearAlgebra
include("DQM_WEIGHTS.jl")

"""
    solve_timoshenko_beam_dqm(N, a, b, E, G, I_z, A, k_s, load, bc_start, bc_end)

Solves the static Timoshenko beam equations using the Differential Quadrature Method (DQM).

The function computes the deflection `w(x)` and the rotation of the cross-section `θ(x)`.

# Arguments
- `N::Int`: Number of grid points.
- `a::Real, b::Real`: Start and end points of the beam's domain.
- `E::Real`: Young's modulus.
- `G::Real`: Shear modulus.
- `I_z::Real`: Second moment of area of the cross-section.
- `A::Real`: Cross-sectional area.
- `k_s::Real`: Shear correction factor (e.g., 5/6 for a rectangular cross-section).
- `load::Vector`: A vector representing the distributed load `q(x)` at each grid point.
- `bc_start::String`: Boundary condition at the start ("clamped", "simply supported", or "free").
- `bc_end::String`: Boundary condition at the end ("clamped", "simply supported", or "free").

# Returns
- `(grid, w, theta)`: A tuple containing the grid points `x`, the resulting deflection `w(x)`, and the rotation `θ(x)`.
"""
function solve_timoshenko_beam_dqm(
    N::Int,
    a::Real,
    b::Real,
    E::Real,
    G::Real,
    I_z::Real,
    A::Real,
    k_s::Real,
    load::Vector,
    bc_start::String,
    bc_end::String,
)

    # --- 1. System Setup ---
    A1, grid = DQM_weights(N, a, b)
    A2 = A1 * A1
    I_x = I(N)

    # Governing equations in matrix form:
    # k_s*G*A*(w'' - θ') = -q
    # E*I_z*θ'' - k_s*G*A*(w' - θ) = 0
    #
    # This creates a 2N x 2N system matrix LHS for the solution vector [w; θ].
    LHS_11 = k_s * G * A * A2
    LHS_12 = -k_s * G * A * A1
    LHS_21 = k_s * G * A * A1
    LHS_22 = E * I_z * A2 - k_s * G * A * I_x

    LHS = [
        LHS_11 LHS_12
        LHS_21 LHS_22
    ]

    # Initialize the load vector (RHS)
    RHS = zeros(2 * N)
    RHS[1:N] = -load

    # --- 2. Apply Boundary Conditions ---
    # Delegate the boundary condition logic to a helper function.
    apply_timoshenko_bc!(LHS, RHS, A1, N, bc_start, "start")
    apply_timoshenko_bc!(LHS, RHS, A1, N, bc_end, "end")

    # --- 3. Solve and Return ---
    solution = LHS \ RHS

    # Extract deflection (w) and rotation (theta) from the solution vector
    w = solution[1:N]
    theta = solution[(N+1):(2*N)]

    return (grid, w, theta)
end


"""
    apply_timoshenko_bc!(LHS, RHS, A1, N, bc_type, position)

Helper function to apply Timoshenko beam boundary conditions. It modifies the
system matrix `LHS` and load vector `RHS` in place. The `!` at the end of the
name is a Julia convention indicating that the function modifies its arguments.

# Arguments
- `LHS`: The 2N x 2N system matrix.
- `RHS`: The 2N load vector.
- `A1`: The first-order DQM weighting matrix.
- `N`: The number of grid points.
- `bc_type`: The boundary condition type ("clamped", "simply supported", "free").
- `position`: The location to apply the BC ("start" or "end").
"""
function apply_timoshenko_bc!(LHS, RHS, A1, N, bc_type, position)
    # Determine the node indices based on the position
    if position == "start"
        w_row = 1         # Row for the first physical condition (related to w)
        theta_row = N + 1 # Row for the second physical condition (related to θ)
        deriv_row = 1     # Row from A1 matrix to use for derivatives
    elseif position == "end"
        w_row = N
        theta_row = 2 * N
        deriv_row = N
    else
        error("Position must be 'start' or 'end'.")
    end

    if bc_type == "clamped"
        # Deflection w = 0
        LHS[w_row, :] .= 0
        LHS[w_row, w_row] = 1
        RHS[w_row] = 0
        # Rotation θ = 0
        LHS[theta_row, :] .= 0
        LHS[theta_row, theta_row] = 1
        RHS[theta_row] = 0

    elseif bc_type == "simply supported"
        # Deflection w = 0
        LHS[w_row, :] .= 0
        LHS[w_row, w_row] = 1
        RHS[w_row] = 0
        # Bending Moment M = E*I*θ' = 0  =>  θ' = 0
        LHS[theta_row, :] .= 0
        LHS[theta_row, (N+1):(2*N)] = A1[deriv_row, :]
        RHS[theta_row] = 0

    elseif bc_type == "free"
        # Shear Force V = k*G*A*(w' - θ) = 0  =>  w' = θ
        LHS[w_row, :] .= 0
        LHS[w_row, 1:N] = A1[deriv_row, :]
        # The specific theta at the node is `w_row` for `w` and `N + w_row` for `θ`.
        # For start (w_row=1), it's θ_1 (col N+1). For end (w_row=N), it's θ_N (col 2N).
        LHS[w_row, N+w_row] = -1
        RHS[w_row] = 0
        # Bending Moment M = E*I*θ' = 0  =>  θ' = 0
        LHS[theta_row, :] .= 0
        LHS[theta_row, (N+1):(2*N)] = A1[deriv_row, :]
        RHS[theta_row] = 0

    else
        error("Unknown boundary condition type: $bc_type")
    end
end
