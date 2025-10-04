using LinearAlgebra

# Assume DQM_WEIGHTS.jl is in the same directory and provides:
# DQM_weights(N, a, b) -> (A1, grid)
include("DQM_WEIGHTS.jl")

"""
    solve_timoshenko_beam_dqm(N, a, b, E, G, I_z, A, k_s, load, bc_start, bc_end)

Solves the static Timoshenko beam equations for a beam with a variable cross-section
using the Differential Quadrature Method (DQM). This version uses standard dense matrices.

The function computes the deflection `w(x)` and the rotation of the cross-section `θ(x)`.

# Arguments
- `N::Int`: Number of grid points.
- `a::Real, b::Real`: Start and end points of the beam's domain.
- `E::Real`: Young's modulus.
- `G::Real`: Shear modulus.
- `I_z::Vector`: Vector of the second moment of area at each grid point.
- `A::Vector`: Vector of the cross-sectional area at each grid point.
- `k_s::Real`: Shear correction factor.
- `load::Vector`: Vector representing the distributed load `q(x)` at each grid point.
- `bc_start::String`: Boundary condition at the start ("clamped", "simply supported", or "free").
- `bc_end::String`: Boundary condition at the end ("clamped", "simply supported", or "free").

# Returns
- `(grid, w, theta)`: A tuple containing the grid points `x`, the resulting deflection `w(x)`,
  and the rotation `θ(x)`.
"""
function solve_timoshenko_beam_dqm(
    N::Int,
    a::Real,
    b::Real,
    E::Real,
    G::Real,
    I_z::Vector,
    A::Vector,
    k_s::Real,
    load::Vector,
    bc_start::String,
    bc_end::String,
)

    # --- 1. System Setup ---
    A1, grid = DQM_weights(N, a, b)
    A2 = A1 * A1
    Id = I(N) # Identity matrix

    # --- Governing equations for a variable cross-section beam in matrix form ---
    kGA_vec = k_s * G .* A
    kGA_prime = A1 * kGA_vec
    EI_vec = E .* I_z
    EI_prime = A1 * EI_vec

    # Top block of the system matrix (Moment Equilibrium):
    # (EI*θ')' + kGA*(w' - θ) = 0
    LHS_11 = diagm(kGA_vec) * A1
    LHS_12 = diagm(EI_prime) * A1 + diagm(EI_vec) * A2 - diagm(kGA_vec) * Id

    # Bottom block of the system matrix (Force Equilibrium):
    # (kGA * (w' - θ))' = -q
    LHS_21 = diagm(kGA_prime) * A1 + diagm(kGA_vec) * A2
    LHS_22 = -diagm(kGA_prime) * Id - diagm(kGA_vec) * A1

    # Assemble the full 2N x 2N system matrix
    LHS = [
        LHS_11 LHS_12
        LHS_21 LHS_22
    ]

    # Initialize the load vector (RHS)
    RHS = zeros(2 * N)
    RHS[(N+1):(2*N)] = -load

    # --- 2. Apply Boundary Conditions ---
    # Apply boundary conditions at each end
    apply_timoshenko_bc!(LHS, RHS, A1, N, E, G, I_z, A, k_s, bc_start, "start")
    apply_timoshenko_bc!(LHS, RHS, A1, N, E, G, I_z, A, k_s, bc_end, "end")

    # --- 3. Solve and Return ---
    solution = LHS \ RHS

    # Extract deflection (w) and rotation (theta) from the solution vector
    w = solution[1:N]
    theta = solution[(N+1):(2*N)]

    return (grid, w, theta)
end


"""
    apply_timoshenko_bc!(LHS, RHS, A1, N, E, G, I_z, A, k_s, bc_type, position)

Helper function to apply Timoshenko beam boundary conditions.
"""
function apply_timoshenko_bc!(
    LHS::Matrix,
    RHS::Vector,
    A1::Matrix,
    N::Int,
    E::Real,
    G::Real,
    I_z::Vector,
    A::Vector,
    k_s::Real,
    bc_type::String,
    position::String,
)
    if position == "start"
        w_row, theta_row, deriv_row = 1, N + 1, 1
    elseif position == "end"
        w_row, theta_row, deriv_row = N, 2 * N, N
    else
        error("Position must be 'start' or 'end'.")
    end

    if bc_type == "clamped"
        # w = 0
        LHS[w_row, :] .= 0
        LHS[w_row, w_row] = 1
        RHS[w_row] = 0
        # θ = 0
        LHS[theta_row, :] .= 0
        LHS[theta_row, theta_row] = 1
        RHS[theta_row] = 0

    elseif bc_type == "simply supported"
        # w = 0
        LHS[w_row, :] .= 0
        LHS[w_row, w_row] = 1
        RHS[w_row] = 0
        # M = EI*θ' = 0
        moment_operator = diagm(E .* I_z) * A1
        LHS[theta_row, :] .= 0
        LHS[theta_row, (N+1):(2*N)] = moment_operator[deriv_row, :]
        RHS[theta_row] = 0

    elseif bc_type == "free"
        # V = kGA*(w' - θ) = 0
        shear_w_op = diagm(k_s * G .* A) * A1
        shear_theta_op = -diagm(k_s * G .* A)
        LHS[w_row, :] .= 0
        LHS[w_row, 1:N] = shear_w_op[deriv_row, :]
        LHS[w_row, (N+1):(2*N)] = shear_theta_op[deriv_row, :]
        RHS[w_row] = 0

        # M = EI*θ' = 0
        moment_operator = diagm(E .* I_z) * A1
        LHS[theta_row, :] .= 0
        LHS[theta_row, (N+1):(2*N)] = moment_operator[deriv_row, :]
        RHS[theta_row] = 0

    else
        error("Unknown boundary condition type: $bc_type")
    end
end