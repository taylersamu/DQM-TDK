using LinearAlgebra

# It is assumed that a file named "DQM_WEIGHTS.jl" exists and contains the
# DQM_weights function, which returns the weighting matrix and grid points.
include("DQM_WEIGHTS.jl")

"""
    solve_reissner_mindlin_plate_dqm(N_x, N_y, a, b, E, nu, h, k_s, load_matrix, bc_config)

Solves the static Reissner-Mindlin plate equations using the Differential Quadrature Method (DQM).

# Arguments
- `N_x::Int, N_y::Int`: Number of grid points along the x and y axes.
- `a::Real, b::Real`: Length of the plate along the x and y axes.
- `E::Real`: Young's modulus.
- `nu::Real`: Poisson's ratio.
- `h::Real`: Plate thickness.
- `k_s::Real`: Shear correction factor (e.g., 5/6).
- `load_matrix::Matrix`: A matrix representing the distributed load `q(x,y)` at each grid point.
- `bc_config::Dict`: A dictionary specifying the boundary conditions for each edge.
                     Example: Dict("top" => "F", "bottom" => "F", "left" => "SS", "right" => "C")

# Returns
- `(grid_x, grid_y, w, phi_x, phi_y)`: A tuple containing the grid points and the resulting
                                      deflection (w) and rotations (phi_x, phi_y).
"""
function solve_reissner_mindlin_plate_dqm(
    N_x::Int,
    N_y::Int,
    a::Real,
    b::Real,
    E::Real,
    nu::Real,
    h::Real,
    k_s::Real,
    load_matrix::Matrix,
    bc_config::Dict,
)

    # --- 1. System Setup ---
    A_normalx, grid_x = DQM_weights(N_x, 0, a)
    A_normaly, grid_y = DQM_weights(N_y, 0, b)

    I_x, I_y = I(N_x), I(N_y)
    A_x = kron(A_normalx, I_y)
    A_y = kron(I_x, A_normaly)
    B_x = kron(A_normalx^2, I_y)
    B_y = kron(I_x, A_normaly^2)

    # Material and section properties
    G = E / (2 * (1 + nu))
    D_b = E * h^3 / (12 * (1 - nu^2)) # Bending stiffness
    D_s = k_s * G * h                # Shear stiffness

    # --- Assemble the 3x3 block stiffness matrix (LHS) based on Equation (D.29) ---
    offset = N_x * N_y
    LHS = zeros(3 * offset, 3 * offset)
    I_full = I(offset)

    # Row 1: Force equilibrium
    LHS[1:offset, 1:offset] = -D_s * (B_x + B_y)
    LHS[1:offset, (offset+1):(2*offset)] = D_s * A_y
    LHS[1:offset, (2*offset+1):(3*offset)] = -D_s * A_x

    # Row 2: Moment equilibrium about y-axis (for phi_x)
    LHS[(offset+1):(2*offset), 1:offset] = D_s * A_y
    LHS[(offset+1):(2*offset), (offset+1):(2*offset)] =
        D_b * (((1 - nu) / 2) * B_x + B_y) - D_s * I_full
    LHS[(offset+1):(2*offset), (2*offset+1):(3*offset)] = -D_b * ((1 + nu) / 2) * A_y * A_x

    # Row 3: Moment equilibrium about x-axis (for phi_y)
    LHS[(2*offset+1):(3*offset), 1:offset] = -D_s * A_x
    LHS[(2*offset+1):(3*offset), (offset+1):(2*offset)] = -D_b * ((1 + nu) / 2) * A_x * A_y
    LHS[(2*offset+1):(3*offset), (2*offset+1):(3*offset)] =
        D_b * (B_x + ((1 - nu) / 2) * B_y) - D_s * I_full

    # --- Assemble Load Vector (RHS) ---
    RHS = zeros(3 * offset)
    RHS[1:offset] = -reshape(load_matrix, offset) # Corresponds to -q_z

    # --- 2. Apply Boundary Conditions ---
    apply_reissner_mindlin_bc!(LHS, RHS, N_x, N_y, bc_config, (A_x, A_y))

    # --- 3. Solve and Return ---
    solution = LHS \ RHS
    w = reshape(solution[1:offset], N_y, N_x)
    phi_x = reshape(solution[(offset+1):(2*offset)], N_y, N_x)
    phi_y = reshape(solution[(2*offset+1):(3*offset)], N_y, N_x)

    return (grid_x, grid_y, w, phi_x, phi_y)
end


"""
    apply_reissner_mindlin_bc!(LHS, RHS, N_x, N_y, bc_config, deriv_ops)

Helper function to apply Reissner-Mindlin plate boundary conditions in place.
"""
function apply_reissner_mindlin_bc!(LHS, RHS, N_x, N_y, bc_config, deriv_ops)
    A_x, A_y = deriv_ops
    offset = N_x * N_y

    for j = 1:N_x, i = 1:N_y
        # Only apply BCs on the boundary nodes
        is_boundary = (i == 1 || i == N_y || j == 1 || j == N_x)
        if !is_boundary
            continue
        end

        idx = (j - 1) * N_y + i

        # Determine which edge the point is on
        bc_type = ""
        if i == 1
            ;
            bc_type = bc_config["top"];
        end
        if i == N_y
            ;
            bc_type = bc_config["bottom"];
        end
        if j == 1
            ;
            bc_type = bc_config["left"];
        end
        if j == N_x
            ;
            bc_type = bc_config["right"];
        end

        # Define indices for w, phi_x, and phi_y rows
        w_row, phi_x_row, phi_y_row = idx, idx + offset, idx + 2 * offset

        if bc_type == "C" # Clamped
            # w = 0, phi_x = 0, phi_y = 0
            LHS[w_row, :] .= 0;
            LHS[w_row, w_row] = 1;
            RHS[w_row] = 0
            LHS[phi_x_row, :] .= 0;
            LHS[phi_x_row, phi_x_row] = 1;
            RHS[phi_x_row] = 0
            LHS[phi_y_row, :] .= 0;
            LHS[phi_y_row, phi_y_row] = 1;
            RHS[phi_y_row] = 0

        elseif bc_type == "SS" # Simply Supported
            # w = 0
            LHS[w_row, :] .= 0;
            LHS[w_row, w_row] = 1;
            RHS[w_row] = 0

            # On vertical edges (left/right): M_x = 0, phi_y = 0
            if j == 1 || j == N_x
                LHS[phi_y_row, :] .= 0;
                LHS[phi_y_row, phi_y_row] = 1;
                RHS[phi_y_row] = 0
                LHS[phi_x_row, :] .= 0
                LHS[phi_x_row, (offset+1):(2*offset)] = A_y[idx, :] # d(phi_x)/dy
                LHS[phi_x_row, (2*offset+1):(3*offset)] = nu * A_x[idx, :] # nu*d(phi_y)/dx
                RHS[phi_x_row] = 0
                # On horizontal edges (top/bottom): M_y = 0, phi_x = 0
            else
                LHS[phi_x_row, :] .= 0;
                LHS[phi_x_row, phi_x_row] = 1;
                RHS[phi_x_row] = 0
                LHS[phi_y_row, :] .= 0
                LHS[phi_y_row, (offset+1):(2*offset)] = nu * A_y[idx, :] # nu*d(phi_x)/dy
                LHS[phi_y_row, (2*offset+1):(3*offset)] = A_x[idx, :] # d(phi_y)/dx
                RHS[phi_y_row] = 0
            end

        elseif bc_type == "F" # Free
            # Shear force Q_n = 0, Moment M_n = 0, Twisting Moment M_ns = 0
            # On vertical edges (left/right): Q_x = 0, M_x = 0, M_xy = 0
            if j == 1 || j == N_x
                # Q_x = D_s * (du_z/dx - phi_y) = 0
                LHS[w_row, :] .= 0;
                LHS[w_row, 1:offset] = A_x[idx, :];
                LHS[w_row, phi_y_row] = -1;
                RHS[w_row] = 0
                # M_x = D_b * (d(phi_y)/dx + nu*d(phi_x)/dy) = 0
                LHS[phi_x_row, :] .= 0;
                LHS[phi_x_row, (offset+1):(2*offset)] = nu * A_y[idx, :];
                LHS[phi_x_row, (2*offset+1):(3*offset)] = A_x[idx, :];
                RHS[phi_x_row] = 0
                # M_xy = D_b*(1-nu)/2 * (d(phi_y)/dy - d(phi_x)/dx) = 0
                LHS[phi_y_row, :] .= 0;
                LHS[phi_y_row, (offset+1):(2*offset)] = -A_x[idx, :];
                LHS[phi_y_row, (2*offset+1):(3*offset)] = A_y[idx, :];
                RHS[phi_y_row] = 0
                # On horizontal edges (top/bottom): Q_y = 0, M_y = 0, M_xy = 0
            else
                # Q_y = D_s * (du_z/dy + phi_x) = 0
                LHS[w_row, :] .= 0;
                LHS[w_row, 1:offset] = A_y[idx, :];
                LHS[w_row, phi_x_row] = 1;
                RHS[w_row] = 0
                # M_y = D_b * (d(phi_x)/dy + nu*d(phi_y)/dx) = 0
                LHS[phi_x_row, :] .= 0;
                LHS[phi_x_row, (offset+1):(2*offset)] = A_y[idx, :];
                LHS[phi_x_row, (2*offset+1):(3*offset)] = nu * A_x[idx, :];
                RHS[phi_x_row] = 0
                # M_xy = D_b*(1-nu)/2 * (d(phi_y)/dy - d(phi_x)/dx) = 0
                LHS[phi_y_row, :] .= 0;
                LHS[phi_y_row, (offset+1):(2*offset)] = -A_x[idx, :];
                LHS[phi_y_row, (2*offset+1):(3*offset)] = A_y[idx, :];
                RHS[phi_y_row] = 0
            end
        end
    end
end
