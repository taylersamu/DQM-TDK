using LinearAlgebra

# It is assumed that a file named "DQM_WEIGHTS.jl" exists in the same directory.
# This file should contain the `DQM_weights(N, a, b)` function, which returns
# the first-order DQM weighting matrix and the corresponding grid points.
include("DQM_WEIGHTS.jl")
BLAS.set_num_threads(16)


function solve_kirchhoff_love_plate_dqm(
    N_x::Int,
    N_y::Int,
    a::Real,
    b::Real,
    E::Real,
    nu::Real,
    h::Real,
    load_matrix::Matrix,
    bc_config::Dict,
)

    # --- 1. System Setup ---
    A_x, grid_x = DQM_weights(N_x, 0, a)
    A_y, grid_y = DQM_weights(N_y, 0, b)

    I_x = I(N_x)
    I_y = I(N_y)

    # Higher-order derivative matrices
    B_x, C_x, D_x = A_x^2, A_x^3, A_x^4
    B_y, C_y, D_y = A_y^2, A_y^3, A_y^4

    # Plate bending rigidity
    D = E * h^3 / (12 * (1 - nu^2))

    # Stiffness matrix (LHS) based on the governing equation (for column-major vectorization)
    LHS = D * (kron(D_x, I_y) + 2 * kron(B_x, B_y) + kron(I_x, D_y))

    # Load vector (RHS)
    RHS = reshape(load_matrix, N_x * N_y)

    # --- 2. Apply Boundary Conditions ---
    apply_kirchhoff_love_bc!(
        LHS,
        RHS,
        N_x,
        N_y,
        nu,
        bc_config,
        (A_x, B_x, C_x),
        (A_y, B_y, C_y),
    )

    # --- 3. Solve and Return ---
    solution = LHS \ RHS
    w = reshape(solution, N_y, N_x)

    return (grid_x, grid_y, w)
end


function apply_kirchhoff_love_bc!(
    LHS,
    RHS,
    N_x,
    N_y,
    nu,
    bc_config,
    deriv_ops_x,
    deriv_ops_y,
)
    A_x, B_x, C_x = deriv_ops_x
    A_y, B_y, C_y = deriv_ops_y
    delta = bc_config["delta"]

    # Create Kronecker products for boundary operators (for column-major vectorization)
    # Note: kron(A_x, I_y) is the y-derivative operator, kron(I_x, A_y) is the x-derivative operator
    Ax_op, Bx_op, Cx_op = kron(I(N_x), A_y), kron(I(N_x), B_y), kron(I(N_x), C_y)
    Ay_op, By_op, Cy_op = kron(A_x, I(N_y)), kron(B_x, I(N_y)), kron(C_x, I(N_y))
    # Mixed derivative operators
    Axyy_op = kron(B_x, A_y) # d3/(dx dy^2)
    Ayxx_op = kron(A_x, B_y) # d3/(dy dx^2)

    for j = 1:N_x, i = 1:N_y
        idx = (j - 1) * N_y + i

        # --- Define Boundary Regions ---
        is_on_d_boundary = (i == 1 || i == N_y || j == 1 || j == N_x)

        is_on_top_n_line = (i == 1 + delta)
        is_on_bottom_n_line = (i == N_y - delta)
        is_on_left_n_line = (j == 1 + delta)
        is_on_right_n_line = (j == N_x - delta)

        is_n_corner =
            (is_on_top_n_line || is_on_bottom_n_line) &&
            (is_on_left_n_line || is_on_right_n_line)

        is_on_n_boundary =
            (
                is_on_top_n_line ||
                is_on_bottom_n_line ||
                is_on_left_n_line ||
                is_on_right_n_line
            ) && !is_n_corner

        # --- Apply Conditions ---
        if is_on_d_boundary
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

            if bc_type == "C" || bc_type == "SS"
                LHS[idx, :] .= 0
                LHS[idx, idx] = 1
                RHS[idx] = 0
            elseif bc_type == "F"
                op = nothing
                if j == 1 || j == N_x # Vertical Edge: M_x = 0
                    op = Bx_op + nu * By_op
                end
                if i == 1 || i == N_y # Horizontal Edge: M_y = 0
                    op = By_op + nu * Bx_op
                end
                LHS[idx, :] = op[idx, :]
                RHS[idx] = 0
            end
        elseif is_on_n_boundary
            bc_type = ""
            if is_on_top_n_line
                ;
                bc_type = bc_config["top"];
            end
            if is_on_bottom_n_line
                ;
                bc_type = bc_config["bottom"];
            end
            if is_on_left_n_line
                ;
                bc_type = bc_config["left"];
            end
            if is_on_right_n_line
                ;
                bc_type = bc_config["right"];
            end

            op = nothing
            if is_on_top_n_line || is_on_bottom_n_line # Horizontal edges
                if bc_type == "C"
                    ;
                    op = Ay_op;
                end
                if bc_type == "SS"
                    ;
                    op = By_op + nu * Bx_op;
                end
                if bc_type == "F"
                    ;
                    op = Cy_op + (2 - nu) * Ayxx_op;
                end
            else # Vertical edges
                if bc_type == "C"
                    ;
                    op = Ax_op;
                end
                if bc_type == "SS"
                    ;
                    op = Bx_op + nu * By_op;
                end
                if bc_type == "F"
                    ;
                    op = Cx_op + (2 - nu) * Axyy_op;
                end
            end
            LHS[idx, :] = op[idx, :]
            RHS[idx] = 0
        end
    end
end