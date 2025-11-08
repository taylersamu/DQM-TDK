using LinearAlgebra

# It is assumed that a file named "DQM_WEIGHTS.jl" exists in the same directory.
include("DQM_WEIGHTS.jl")
BLAS.set_num_threads(16)

# --- Top-Level Profiling Macro ---
# Defined once, available to all functions in this file.
macro time_block(profile_flag, timings_dict, name, expr)
    return quote
        if $(esc(profile_flag))
            t0 = time_ns()
            result = $(esc(expr))
            $(esc(timings_dict))[$(esc(name))] = (time_ns() - t0) / 1e6 # Convert ns to ms
            result
        else
            $(esc(expr)) # Execute code without timing
        end
    end
end
"""
Solves the Kirchhoff-Love plate bending equation using the Differential Quadrature Method (DQM).
Set `profile=true` to enable performance profiling.
"""
function solve_kirchhoff_love_plate_dqm(
    N_x::Int,
    N_y::Int,
    a::Real,
    b::Real,
    E::Real,
    nu::Real,
    h::Real,
    load_matrix::Matrix,
    bc_config::Dict;
    profile::Bool = false, # Optional keyword for profiling
)
    # --- Profiling Setup ---
    timings = profile ? Dict{String,Float64}() : nothing

    # --- 1. System Setup ---
    local LHS, RHS, grid_x, grid_y
    @time_block profile timings "System Setup" begin
        A_x, grid_x = DQM_weights(N_x, 0, a)
        A_y, grid_y = DQM_weights(N_y, 0, b)

        I_x, I_y = I(N_x), I(N_y)
        B_x, D_x = A_x^2, A_x^4
        B_y, D_y = A_y^2, A_y^4

        D_rigidity = E * h^3 / (12 * (1 - nu^2))
        LHS = D_rigidity * (kron(D_x, I_y) + 2 * kron(B_x, B_y) + kron(I_x, D_y))
        RHS = reshape(load_matrix, N_x * N_y)
    end
    # These are needed for BCs and must be calculated outside the timed block
    A_x_bc, B_x_bc, C_x_bc = DQM_weights(N_x, 0, a)[1] |> x -> (x, x^2, x^3)
    A_y_bc, B_y_bc, C_y_bc = DQM_weights(N_y, 0, b)[1] |> y -> (y, y^2, y^3)

    # --- 2. Apply Boundary Conditions ---

    @time_block profile timings "Boundary Conditions" begin
        apply_kirchhoff_love_bc!(
            LHS,
            RHS,
            N_x,
            N_y,
            nu,
            bc_config,
            (A_x_bc, B_x_bc, C_x_bc),
            (A_y_bc, B_y_bc, C_y_bc),
        )
    end

    # --- 3. Solve and Return ---
    solution = @time_block profile timings "Linear Solve" begin
        LHS \ RHS
    end
    w = reshape(solution, N_y, N_x)

    if profile
        return (grid_x, grid_y, w), timings
    else
        return (grid_x, grid_y, w)
    end
end

"""
Applies boundary conditions to the  matrix (LHS) and load vector (RHS).
This function modifies the LHS and RHS matrices in place.
"""
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
    # This function's content is unchanged.
    A_x, B_x, C_x = deriv_ops_x
    A_y, B_y, C_y = deriv_ops_y
    delta = bc_config["delta"]
    # Kronecker Product Definitions
    Ax_op, Bx_op, Cx_op = kron(A_x, I(N_y)), kron(B_x, I(N_y)), kron(C_x, I(N_y))
    Ay_op, By_op, Cy_op = kron(I(N_x), A_y), kron(I(N_x), B_y), kron(I(N_x), C_y)
    Ayxx_op = kron(B_x, A_y)
    Axyy_op = kron(A_x, B_y)

    # Iterate through all grid points to apply BCs
    for j = 1:N_x, i = 1:N_y
        idx = (j - 1) * N_y + i
        is_on_d_boundary = (i == 1 || i == N_y || j == 1 || j == N_x)
        is_on_top_n_line = (i == N_y - delta)
        is_on_bottom_n_line = (i == 1 + delta)
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
        if is_on_d_boundary
            bc_type, edge = if j == 1
                ;
                (bc_config["left"], "vertical")
            elseif j == N_x
                ;
                (bc_config["right"], "vertical")
            elseif i == 1
                ;
                (bc_config["bottom"], "horizontal")
            elseif i == N_y
                ;
                (bc_config["top"], "horizontal")
            end

            if bc_type in ("C", "SS") # Deflection u_z = 0
                LHS[idx, :] .= 0;
                LHS[idx, idx] = 1;
                RHS[idx] = 0
            elseif bc_type == "F" # Normal Bending Moment = 0
                op = (edge == "vertical") ? (Bx_op + nu * By_op) : (By_op + nu * Bx_op)
                LHS[idx, :] = op[idx, :];
                RHS[idx] = 0
            end
        elseif is_on_n_boundary
            bc_type, edge = if is_on_left_n_line
                ;
                (bc_config["left"], "vertical")
            elseif is_on_right_n_line
                ;
                (bc_config["right"], "vertical")
            elseif is_on_bottom_n_line
                ;
                (bc_config["bottom"], "horizontal")
            elseif is_on_top_n_line
                ;
                (bc_config["top"], "horizontal")
            end

            op = nothing
            if edge == "horizontal"
                if bc_type == "C"
                    ;
                    op = Ay_op
                elseif bc_type == "SS"
                    ;
                    op = By_op + nu * Bx_op
                elseif bc_type == "F"
                    ;
                    op = Cy_op + (2 - nu) * Ayxx_op
                end
            elseif edge == "vertical"
                if bc_type == "C"
                    ;
                    op = Ax_op
                elseif bc_type == "SS"
                    ;
                    op = Bx_op + nu * By_op
                elseif bc_type == "F"
                    ;
                    op = Cx_op + (2 - nu) * Axyy_op
                end
            end
            if op !== nothing
                LHS[idx, :] = op[idx, :];
                RHS[idx] = 0
            end
        end
    end
end
