using LinearAlgebra
using SparseArrays # Library for sparse matrix operations

# It is assumed that a file named "DQM_WEIGHTS.jl" exists and contains the
# DQM_weights function, which returns the weighting matrix and grid points.
include("DQM_WEIGHTS.jl")

# --- Top-Level Profiling Macro ---
# This macro conditionally times a block of code.
macro time_block(profile_flag, timings_dict, name, expr)
    return quote
        if $(esc(profile_flag))
            t0 = time_ns()
            result = $(esc(expr))
            # Store elapsed time in the provided dictionary in milliseconds
            $(esc(timings_dict))[$(esc(name))] = (time_ns() - t0) / 1e6 
            result
        else
            $(esc(expr)) # Execute code without timing if profiling is off
        end
    end
end


"""
    solve_reissner_mindlin_plate_dqm(..., profile=false)

Solves the static Reissner-Mindlin plate equations using the Differential Quadrature Method (DQM).
Set `profile=true` to enable performance benchmarking.
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
    bc_config::Dict;
    profile::Bool = false, # Optional keyword to enable profiling
)

    # --- Profiling Setup ---
    timings = profile ? Dict{String,Float64}() : nothing

    # Declare variables that will be assigned inside the timed block
    local LHS, RHS, grid_x, grid_y, A_x, A_y, D_b, D_s

    # --- 1. System Setup ---
    @time_block profile timings "System Setup" begin
        A_normalx, grid_x = DQM_weights(N_x, 0, a)
        A_normaly, grid_y = DQM_weights(N_y, 0, b)

        I_x, I_y = sparse(I(N_x)), sparse(I(N_y))

        # Kronecker products for column-major vectorization
        A_x = kron(A_normalx, I_y)
        A_y = kron(I_x, A_normaly)
        B_x = kron(A_normalx^2, I_y)
        B_y = kron(I_x, A_normaly^2)
        A_xy = A_x * A_y # Mixed derivative operator

        # Material and section properties
        G = E / (2 * (1 + nu))
        D_b = E * h^3 / (12 * (1 - nu^2)) # Bending stiffness
        D_s = k_s * G * h               # Shear stiffness

        # --- Assemble the 3x3 block stiffness matrix (LHS) ---
        offset = N_x * N_y
        LHS = zeros(3 * offset, 3 * offset)
        I_full = I(offset)

        # Block K_1j (Row 1: Force equilibrium in z-direction)
        LHS[1:offset, 1:offset] = -D_s * (B_x + B_y)
        LHS[1:offset, (offset+1):(2*offset)] = D_s * A_y
        LHS[1:offset, (2*offset+1):(3*offset)] = -D_s * A_x

        # Block K_2j (Row 2: Moment equilibrium about y-axis for phi_x)
        LHS[(offset+1):(2*offset), 1:offset] = D_s * A_y
        LHS[(offset+1):(2*offset), (offset+1):(2*offset)] =
            D_b * (((1 - nu) / 2) * B_x + B_y) - D_s * I_full
        LHS[(offset+1):(2*offset), (2*offset+1):(3*offset)] = -D_b * ((1 + nu) / 2) * A_xy

        # Block K_3j (Row 3: Moment equilibrium about x-axis for phi_y)
        LHS[(2*offset+1):(3*offset), 1:offset] = -D_s * A_x
        LHS[(2*offset+1):(3*offset), (offset+1):(2*offset)] = -D_b * ((1 + nu) / 2) * A_xy
        LHS[(2*offset+1):(3*offset), (2*offset+1):(3*offset)] =
            D_b * (B_x + ((1 - nu) / 2) * B_y) - D_s * I_full

        # --- Assemble Load Vector (RHS) ---
        RHS = zeros(3 * offset)
        RHS[1:offset] = reshape(load_matrix, offset)
    end

    # --- 2. Apply Boundary Conditions ---
    @time_block profile timings "Boundary Conditions" begin
        apply_reissner_mindlin_bc!(LHS, RHS, N_x, N_y, nu, D_s, bc_config, (A_x, A_y, D_b))
    end

    # --- 3. Solve and Return ---
    solution = @time_block profile timings "Linear Solve" begin
        LHS \ RHS
    end

    offset = N_x * N_y
    w = reshape(solution[1:offset], N_y, N_x)
    phi_x = reshape(solution[(offset+1):(2*offset)], N_y, N_x)
    phi_y = reshape(solution[(2*offset+1):(3*offset)], N_y, N_x)
    
    results = (grid_x, grid_y, w, phi_x, phi_y)

    if profile
        return results, timings
    else
        return results
    end
end

"""
    apply_reissner_mindlin_bc!(LHS, RHS, N_x, N_y, nu, D_s, bc_config, ops)

Applies boundary conditions for a Reissner-Mindlin plate in place. This function
modifies the global Left-Hand-Side (LHS) matrix and Right-Hand-Side (RHS) vector
to enforce the specified boundary conditions.

# Arguments
- `LHS`: The sparse global stiffness matrix.
- `RHS`: The global load vector.
- `N_x`, `N_y`: Number of grid points in the x and y directions.
- `nu`: Poisson's ratio.
- `D_s`: Shear stiffness of the plate (k_s * G * h).
- `bc_config`: A dictionary specifying the boundary type ("SS", "C", "F") for each edge ("top", "bottom", "left", "right").
- `ops`: A tuple containing the DQM operators and bending stiffness `(A_x, A_y, D_b)`.
"""
function apply_reissner_mindlin_bc!(LHS, RHS, N_x, N_y, nu, D_s, bc_config, ops)
    A_x, A_y, D_b = ops
    offset = N_x * N_y
    I_offset = sparse(I(offset))

    for j = 1:N_x, i = 1:N_y
        idx = (j - 1) * N_y + i
        w_row, phi_x_row, phi_y_row = idx, idx + offset, idx + 2 * offset

        # --- Boundary Condition Checks ---
        is_bottom = (i == 1)
        is_top = (i == N_y)
        is_left = (j == 1)
        is_right = (j == N_x)

        if !(is_bottom || is_top || is_left || is_right)
            continue # Skip interior points
        end

        # --- Apply Conditions Based on Edge ---
        if is_bottom
            bc_type = bc_config["bottom"]
            if bc_type == "SS"
                # Condition 1: w = 0
                LHS[w_row, :] .= 0; LHS[w_row, w_row] = 1; RHS[w_row] = 0
                # Condition 2: Tangential rotation phi_y = 0
                LHS[phi_y_row, :] .= 0; LHS[phi_y_row, phi_y_row] = 1; RHS[phi_y_row] = 0
                # Condition 3: Normal moment M_yy = 0
                LHS[phi_x_row, :] .= 0
                LHS[phi_x_row, (offset+1):(2*offset)] = D_b * A_y[idx, :]
                LHS[phi_x_row, (2*offset+1):(3*offset)] = D_b * nu * A_x[idx, :]
                RHS[phi_x_row] = 0
            elseif bc_type == "C" # Clamped
                # Conditions: w = 0, phi_x = 0, phi_y = 0
                LHS[w_row, :] .= 0; LHS[w_row, w_row] = 1; RHS[w_row] = 0
                LHS[phi_x_row, :] .= 0; LHS[phi_x_row, phi_x_row] = 1; RHS[phi_x_row] = 0
                LHS[phi_y_row, :] .= 0; LHS[phi_y_row, phi_y_row] = 1; RHS[phi_y_row] = 0
            elseif bc_type == "F" # Free
                # Condition 1: Shear force Q_y = D_s * (dw/dy + phi_y) = 0
                LHS[w_row, :] .= 0
                LHS[w_row, 1:offset] = D_s * A_y[idx, :]
                LHS[w_row, (offset+1):(2*offset)] = D_s * I_offset[idx, :]
                RHS[w_row] = 0
                # Condition 2: Normal moment M_yy = 0
                LHS[phi_x_row, :] .= 0
                LHS[phi_x_row, (offset+1):(2*offset)] = D_b * A_y[idx, :]
                LHS[phi_x_row, (2*offset+1):(3*offset)] = D_b * nu * A_x[idx, :]
                RHS[phi_x_row] = 0
                # Condition 3: Twisting moment M_yx = 0
                LHS[phi_y_row, :] .= 0
                LHS[phi_y_row, (offset+1):(2*offset)] = D_b * ((1 - nu) / 2) * A_x[idx, :]
                LHS[phi_y_row, (2*offset+1):(3*offset)] = D_b * ((1 - nu) / 2) * A_y[idx, :]
                RHS[phi_y_row] = 0
            end
        end

        if is_top
            bc_type = bc_config["top"]
            if bc_type == "SS"
                LHS[w_row, :] .= 0; LHS[w_row, w_row] = 1; RHS[w_row] = 0
                LHS[phi_y_row, :] .= 0; LHS[phi_y_row, phi_y_row] = 1; RHS[phi_y_row] = 0
                LHS[phi_x_row, :] .= 0
                LHS[phi_x_row, (offset+1):(2*offset)] = D_b * A_y[idx, :]
                LHS[phi_x_row, (2*offset+1):(3*offset)] = D_b * nu * A_x[idx, :]
                RHS[phi_x_row] = 0
            elseif bc_type == "C"
                LHS[w_row, :] .= 0; LHS[w_row, w_row] = 1; RHS[w_row] = 0
                LHS[phi_x_row, :] .= 0; LHS[phi_x_row, phi_x_row] = 1; RHS[phi_x_row] = 0
                LHS[phi_y_row, :] .= 0; LHS[phi_y_row, phi_y_row] = 1; RHS[phi_y_row] = 0
            elseif bc_type == "F"
                LHS[w_row, :] .= 0
                LHS[w_row, 1:offset] = D_s * A_y[idx, :]
                LHS[w_row, (offset+1):(2*offset)] = D_s * I_offset[idx, :]
                RHS[w_row] = 0
                LHS[phi_x_row, :] .= 0
                LHS[phi_x_row, (offset+1):(2*offset)] = D_b * A_y[idx, :]
                LHS[phi_x_row, (2*offset+1):(3*offset)] = D_b * nu * A_x[idx, :]
                RHS[phi_x_row] = 0
                LHS[phi_y_row, :] .= 0
                LHS[phi_y_row, (offset+1):(2*offset)] = D_b * ((1 - nu) / 2) * A_x[idx, :]
                LHS[phi_y_row, (2*offset+1):(3*offset)] = D_b * ((1 - nu) / 2) * A_y[idx, :]
                RHS[phi_y_row] = 0
            end
        end

        if is_left
            bc_type = bc_config["left"]
            if bc_type == "SS"
                LHS[w_row, :] .= 0; LHS[w_row, w_row] = 1; RHS[w_row] = 0
                # Condition 2: Tangential rotation phi_x = 0
                LHS[phi_x_row, :] .= 0; LHS[phi_x_row, phi_x_row] = 1; RHS[phi_x_row] = 0
                # Condition 3: Normal moment M_xx = 0
                LHS[phi_y_row, :] .= 0
                LHS[phi_y_row, (offset+1):(2*offset)] = D_b * nu * A_y[idx, :]
                LHS[phi_y_row, (2*offset+1):(3*offset)] = D_b * A_x[idx, :]
                RHS[phi_y_row] = 0
            elseif bc_type == "C"
                LHS[w_row, :] .= 0; LHS[w_row, w_row] = 1; RHS[w_row] = 0
                LHS[phi_x_row, :] .= 0; LHS[phi_x_row, phi_x_row] = 1; RHS[phi_x_row] = 0
                LHS[phi_y_row, :] .= 0; LHS[phi_y_row, phi_y_row] = 1; RHS[phi_y_row] = 0
            elseif bc_type == "F"
                # Condition 1: Shear force Q_x = D_s * (dw/dx + phi_x) = 0
                LHS[w_row, :] .= 0
                LHS[w_row, 1:offset] = D_s * A_x[idx, :]
                LHS[w_row, (2*offset+1):(3*offset)] = D_s * I_offset[idx, :]
                RHS[w_row] = 0
                # Condition 2: Normal moment M_xx = 0
                LHS[phi_y_row, :] .= 0
                LHS[phi_y_row, (offset+1):(2*offset)] = D_b * nu * A_y[idx, :]
                LHS[phi_y_row, (2*offset+1):(3*offset)] = D_b * A_x[idx, :]
                RHS[phi_y_row] = 0
                # Condition 3: Twisting moment M_xy = 0
                LHS[phi_x_row, :] .= 0
                LHS[phi_x_row, (offset+1):(2*offset)] = D_b * ((1 - nu) / 2) * A_y[idx, :]
                LHS[phi_x_row, (2*offset+1):(3*offset)] = D_b * ((1 - nu) / 2) * A_x[idx, :]
                RHS[phi_x_row] = 0
            end
        end

        if is_right
            bc_type = bc_config["right"]
            if bc_type == "SS"
                LHS[w_row, :] .= 0; LHS[w_row, w_row] = 1; RHS[w_row] = 0
                LHS[phi_x_row, :] .= 0; LHS[phi_x_row, phi_x_row] = 1; RHS[phi_x_row] = 0
                LHS[phi_y_row, :] .= 0
                LHS[phi_y_row, (offset+1):(2*offset)] = D_b * nu * A_y[idx, :]
                LHS[phi_y_row, (2*offset+1):(3*offset)] = D_b * A_x[idx, :]
                RHS[phi_y_row] = 0
            elseif bc_type == "C"
                LHS[w_row, :] .= 0; LHS[w_row, w_row] = 1; RHS[w_row] = 0
                LHS[phi_x_row, :] .= 0; LHS[phi_x_row, phi_x_row] = 1; RHS[phi_x_row] = 0
                LHS[phi_y_row, :] .= 0; LHS[phi_y_row, phi_y_row] = 1; RHS[phi_y_row] = 0
            elseif bc_type == "F"
                LHS[w_row, :] .= 0
                LHS[w_row, 1:offset] = D_s * A_x[idx, :]
                LHS[w_row, (2*offset+1):(3*offset)] = D_s * I_offset[idx, :]
                RHS[w_row] = 0
                LHS[phi_y_row, :] .= 0
                LHS[phi_y_row, (offset+1):(2*offset)] = D_b * nu * A_y[idx, :]
                LHS[phi_y_row, (2*offset+1):(3*offset)] = D_b * A_x[idx, :]
                RHS[phi_y_row] = 0
                LHS[phi_x_row, :] .= 0
                LHS[phi_x_row, (offset+1):(2*offset)] = D_b * ((1 - nu) / 2) * A_y[idx, :]
                LHS[phi_x_row, (2*offset+1):(3*offset)] = D_b * ((1 - nu) / 2) * A_x[idx, :]
                RHS[phi_x_row] = 0
            end
        end
    end
end