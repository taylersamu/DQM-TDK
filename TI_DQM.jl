using LinearAlgebra
include("DQM_WEIGHTS.jl")

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
    solve_timoshenko_beam_dqm(..., profile=false)

Solves the static Timoshenko beam equations using DQM.
Set `profile=true` to enable performance profiling.
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
    bc_end::String;
    profile::Bool = false, # Optional keyword for profiling
)
    # --- Profiling Setup ---
    timings = profile ? Dict{String,Float64}() : nothing

    # --- 1. System Setup ---
    local LHS, RHS, grid, A1
    @time_block profile timings "System Setup" begin
        A1, grid = DQM_weights(N, a, b)
        A2 = A1 * A1
        I_x = I(N)

        LHS_11 = k_s * G * A * A2
        LHS_12 = -k_s * G * A * A1
        LHS_21 = k_s * G * A * A1
        LHS_22 = E * I_z * A2 - k_s * G * A * I_x

        LHS = [LHS_11 LHS_12; LHS_21 LHS_22]

        RHS = zeros(2 * N)
        RHS[1:N] = -load
    end

    # --- 2. Apply Boundary Conditions ---
    @time_block profile timings "Boundary Conditions" begin
        apply_timoshenko_bc!(LHS, RHS, A1, N, bc_start, "start")
        apply_timoshenko_bc!(LHS, RHS, A1, N, bc_end, "end")
    end

    # --- 3. Solve and Return ---
    solution = @time_block profile timings "Linear Solve" begin
        LHS \ RHS
    end

    w = solution[1:N]
    theta = solution[(N+1):(2*N)]

    if profile
        return (grid, w, theta), timings
    else
        return (grid, w, theta)
    end
end

"""
    apply_timoshenko_bc!(LHS, RHS, A1, N, bc_type, position)

Helper function to apply Timoshenko beam boundary conditions.
"""
function apply_timoshenko_bc!(LHS, RHS, A1, N, bc_type, position)
    w_row, theta_row, deriv_row = if position == "start"
        (1, N + 1, 1)
    elseif position == "end"
        (N, 2 * N, N)
    else
        error("Position must be 'start' or 'end'.")
    end

    if bc_type == "clamped"
        # Deflection w = 0
        LHS[w_row, :] .= 0;
        LHS[w_row, w_row] = 1;
        RHS[w_row] = 0
        # Rotation θ = 0
        LHS[theta_row, :] .= 0;
        LHS[theta_row, theta_row] = 1;
        RHS[theta_row] = 0
    elseif bc_type == "simply supported"
        # Deflection w = 0
        LHS[w_row, :] .= 0;
        LHS[w_row, w_row] = 1;
        RHS[w_row] = 0
        # Bending Moment M = E*I*θ' = 0  =>  θ' = 0
        LHS[theta_row, :] .= 0;
        LHS[theta_row, (N+1):(2*N)] = A1[deriv_row, :];
        RHS[theta_row] = 0
    elseif bc_type == "free"
        # Shear Force V = k*G*A*(w' - θ) = 0  =>  w' = θ
        LHS[w_row, :] .= 0;
        LHS[w_row, 1:N] = A1[deriv_row, :];
        LHS[w_row, N+w_row] = -1;
        RHS[w_row] = 0
        # Bending Moment M = E*I*θ' = 0  =>  θ' = 0
        LHS[theta_row, :] .= 0;
        LHS[theta_row, (N+1):(2*N)] = A1[deriv_row, :];
        RHS[theta_row] = 0
    else
        error("Unknown boundary condition type: $bc_type")
    end
end
