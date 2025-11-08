using LinearAlgebra
BLAS.set_num_threads(16)
# Include the necessary helper files
include("DQM_WEIGHTS.jl")
include("Smooth_load.jl") # Makes the Smooth_load function available here

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
    solve_beam_dqm( ...; load_F, ..., profile=false)

A convenience method for the DQM solver that generates a smoothed step load internally.
Set `profile=true` to enable performance profiling.
"""
function solve_beam_dqm(
    N::Int,
    a::Real,
    b::Real,
    E::Real,
    I_z,
    bc_start::String,
    bc_end::String,
    delta::Int;
    load_F::Real,
    load_start::Real,
    load_end::Real,
    load_steepness::Real,
    profile::Bool = false, # Pass profile flag through
)
    # This function is a user-friendly wrapper.
    _, grid = DQM_weights(N, a, b)
    load_vector = Smooth_load(grid, load_F, load_steepness, load_start, load_end)

    # Call the core solver, passing the profile flag to it
    return solve_beam_dqm(
        N,
        a,
        b,
        E,
        I_z,
        load_vector,
        bc_start,
        bc_end,
        delta;
        profile = profile,
    )
end

"""
    solve_beam_dqm(..., load::Vector, ...; profile=false)

This is the core DQM solver. Set `profile=true` to enable performance profiling.
"""
function solve_beam_dqm(
    N::Int,
    a::Real,
    b::Real,
    E::Real,
    I_z,
    load::Vector,
    bc_start::String,
    bc_end::String,
    delta::Int;
    profile::Bool = false, # Optional keyword for profiling
)
    # --- Profiling Setup ---
    timings = profile ? Dict{String,Float64}() : nothing

    # --- 1. System Setup ---
    # <--- FIX 1: Declare all needed variables in the function scope, just like Timoshenko
    local As, LHS, RHS, grid, A1
    @time_block profile timings "System Setup" begin
        # <--- FIX 2: Call DQM_weights ONCE and assign to outer-scope vars
        A1, grid = DQM_weights(N, a, b)
        As = (A1, A1^2, A1^3, A1^4)
        LHS = (As[2] * E * I_z) * As[2]
        RHS = load
    end

    # --- 2. Apply Boundary Conditions ---
    @time_block profile timings "Boundary Conditions" begin
        # This now correctly uses the `As`, `LHS`, `RHS` from the outer scope
        apply_boundary_condition!(LHS, RHS, As, bc_start, 1, 1 + delta)
        apply_boundary_condition!(LHS, RHS, As, bc_end, N, N - delta)
    end

    # --- 3. Solve and Return ---
    w = @time_block profile timings "Linear Solve" begin
        LHS \ RHS
    end

    # <--- FIX 3: REMOVED the redundant, bug-causing call to DQM_weights
    # grid, _ = DQM_weights(N, a, b) # <-- BUGGY LINE REMOVED

    if profile
        return (grid, w), timings # Returns the correct, consistent grid
    else
        return (grid, w) # Returns the correct, consistent grid
    end
end

"""
Helper function to apply a boundary condition to the LHS and RHS matrices.
"""
function apply_boundary_condition!(LHS, RHS, As, bc_type, node_idx, delta_node_idx)
    A1, A2, A3 = As[1], As[2], As[3]

    if bc_type == "clamped"
        # Displacement w = 0
        LHS[node_idx, :] .= 0;
        LHS[node_idx, node_idx] = 1;
        RHS[node_idx] = 0
        # Slope w' = 0
        LHS[delta_node_idx, :] = A1[delta_node_idx, :];
        RHS[delta_node_idx] = 0
    elseif bc_type == "simply supported"
        # Displacement w = 0
        LHS[node_idx, :] .= 0;
        LHS[node_idx, node_idx] = 1;
        RHS[node_idx] = 0
        # Bending Moment M ~ w'' = 0
        LHS[delta_node_idx, :] = A2[delta_node_idx, :];
        RHS[delta_node_idx] = 0
    elseif bc_type == "free"
        # Bending Moment M ~ w'' = 0
        LHS[node_idx, :] = A2[node_idx, :];
        RHS[node_idx] = 0
        # Shear Force V ~ w''' = 0
        LHS[delta_node_idx, :] = A3[delta_node_idx, :];
        RHS[delta_node_idx] = 0
    else
        error("Unknown boundary condition type: $bc_type")
    end
end