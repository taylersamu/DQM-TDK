# BE_DQM.jl

using LinearAlgebra
BLAS.set_num_threads(16)
# Include the necessary helper files
include("DQM_WEIGHTS.jl")
include("Smooth_load.jl") # Makes the Smooth_load function available here

"""
    solve_beam_dqm( ...; load_F, load_start, load_end, load_steepness)

A convenience method for the DQM solver that generates a smoothed step load internally.
The load parameters are provided as **keyword arguments**.

# Keyword Arguments
- `load_F::Real`: The total integrated force to be applied (e.g., -100.0).
- `load_start::Real`: The coordinate where the load begins.
- `load_end::Real`: The coordinate where the load ends.
- `load_steepness::Real`: A factor `s` controlling how sharp the load transition is.
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
)
    # This function is a user-friendly wrapper.

    # 1. Generate the grid, which is needed to create the load vector.
    _, grid = DQM_weights(N, a, b)

    # 2. Call the Smooth_load function (from the included file) to create the load vector.
    load_vector = Smooth_load(grid, load_F, load_steepness, load_start, load_end)

    # 3. Call the core "worker" solver below, passing it the generated load vector.
    return solve_beam_dqm(N, a, b, E, I_z, load_vector, bc_start, bc_end, delta)
end


"""
    solve_beam_dqm(..., load::Vector, ...)

This is the core DQM solver. It performs the calculation once the load profile
has been provided as a vector. It's called by the convenience function above.
"""
function solve_beam_dqm(
    N::Int,
    a::Real,
    b::Real,
    E::Real,
    I_z,
    load::Vector, # Accepts a pre-made load vector
    bc_start::String,
    bc_end::String,
    delta::Int,
)
    # --- 1. System Setup ---
    A1, grid = DQM_weights(N, a, b)
    As = (A1, A1^2, A1^3, A1^4)
    LHS = (As[2]*E * I_z) * As[2]
    RHS = load

    # --- 2. Apply Boundary Conditions ---
    apply_boundary_condition!(LHS, RHS, As, bc_start, 1, 1 + delta)
    apply_boundary_condition!(LHS, RHS, As, bc_end, N, N - delta)

    # --- 3. Solve and Return ---
    w = LHS \ RHS
    return (grid, w)
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
