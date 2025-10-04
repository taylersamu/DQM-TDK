# In file: BE_DQM_varying_cross.jl

using LinearAlgebra

# Wrapper function remains the same, but we will call the core function directly
# for this example to ensure clarity.

"""
Solves the static Euler-Bernoulli beam equation for a beam with a variable
cross-section, `(EI(x)w''(x))'' = q(x)`, using the Differential Quadrature Method.
It performs the calculation once the load profile has been provided as a vector.
"""
function solve_beam_dqm(
    N::Int,
    a::Real,
    b::Real,
    E::Real,
    I_z::Vector, # I_z must be a vector, even if constant
    load::Vector,
    bc_start::String,
    bc_end::String,
    delta::Int,  # Index offset for applying derivative boundary conditions
)
    # --- 1. System Setup ---
    A1, grid = DQM_weights(N, a, b)
    As = (A1, A1^2, A1^3, A1^4)

    # Governing equation: (EI)w'''' + 2(EI)'w''' + (EI)''w'' = q
    EI_vec = E .* I_z
    EI_prime = As[1] * EI_vec      # (EI)'
    EI_prime_prime = As[2] * EI_vec # (EI)''
    LHS =
        diagm(EI_prime_prime) * As[2] + 2 * diagm(EI_prime) * As[3] + diagm(EI_vec) * As[4]

    RHS = load

    # --- 2. Apply Boundary Conditions ---
    # Apply boundary conditions at the start and end of the beam.
    apply_boundary_condition!(LHS, RHS, As, E, I_z, bc_start, 1, 1 + delta)
    apply_boundary_condition!(LHS, RHS, As, E, I_z, bc_end, N, N - delta)

    # --- 3. Solve and Return ---
    w = LHS \ RHS
    return (grid, w)
end


"""
Helper function to apply a boundary condition to the LHS and RHS matrices.
"""
function apply_boundary_condition!(
    LHS::Matrix,
    RHS::Vector,
    As::Tuple,
    E::Real,
    I_z::Vector,
    bc_type::String,
    node_idx::Int,
    delta_node_idx::Int,
)
    A1, A2, A3 = As[1], As[2], As[3]
    N = size(LHS, 1)

    if bc_type == "clamped"
        # Deflection w = 0 at the boundary node.
        LHS[node_idx, :] .= 0
        LHS[node_idx, node_idx] = 1
        RHS[node_idx] = 0
        # Slope w' = 0 at the delta-node (adjacent to boundary).
        LHS[delta_node_idx, :] = A1[delta_node_idx, :]
        RHS[delta_node_idx] = 0

    elseif bc_type == "free"
        # Use the general physics for a variable cross-section beam.

        # Bending Moment M = EIw'' = 0 at the boundary node.
        moment_operator = diagm(E .* I_z) * A2
        LHS[node_idx, :] = moment_operator[node_idx, :]
        RHS[node_idx] = 0

        # Shear Force V = (EIw'')' = 0 at the delta-node.
        EI_vec = E .* I_z
        EI_prime = A1 * EI_vec
        shear_operator = diagm(EI_prime) * A2 + diagm(EI_vec) * A3

        LHS[delta_node_idx, :] = shear_operator[delta_node_idx, :]
        RHS[delta_node_idx] = 0

    else
        error("Unknown boundary condition type: $bc_type")
    end
end