# BE_DifferentialEquations.jl

using DifferentialEquations, LinearAlgebra

# It is assumed these files are in the same directory.
include("Smooth_load.jl")
include("DQM_WEIGHTS.jl") # Required for the chebyshev_grid function

"""
    solve_beam_diffeq(N, a, b, E, I_z, bc_start, bc_end; kwargs...)

Solves the Bernoulli-Euler beam equation using DifferentialEquations.jl.

The 4th-order ODE w''''(x) = q(x)/EI is converted into a system of
four 1st-order ODEs with a state vector u = [w, w', w'', w'''].
"""
function solve_beam_diffeq(
    N::Int,
    a::Float64,
    b::Float64,
    E::Float64,
    I_z::Float64,
    bc_start::String,
    bc_end::String;
    load_F::Float64,
    load_start::Float64,
    load_end::Float64,
    load_steepness::Int,
)
    # --- 1. Define the ODE System ---
    # State vector u = [w, w', w'', w''']
    # Parameters p = (EI, F, steepness, start, end)
    function beam_ode!(du, u, p, x)
        EI, F, s, x_s, x_e = p
        q_x = Smooth_load(x, F, s, x_s, x_e)

        du[1] = u[2]      # dw/dx   = w'
        du[2] = u[3]      # dw'/dx  = w''
        du[3] = u[4]      # dw''/dx = w'''
        du[4] = q_x / EI  # dw'''/dx = q(x)/EI
    end

    # --- 2. Define the Boundary Conditions ---
    # This function calculates the residual based on the solution at the boundaries, u = (u(a), u(b)).
    # The solver aims to drive these residuals to zero.
    function beam_bc!(residual, u, p, x)
        u_start, u_end = u

        # Boundary conditions at the start of the beam (x=a)
        if bc_start == "clamped"
            residual[1] = u_start[1]  # Deflection w(a) = 0
            residual[2] = u_start[2]  # Slope w'(a) = 0
        else
            error("Unsupported boundary condition at start: $bc_start")
        end

        # Boundary conditions at the end of the beam (x=b)
        if bc_end == "clamped"
            residual[3] = u_end[1]    # Deflection w(b) = 0
            residual[4] = u_end[2]    # Slope w'(b) = 0
        elseif bc_end == "simply supported"
            residual[3] = u_end[1]    # Deflection w(b) = 0
            residual[4] = u_end[3]    # Moment EIw''(b) = 0
        elseif bc_end == "free"
            residual[3] = u_end[3]    # Moment EIw''(b) = 0
            residual[4] = u_end[4]    # Shear EIw'''(b) = 0
        else
            error("Unsupported boundary condition at end: $bc_end")
        end
    end

    # --- 3. Setup and Solve the BVP ---
    xspan = (a, b)
    params = (E * I_z, load_F, load_steepness, load_start, load_end)

    # An initial guess of all zeros for the solution is often sufficient.
    bvp = BVPProblem(beam_ode!, beam_bc!, zeros(4), xspan, params)

    # Define the grid where the solution should be saved. This matches the DQM grid.
    grid = chebyshev_grid(a, b, N)

    # Solve the BVP. GeneralMIRK4 is a robust choice.
    # `saveat` ensures the output is on our desired grid.
    # Tolerances are set tightly to ensure accuracy is mainly controlled by the grid density (N).
    sol = solve(bvp, GeneralMIRK4(), saveat = grid, abstol = 1e-4, reltol = 1e-4)

    # --- 4. Extract and Return Results ---
    # The solution `sol.u` is a vector of state vectors. We extract the first component (deflection) from each.
    deflection = [u[1] for u in sol.u]

    return sol.t, deflection # Return grid and deflection
end