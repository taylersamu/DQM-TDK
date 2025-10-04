using SymPy
BLAS.set_num_threads(16)
"""
    get_bernoulli_euler_analytical_solution(E, I, L, load_start, load_end, q_val, bc_start, bc_end)

Generates a fast numerical function for the analytical deflection of a Bernoulli-Euler beam.

The beam is subjected to a uniform distributed load `q_val` between `load_start` and `load_end`.

# Arguments
- `E::Real`: Modulus of Elasticity.
- `I::Real`: Second moment of inertia.
- `L::Real`: Length of the beam.
- `load_start::Real`: Starting position of the distributed load.
- `load_end::Real`: Ending position of the distributed load.
- `q_val::Real`: Magnitude of the distributed load (force per unit length).
- `bc_start::String`: Boundary condition at the start (x=0). Options: "clamped", "simply supported", "free".
- `bc_end::String`: Boundary condition at the end (x=L). Options: "clamped", "simply supported", "free".

# Returns
- `Function`: A function `w(x)` that takes a position `x` along the beam and returns the analytical deflection.
"""
function get_bernoulli_euler_analytical_solution(
    E,
    I,
    L,
    load_start,
    load_end,
    q_val,
    bc_start::String,
    bc_end::String,
)

    # 1. Define symbolic variables
    @syms x_s E_s I_s L_s a_s b_s q_s
    @syms A_1 A_2 A_3 A_4 B_1 B_2 B_3 B_4 C_1 C_2 C_3 C_4

    # 2. Define general solutions for the three beam segments
    # Segment 1: 0 <= x < load_start
    w_1 = A_1*x_s^3/6 + A_2*x_s^2/2 + A_3*x_s + A_4

    # Segment 2: load_start <= x <= load_end
    w_2 = q_s/(E_s*I_s) * x_s^4/24 + B_1*x_s^3/6 + B_2*x_s^2/2 + B_3*x_s + B_4

    # Segment 3: load_end < x <= L
    w_3 = C_1*x_s^3/6 + C_2*x_s^2/2 + C_3*x_s + C_4

    # Derivatives for continuity (rotation, moment, shear)
    w_1_derivs = [diff(w_1, x_s, i) for i = 1:3]
    w_2_derivs = [diff(w_2, x_s, i) for i = 1:3]
    w_3_derivs = [diff(w_3, x_s, i) for i = 1:3]

    # 3. Build the system of 12 equations from boundary and continuity conditions
    equations = Sym[]

    # Boundary conditions at start (x=0)
    if bc_start == "clamped"
        push!(equations, subs(w_1, (x_s, 0))) # Deflection is 0
        push!(equations, subs(w_1_derivs[1], (x_s, 0))) # Slope is 0
    elseif bc_start == "simply supported"
        push!(equations, subs(w_1, (x_s, 0))) # Deflection is 0
        push!(equations, subs(w_1_derivs[2], (x_s, 0))) # Moment is 0
    elseif bc_start == "free"
        push!(equations, subs(w_1_derivs[2], (x_s, 0))) # Moment is 0
        push!(equations, subs(w_1_derivs[3], (x_s, 0))) # Shear is 0
    else
        error("Invalid start boundary condition: $bc_start")
    end

    # Boundary conditions at end (x=L)
    if bc_end == "clamped"
        push!(equations, subs(w_3, (x_s, L_s))) # Deflection is 0
        push!(equations, subs(w_3_derivs[1], (x_s, L_s))) # Slope is 0
    elseif bc_end == "simply supported"
        push!(equations, subs(w_3, (x_s, L_s))) # Deflection is 0
        push!(equations, subs(w_3_derivs[2], (x_s, L_s))) # Moment is 0
    elseif bc_end == "free"
        push!(equations, subs(w_3_derivs[2], (x_s, L_s))) # Moment is 0
        push!(equations, subs(w_3_derivs[3], (x_s, L_s))) # Shear is 0
    else
        error("Invalid end boundary condition: $bc_end")
    end

    # Continuity conditions at the start of the load (x = a_s)
    push!(equations, subs(w_1 - w_2, (x_s, a_s))) # Equal deflection
    push!(equations, subs(w_1_derivs[1] - w_2_derivs[1], (x_s, a_s))) # Equal rotation
    push!(equations, subs(w_1_derivs[2] - w_2_derivs[2], (x_s, a_s))) # Equal moment
    push!(equations, subs(w_1_derivs[3] - w_2_derivs[3], (x_s, a_s))) # Equal shear

    # Continuity conditions at the end of the load (x = b_s)
    push!(equations, subs(w_2 - w_3, (x_s, b_s))) # Equal deflection
    push!(equations, subs(w_2_derivs[1] - w_3_derivs[1], (x_s, b_s))) # Equal rotation
    push!(equations, subs(w_2_derivs[2] - w_3_derivs[2], (x_s, b_s))) # Equal moment
    push!(equations, subs(w_2_derivs[3] - w_3_derivs[3], (x_s, b_s))) # Equal shear

    # 4. Solve for the 12 constants
    constants = [A_1, A_2, A_3, A_4, B_1, B_2, B_3, B_4, C_1, C_2, C_3, C_4]
    solution = solve(equations, constants)

    # 5. Substitute the solved constants back into the general solutions
    w_1_sol = subs(w_1, solution...)
    w_2_sol = subs(w_2, solution...)
    w_3_sol = subs(w_3, solution...)

    # 6. Convert the symbolic expressions into fast, callable numerical functions
    params = [E_s, I_s, L_s, a_s, b_s, q_s, x_s]
    w_1_lambda = lambdify(w_1_sol, params)
    w_2_lambda = lambdify(w_2_sol, params)
    w_3_lambda = lambdify(w_3_sol, params)

    # 7. Return a single function that uses the correct segment based on x
    return function w(x)
        if x < load_start
            return w_1_lambda(E, I, L, load_start, load_end, q_val, x)
        elseif x >= load_start && x <= load_end
            return w_2_lambda(E, I, L, load_start, load_end, q_val, x)
        else
            return w_3_lambda(E, I, L, load_start, load_end, q_val, x)
        end
    end
end