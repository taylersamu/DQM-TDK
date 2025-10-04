using SymPy

"""
    get_timoshenko_analytical_solution(E, I, G, A, k, L, load_start, load_end, q_val, bc_start, bc_end)

Generates fast numerical functions for the analytical deflection and rotation of a Timoshenko beam.

# Arguments
- `E::Real`: Modulus of Elasticity.
- `I::Real`: Second moment of inertia.
- `G::Real`: Shear Modulus.
- `A::Real`: Cross-sectional area.
- `k::Real`: Timoshenko shear coefficient.
- `L::Real`: Length of the beam.
- `load_start::Real`: Start of the distributed load.
- `load_end::Real`: End of the distributed load.
- `q_val::Real`: Magnitude of the distributed load.
- `bc_start::String`: Boundary condition at x=0. Options: "clamped", "simply-supported", "free".
- `bc_end::String`: Boundary condition at x=L. Options: "clamped", "simply-supported", "free".

# Returns
- `Tuple{Function, Function}`: A tuple containing two functions, `(w(x), theta(x))`, for deflection and rotation.
"""
function get_timoshenko_analytical_solution(
    E,
    I,
    G,
    A,
    k,
    L,
    load_start,
    load_end,
    q_val,
    bc_start::String,
    bc_end::String,
)

    # 1. Define symbolic variables
    @syms x_s E_s I_s G_s A_s k_s L_s a_s b_s q_s
    @syms A_1 A_2 A_3 A_4 B_1 B_2 B_3 B_4 C_1 C_2 C_3 C_4

    # 2. Define general solutions for deflection (w) and rotation (theta)
    # Segment 1: 0 <= x < load_start
    w_1 = A_1*x_s^3/6 + A_2*x_s^2/2 + A_3*x_s + A_4
    theta_1 = A_1*x_s^2/2 + A_2*x_s + A_3 + (E_s*I_s*A_1)/(k_s*G_s*A_s)

    # Segment 2: load_start <= x <= load_end
    w_2 = B_1*x_s^3/6 + B_2*x_s^2/2 + B_3*x_s + B_4 + q_s*x_s^4/(24*E_s*I_s)
    theta_2 =
        B_1*x_s^2/2 +
        B_2*x_s +
        B_3 +
        (E_s*I_s*B_1)/(k_s*G_s*A_s) +
        q_s*x_s^3/(6*E_s*I_s) +
        q_s*x_s/(k_s*G_s*A_s)

    # Segment 3: load_end < x <= L
    w_3 = C_1*x_s^3/6 + C_2*x_s^2/2 + C_3*x_s + C_4
    theta_3 = C_1*x_s^2/2 + C_2*x_s + C_3 + (E_s*I_s*C_1)/(k_s*G_s*A_s)

    # Moment (M) and Shear (V) expressions
    M_1 = E_s*I_s*diff(theta_1, x_s)
    V_1 = k_s*G_s*A_s*(diff(w_1, x_s) - theta_1)
    M_2 = E_s*I_s*diff(theta_2, x_s)
    V_2 = k_s*G_s*A_s*(diff(w_2, x_s) - theta_2)
    M_3 = E_s*I_s*diff(theta_3, x_s)
    V_3 = k_s*G_s*A_s*(diff(w_3, x_s) - theta_3)

    # 3. Build the system of 12 equations
    equations = Sym[]

    # Boundary conditions at start (x=0)
    if bc_start == "clamped"
        push!(equations, subs(w_1, (x_s, 0)))
        push!(equations, subs(theta_1, (x_s, 0)))
    elseif bc_start == "simply-supported"
        push!(equations, subs(w_1, (x_s, 0)))
        push!(equations, subs(M_1, (x_s, 0)))
    elseif bc_start == "free"
        push!(equations, subs(M_1, (x_s, 0)))
        push!(equations, subs(V_1, (x_s, 0)))
    else
        error("Invalid start boundary condition: $bc_start")
    end

    # Boundary conditions at end (x=L)
    if bc_end == "clamped"
        push!(equations, subs(w_3, (x_s, L_s)))
        push!(equations, subs(theta_3, (x_s, L_s)))
    elseif bc_end == "simply-supported"
        push!(equations, subs(w_3, (x_s, L_s)))
        push!(equations, subs(M_3, (x_s, L_s)))
    elseif bc_end == "free"
        push!(equations, subs(M_3, (x_s, L_s)))
        push!(equations, subs(V_3, (x_s, L_s)))
    else
        error("Invalid end boundary condition: $bc_end")
    end

    # Continuity conditions at x = a_s
    push!(equations, subs(w_1 - w_2, (x_s, a_s)))
    push!(equations, subs(theta_1 - theta_2, (x_s, a_s)))
    push!(equations, subs(M_1 - M_2, (x_s, a_s)))
    push!(equations, subs(V_1 - V_2, (x_s, a_s)))

    # Continuity conditions at x = b_s
    push!(equations, subs(w_2 - w_3, (x_s, b_s)))
    push!(equations, subs(theta_2 - theta_3, (x_s, b_s)))
    push!(equations, subs(M_2 - M_3, (x_s, b_s)))
    push!(equations, subs(V_2 - V_3, (x_s, b_s)))

    # 4. Solve for the constants
    constants = [A_1, A_2, A_3, A_4, B_1, B_2, B_3, B_4, C_1, C_2, C_3, C_4]
    solution = solve(equations, constants)

    # 5. Substitute solutions back into deflection and rotation equations
    w_1_sol = subs(w_1, solution...);
    theta_1_sol = subs(theta_1, solution...)
    w_2_sol = subs(w_2, solution...);
    theta_2_sol = subs(theta_2, solution...)
    w_3_sol = subs(w_3, solution...);
    theta_3_sol = subs(theta_3, solution...)

    # 6. Lambdify expressions for numerical evaluation
    params = [E_s, I_s, G_s, A_s, k_s, L_s, a_s, b_s, q_s, x_s]
    w_1_lambda = lambdify(w_1_sol, params);
    theta_1_lambda = lambdify(theta_1_sol, params)
    w_2_lambda = lambdify(w_2_sol, params);
    theta_2_lambda = lambdify(theta_2_sol, params)
    w_3_lambda = lambdify(w_3_sol, params);
    theta_3_lambda = lambdify(theta_3_sol, params)

    # 7. Return a pair of functions for deflection and rotation
    w_func = function (x::Real)
        if x < load_start
            return w_1_lambda(E, I, G, A, k, L, load_start, load_end, q_val, x)
        elseif x >= load_start && x <= load_end
            return w_2_lambda(E, I, G, A, k, L, load_start, load_end, q_val, x)
        else
            return w_3_lambda(E, I, G, A, k, L, load_start, load_end, q_val, x)
        end
    end

    theta_func = function (x::Real)
        if x < load_start
            return theta_1_lambda(E, I, G, A, k, L, load_start, load_end, q_val, x)
        elseif x >= load_start && x <= load_end
            return theta_2_lambda(E, I, G, A, k, L, load_start, load_end, q_val, x)
        else
            return theta_3_lambda(E, I, G, A, k, L, load_start, load_end, q_val, x)
        end
    end

    return w_func, theta_func
end