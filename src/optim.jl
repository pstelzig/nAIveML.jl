"""
Simple and self-contained implementation of some optimization algorithms for 
use with the nAIveML.jl project. 

Copyright Dr. Philipp Emanuel Stelzig, 2023-2024
"""
module Optim

using LinearAlgebra
using JuMP
using Ipopt
using NLsolve

"""
Solves the unconstrained optimization problem

f(x) = min! 

x_low <= x < x_high

with initial value x_0 using a simple gradient descent algorithm with elementary stepsize control.
The derivative of the objective function f must be given in the argument df.
"""
function gdunconstr(f::Function, df::Function, x_0, x_low, x_high, s::Real, sRel::Real, xTol::Real, n_max::Integer)
    x_prev = x_0
    f_prev = f(x_prev)

    x_new = x_prev
    f_new = f_prev

    steps = 1

    for k in 1:n_max
        println("Step $k: Current candidate for argmin x_prev=$x_prev with value f_prev=$f_prev")

        steps = k
        if steps == n_max
            println("Maximum number of steps n_max=$n_max reached.")
        end        

        D = df(x_prev)
        x_new = x_prev - s*D   
        
        # To check (x_new < x_low) || (x_new > x_high) need to flatten multidimensional arrays to vectors
        if (collect(Iterators.flatten(x_new)) < collect(Iterators.flatten(x_low))) || (collect(Iterators.flatten(x_new)) > collect(Iterators.flatten(x_high)))
            println("The new argmin candidate x_new=$x_new lies outside of the bounds. Repeating with smaller stepsize")
            s = sRel * s
            continue
        end        

        f_new = f(x_new)

        if norm(x_new - x_prev) < xTol
            println("Difference of current x_new=$x_new and previous x_prev=$x_prev dropped below xTol=$xTol. Aborting loop.")
            break
        end

        if f_new >= f_prev
            println("Value f_new=$f_new has not fallen below f_prev=$f_prev. Repeating step with smaller stepsize")
            s = sRel * s
            continue
        end

        x_prev = x_new
        f_prev = f_new
    end

    println("Finished after $steps steps. Candidate for argmin is x=$x_prev with value f(x)=$(f(x_prev))")

    return x_new, f_new
end

"""
Naive solver that solves a nonlinear program with linear constraints

f(x) = min! 

Ax = b

for a differentiable function f : R^n -> R, A in R^{m x n} (m<=n) and b in R^m 
"""
function gdlinconstr(f::Function, df::Function, A, b, x_0, s::Real, sRel::Real, tol::Real, n_max::Integer)
    Q,R = qr(A')
    l = rank(A)
    (m,n) = size(A)

    # Projection matrix subspace U = b + Ker(A) = b + lin(a_1,...,a_m)^orth = b + lin(q_1,...,q_l)^orth = b + lin(q_{l+1},...,q_n)
    Q_U = Q[:,l+1:end] 

    # Solve Ax=b
    x_b = Q*[R'\b; zeros(n-m)]

    # Project initial value onto U
    x_prev = x_b + Q_U*Q_U'*(x_0 - x_b)
    f_prev = f(x_prev)

    x_new = x_prev
    f_new = f_prev

    steps = 1

    for k in 1:n_max
        println("Step $k: Current candidate for argmin x_prev=$x_prev with value f_prev=$f_prev")

        steps = k
        if steps == n_max
            println("Maximum number of steps n_max=$n_max reached.")
        end        

        D = df(x_prev)

        # Project onto x_prev + h*df(x_prev) onto subspace U
        x_new = x_b + Q_U*Q_U'*(x_prev - s*D - x_b)

        f_new = f(x_new)

        if norm(x_new - x_prev) < tol
            println("Difference of current x_new=$x_new and previous x_prev=$x_prev dropped below xTol=$tol. Aborting loop.")
            break
        end

        if f_new >= f_prev
            println("Value f_new=$f_new has not fallen below f_prev=$f_prev. Repeating step with smaller stepsize")
            s = sRel * s
            continue
        end

        x_prev = x_new
        f_prev = f_new
    end

    println("Finished after $steps steps. Candidate for argmin is x=$x_prev with value f(x)=$(f(x_prev))")

    return x_new, f_new
end

"""
Naive solver algorithm that solves a nonlinear program with nonlinear equality constraints

f(x) = min! 

h(x) = 0

for differentiable functions f : R^n -> R, h : R^n -> R^m, m<=n. Instead of using
a Lagrangian formulation, this implementation linearizes the equality constraint in 
each step and solves sequence a linearly constrained problem, where the solution 
from each step is projected back onto the manifold {x : h(x) = 0}. 

Drawn up by Philipp Emanuel Stelzig on the whiteboard as a nonlinear iterative 
extension of gdlinconstr. 
"""
function gdnonlinconstr(f::Function, df::Function, h::Function, dh::Function, x_0, s::Real, sRel::Real, tol::Real, n_max::Integer)
    s_init = s
    x_prev = x_0
    f_prev = f(x_0)

    x_new = x_prev
    f_new = f_prev

    steps = 1

    for k in 1:n_max
        println("Step $k, step size $s: Current candidate for argmin x_prev=$x_prev with value f_prev=$f_prev")

        steps = k
        if steps == n_max
            println("Maximum number of steps n_max=$n_max reached.")
        end     
        
        A = dh(x_prev)
        b = A*x_prev

        Q,R = qr(A')
        (m,n) = size(A)

        l = rank(A)
    
        # Projection matrix for tangential space T
        # T = x_prev + Ker(Dh(x_prev)) = x_prev + lin(a_1,...,a_m)^orth = x_prev + lin(q_1,...,q_l)^orth = b + lin(q_{l+1},...,q_n)
        Q_T = Q[:,l+1:end] 
    
        # Solve Ax=b
        z_b = Q*[R'\b; zeros(n-m)]        

        # Make optimization step in tangential space T
        D = df(x_prev)
        z_new = z_b + Q_T*Q_T'*(x_prev - s*D - z_b)

        # Project z_new onto manifold M = {x : h(x) = 0}
        function proj(v)
            res = h(z_new + A'*v)
            return res
        end

        sol = nlsolve(proj, zeros(m))
        lambda = sol.zero

        x_new = z_new + A'*lambda

        if norm(h(x_new)) > tol
            println("Candidate x_new=$x_new does not satisfy h(x_new)=0, is $(h(x_new)). Repeating with smaller stepsize.")
            s = sRel * s
            continue            
        end

        f_new = f(x_new)

        if norm(x_new - x_prev) < tol
            println("Difference of current x_new=$x_new and previous x_prev=$x_prev dropped below xTol=$tol. Aborting loop.")
            break
        end

        if f_new >= f_prev
            println("Value f_new=$f_new has not fallen below f_prev=$f_prev. Repeating step with smaller stepsize")
            s = sRel * s
            continue
        end

        x_prev = x_new
        f_prev = f_new
        s = s_init        
    end

    println("Finished after $steps steps. Candidate for argmin is x=$x_prev with value f(x)=$(f(x_prev))")

    return x_new, f_new
end

"""
Solves the constrained quatratic program

J(x) := 1/2*x'Qx + c'x = min!

s.t. Ax <= b

where Q in R^{n x n}, c in R^n, A in R^{m x n}, b in R^m. It is assumed that Q is symmetric and 
positive definite, hence in particular invertible. Solving is done by gradient descent on the Lagrangian dual 

q(v) := 1/2 v'Sv + r'v  = min!

s.t. v >= 0

with S = AQ^{-1}A' and r = c'Q^{-1}A' + b', yielding by Karish-Kuhn-Tucker conditions the minimizer 

x_min = -Q^{-1}(A'v_min + c)
"""
function qplinconstr(Q::Matrix{<:Real}, c::Vector{<:Real}, A::Matrix{<:Real}, b::Vector{<:Real}, s::Real, sRel::Real, tol::Real, n_max::Integer)
    m = size(b)
    Q_inv = inv(Q)

    S = A*Q_inv*A'
    r = A*Q_inv*c + b

    function q(v)
        qv = 0.5*v'*S*v + r'v

        return qv
    end

    function Dq(v)
        dqv = S*v + r

        return dqv
    end

    v_0 = ones(m)
    v_low = zeros(m)
    v_high = 10^10*ones(m)

    v_min, _ = gdunconstr(q, Dq, v_0, v_low, v_high, s, sRel, tol, n_max)

    x_min = -Q_inv*(A'*v_min + c)
    J_min = 0.5*x_min'*Q*x_min + c'*x_min

    return x_min, J_min
end

"""
Solves the fully constrained nonlinear program

f(x) = min!

s.t. g(x) <= 0 and h(x) = 0

for f : R^n -> R, g : R^n -> R^m, h : R^n -> R^l where the constraints are understood elementwise. 
This function simply wraps JuMP and Ipopt. 

JuMP uses automatic differentiation. It is therefore required to use Julia-native function definitions 
for f, g, and h only. JuMP does not support black box optimization. For automatic differentiation to 
work, a function must (!) have abstract argument types Real or Number. Concrete data types are not 
supported.
    
See https://jump.dev/JuMP.jl/stable/manual/nonlinear/#Automatic-differentiation    
"""
function gdfullconstr(f::Function, g::Union{Function,Nothing}, h::Union{Function,Nothing}, x_0, tol::Real, n_max::Integer)
    model = Model(Ipopt.Optimizer)
    n = length(x_0)
    @variable(model, x[1:n])
    set_start_value.(x, x_0)

    @objective(model, Min, f(x))

    if g !== nothing
        @constraint(model, g(x) <= 0)
    end

    if h !== nothing
        @constraint(model, h(x) == 0)
    end

    set_optimizer_attribute(model, "max_iter", n_max)

    optimize!(model)
    
    solution_summary(model)

    x_min = value.(x)
    f_min = f(x_min)

    if g !== nothing
        @assert all(g(x_min) .<= tol) "Minimizer x_min violates constraint g(x)<=0"
    end

    if h !== nothing
        @assert all(norm.(h(x_min)) .<= tol) "Minimizer x_min violates constraint t(x)=0"
    end

    return x_min, f_min
end

end