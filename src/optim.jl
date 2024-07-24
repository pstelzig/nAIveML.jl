"""
Simple and self-contained implementation of some optimization algorithms for 
use with the nAIveML.jl project. 

Copyright Dr. Philipp Emanuel Stelzig, 2023-2024
"""
module Optim

using LinearAlgebra

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

end