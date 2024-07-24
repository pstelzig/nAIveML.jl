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

end