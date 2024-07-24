"""
Simple self-contained implementation of support vector machines. 
Part of nAIveML.jl. 

For the theory see e.g. Yaser Abu-Mostafa's lecture on Machine Learning from 2012.

Copyright Dr. Philipp Emanuel Stelzig, 2023-2024
"""
module SupportVectorMachines

include("optim.jl")

import .Optim as optim

using LinearAlgebra

mutable struct LinHardMargin
    w::Array{<:Real}
    b::Real
    sv::Matrix{<:Real}
    ysv::Array{<:Integer}
    lambdasv::Array{<:Real}

    function LinHardMargin()
        new(Array{Real}([]), 0.0, Array{Real}(undef, 0, 0), Array{Integer}([]), Array{Real}([]))
    end
end

function train!(svm::LinHardMargin, X::Matrix{<:Real}, y::Vector{<:Integer}, tol::Real, maxSteps::Integer)
    N = size(X)[2]

    @assert (size(X)[2] == N) && (length(y) == N) "Dimensions do not match"

    M = zeros(Real, N, N)
    for i in 1:N
        for j in 1:N
            M[i,j] = y[i]*(X[:,i]'*X[:,j])*y[j]
        end
    end

    """
    Objective associated with the SVM's dual problem
        q(v) = 1/2* v'Mv - sum_i v_i
    """    
    function q(v)
        return 0.5*v'*M*v - ones(Real, N)'*v
    end

    # Minimizes the objective associated with the SVM's dual problem
    #   q(v) = 1/2* v'Mv - sum_i v_i
    #   s.t. y'v =0 and v>=0
    lambda_0 = ones(Real, N)
    lambda, _ = optim.gdfullconstr(q, x -> -x, x -> y'*x, lambda_0, tol, maxSteps)

    # Support vector indices
    svidx = [j for j in 1:N if abs(lambda[j]) > tol] 

    # Weights and bias
    svm.w = sum([lambda[j]*y[j]*X[:,j] for j in svidx])

    n_sv = length(svidx)
    svm.sv = X[:,svidx]
    svm.ysv = y[svidx]
    svm.lambdasv = lambda[svidx]
    bs = zeros(n_sv)
    for l in 1:n_sv
        bs[l] = 1/y[svidx[l]] - svm.w'*X[:,svidx[l]]
    end

    # Check correctness: Entries of b must be identical
    bmax = maximum(bs)
    bmin = minimum(bs)
    @assert abs(bmax - bmin) < tol "SVM calculation yielded inconistent results for b: max_i b_i=$bmax, min_i b_i = $bmin"

    svm.b = sum(bs)/n_sv
end

function forward(svm::LinHardMargin, x_in::Vector{<:Real})
    r = sign(svm.w'*x_in + svm.b)
    return r
end


end