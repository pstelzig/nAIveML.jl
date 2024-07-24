"""
Simple and self contained implementation of some regression formulae. 
Part of nAIveML.jl. 

Copyright Dr. Philipp Emanuel Stelzig, 2023-2024
"""
module Regression

include("optim.jl")

import .Optim as optim

using LinearAlgebra

"""
Assumes the data X to be a matrix like 

[x1 | x2 | ... | xN] of dimensions d x N 

where xi in R^d and there are N samples x1,...,xN.

The matrix of labels Y shall be like

[y1 | y2 | ... | yN] of dimensions m x N 

where yi in R^m and there are again N samples y1,...,yN.

It returns a Matrix of weights W in R^{m x d} and a vector c in R^m
as a tuple (W,c) such that

sum_{i=1}^{N}||(W*xi + c) - yi ||^2 = min! 

when minimizing over both W and c. Here, ||  || denotes the Frobenius Norm.
"""
function Linear(X::Matrix{<:Real}, Y::Matrix{<:Real})
    if size(X)[2] != size(Y)[2]
        throw(DimensionMismatch("Arguments data and label must have the same number of columns."))
    end

    N = size(X)[2]
    X_ = [ones(1, N); X]

    W_ = transpose((X_ * transpose(X_))\(X_*transpose(Y)))

    c = W_[:,1]
    W = W_[:,2:end]

    return W, c
end

"""
Ridge regression which adds a regularizing term on the weights to the 
linear regression objective to avoid overfitting.
"""
function Ridge(X::Matrix{<:Real}, Y::Matrix{<:Real}, alpha::Real)
    if size(X)[2] != size(Y)[2]
        throw(DimensionMismatch("Arguments data and label must have the same number of columns."))
    end

    N = size(X)[2]
    X_ = [ones(1, N); X]

    W_ = transpose((alpha*I + X_ * transpose(X_))\(X_*transpose(Y)))

    c = W_[:,1]
    W = W_[:,2:end]

    return W, c
end

"""
Multinomial logistic regression data structure
"""
mutable struct Logistic
    W::Matrix{<:Real}
    d::Integer
    m::Integer

    function Logistic(dim, nClasses)
        new(zeros(nClasses-1, dim + 1), dim, nClasses)
    end
end

function softmax(k, x, W)
    m = size(W)[1] + 1
    d = size(W)[2] - 1

    x_ = [1; x]

    @assert length(x) == d "Dimensions do not match"

    if k != 0 
        return exp(W[k,:]' * x_) / (1 + sum([exp(W[l,:]' * x_) for l in 1:m-1]))
    else
        return 1 - sum([softmax(l, x, W) for l in 1:m-1])
    end
end

function train!(logit::Logistic, X::Matrix{<:Real}, Y::Vector{<:Integer}, tol::Real, maxSteps::Integer)
    # Number of data points
    N = length(Y)

    function p(k, j, W)
        return softmax(k, X[:,j], W)
    end

    """
    Negative of the log-likelihood function for logistic regression
    """
    function J(W)
        s = 0.0
        for j in 1:N
            for k in 0:logit.m-1
                if Y[j] == k
                    s += log(p(k,j,W))
                end
            end
        end

        return -s
    end

    """
    Derivative of the negative of the log-likelihood function for logistic regression
    """
    function DJ(W)
        D = zeros(logit.m-1, logit.d+1)

        X_ = [1; X]
        for a in 1:logit.m-1
            for b in 1:logit.d+1
                D[a,b] = sum([(p(a,j,W) - ==(a, Y[j]))*X_[b,j] for j in 1:N])
            end
        end

        return D
    end

    W_low = -1e10*ones(logit.m-1, logit.d+1)
    W_high = 1e10*ones(logit.m-1, logit.d+1)
    logit.W, _ = optim.gdunconstr(J, DJ, logit.W, W_low, W_high, 1.0, 0.8, tol, maxSteps)
end

"""
Forward pass for the logistic regression. Returns the softmax probabilities
"""
function forward(logit::Logistic, x_in::Vector{<:Real})
    p = zeros(logit.m)

    for k in 0:logit.m-1
        p[k+1] = softmax(k, x_in, logit.W)
    end

    return p
end

end