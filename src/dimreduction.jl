"""
Simple and self-contained implementation of some dimension reduction formulae. 
Part of nAIveML.jl. 

Copyright Dr. Philipp Emanuel Stelzig, 2023-2024
"""

module DimensionReduction

using LinearAlgebra

"""
Assumes the data X to be a matrix like 

[x1 | x2 | ... | xN] of dimensions d x N 

where xi in R^d and there are N samples x1,...,xN.

It returns the principle components and the respective eigenvalues as a tuple 
(vals, vecs) where vals are sorted in descending order and the kth entry of
vals corresponds to the eigenvector in the kth column of vecs.
"""
function pca(X::Matrix{<:Real}, make_meanval_free::Bool)
    if make_meanval_free
        N = size(X)[2]
        m = sum(X, dims=2)/N
        X = X .- m 
    end

    vals, vecs = eigen(X*transpose(X))

    # Change to descending order for convenience
    vals = vals[end:-1:1]
    vecs = vecs[:,end:-1:1]

    if make_meanval_free
        return vals, vecs, m
    else
        return vals, vecs
    end
end

end