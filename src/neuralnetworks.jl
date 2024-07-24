"""
Simple and self-contained implementation of a forward neural network. 
Part of nAIveML.jl. 

For the theory see Yaser Abu-Mostafa's lecture on Machine Learning from 2012.

Copyright Dr. Philipp Emanuel Stelzig, 2023-2024
"""
module NeuralNetworks

using LinearAlgebra

mutable struct Network
    nIn::Integer
    nLayers::Integer
    nNeuronsInLayer::Array{<:Integer}
    W::Dict # weights[l] is matrix of dimensions (nNeuronsInLayer[l], nNeuronsInLayer[l-1] + 1)

    theta::Function # Activation function
    dtheta::Function

    errdist::Function # Loss function
    derrdistx::Function


    function initWeight(d1::Integer, d2::Integer)
        W_init = rand(d1, d2+1)

        W_init[:,1] = zeros(d1) # Bias terms to zero

        colSum = sum(W_init[:,2:end], dims=2)
        W_init[:,2:end] = (W_init[:,2:end] .- colSum/d2) * sqrt(1.0/d2) # Xavier initialization
        
        return W_init
    end

    """
    Constructor for the NeuralNetwork struct

    d: Int64 
        Dimension of input vector

    nInLayers: Array{Float64}
        Array of number of neurons in each layer. Number of layers equals the length of nInLayers.
    """    
    function Network(d::Integer, nInLayers::Array{<:Integer}, actFunc::Function, dactFunc::Function, loss::Function, dloss::Function)
        nL = length(nInLayers)
        W_init = Dict()

        # Initialize weights: Random, mean-value free, zero bias
        for l in 2:nL
            W_init[l] = initWeight(nInLayers[l], nInLayers[l-1])
        end

        W_init[1] = initWeight(nInLayers[1], d)

        new(d, nL, nInLayers, W_init, actFunc, dactFunc, loss, dloss)
    end
end

"""
    X = forward(nn, x_in)


Simple feed forward step. 

Returns the outputs of all layers in the neural network
as a dictionary where the key is the layer index (starting from 0 as the input layer).
"""
function forward(nn::Network, x_in::Array{<:Real})
    X = Dict(l => zeros(nn.nNeuronsInLayer[l] + 1) for l in 1:nn.nLayers)
    X[0] = [1; x_in]

    for l in 1:nn.nLayers
        X[l] = [1; nn.theta.(nn.W[l]*X[l-1])]
    end

    return X
end

"""
    X = forward(nn, x_inject, idx)

Performs a forward pass when injecting the state x_inject into the layer
at hidden layer with index idx. Returns the outputs of all layers as a 
dictionary with idx as the starting index. 
"""
function forward(nn::Network, x_inject::Array{<:Real}, idx::Integer)
    if (idx < 1) || (idx > nn.nLayers-1)
        error("Injection index $idx must correspond to the first layer 0 or the last layer $(nn.nLayers)")
    end

    X = Dict(l => zeros(nn.nNeuronsInLayer[l] + 1) for l in idx:nn.nLayers)
    X[idx] = x_inject

    for l in idx+1:nn.nLayers
        X[l] = [1; nn.theta.(nn.W[l]*X[l-1])]
    end

    return X
end

"""
    Delta = backward(nn, X, y_out)

Backward step for a feed forward network. Uses the backpropagation algorithm to compute
the weight update Delta. 
"""
function backward(nn::Network, X::Dict, y_out::Array{<:Real})
    Delta = Dict(l => zeros(nn.nNeuronsInLayer[l]) for l in 1:nn.nLayers)

    Delta[nn.nLayers] = Diagonal(nn.dtheta.(nn.W[nn.nLayers] * X[nn.nLayers-1]))*nn.derrdistx(X[nn.nLayers][2:end], y_out)

    for l in Iterators.reverse(1:nn.nLayers-1)
        Delta[l] = nn.dtheta.(nn.W[l]*X[l-1]).*((nn.W[l+1]')[2:end,:]*Delta[l+1])
    end

    return Delta
end

function trainingStep!(nn::Network, x::Matrix{<:Real}, y::Matrix{<:Real}, stepSize::Real)
    abserr = 0.0
    nData = size(x)[2]

    for i in 1:nData
        X = forward(nn, x[:,i])
        Delta = backward(nn, X, y[:,i])

        for l in 1:nn.nLayers
            nn.W[l] = nn.W[l] - stepSize*Delta[l]*X[l-1]'
        end

        abserr += nn.errdist(X[nn.nLayers][2:end], y[:,i])
    end

    return abserr
end

function calcRelStepSize!(nn::Network, W_prev::Dict, loss::Real, loss_prev::Real)
    sRel = 0.8

    # Step size in last step too large, repeat with smaller step size
    if loss > loss_prev
        nn.W = copy(W_prev)
        return sRel
    else
        return 1.0
    end

    return sRel
end

function train!(nn::Network, 
                    x_train::Matrix{<:Real}, y_train::Matrix{<:Real},
                    x_test::Matrix{<:Real}, y_test::Matrix{<:Real},
                    relErrFunc::Function,
                    maxSteps::Integer, tol::Real, sMax::Real, sMin::Real,
                    nEarlyStopPeriod::Integer=10)
    loss_prev = 1e8
    testErr_prev = 1e8

    s = sMax
    nSteps = 0
    nStepsEarlyStopCheck = 0
    W_earlyStop = copy(nn.W)

    while true
        if nSteps >= maxSteps
            println("Stopping: Maximum number of $maxSteps steps reached")
            break
        end

        W_prev = copy(nn.W)

        loss = trainingStep!(nn, x_train, y_train, s)
        nSteps += 1
        nStepsEarlyStopCheck += 1

        
        trainErr = relErrFunc(nn, x_train, y_train)
        println("Training step $nSteps finished: s=$s, loss=$loss, trainErr=$trainErr")
        if trainErr < tol
            println("Stopping: Training error $trainErr dropped below tolerance $tol.")
            break
        end

        # Early stopping check: If error increased after nEarlyStopPeriod steps, fall back to weights in last check
        if nStepsEarlyStopCheck >= nEarlyStopPeriod
            testErr = relErrFunc(nn, x_test, y_test)
            if testErr > testErr_prev
                println("Stopping: Early stopping because test error starts increasing.")
                nn.W = copy(W_earlyStop)
                break
            end
            testErr_prev = testErr

            W_earlyStop = copy(nn.W)
            nStepsEarlyStopCheck = 0
        end

        sRel = calcRelStepSize!(nn, W_prev, loss, loss_prev)
        s = min(max(sMin, s*sRel), sMax)
        loss_prev = loss    
    end

    println("Training end after $nSteps steps.")
end

module Losses

using LinearAlgebra

square_error(x, y) = 1/2*norm(x - y)^2
derx_square_error(x, y) = (x-y)

logeps = 1e-8
cross_entropy(x, y) = -sum(y.*log.(x .+ logeps) + (1 .- y).*log.(1 .- x .+ logeps))
derx_cross_entropy(x, y) = -y.*1 ./ (x .+ logeps) + (1 .- y) ./ (1 .- x .+ logeps)    

end

module Activations

relu(x) = max(0.0, x)
der_relu(x) = x < 0.0 ? 0.0 : 1.0

sigmoid(x) = 1 ./ (1 + exp.(-x))
der_sigmoid(x) = exp.(-x) ./ (1 .+ exp.(-x)).^2

end

end