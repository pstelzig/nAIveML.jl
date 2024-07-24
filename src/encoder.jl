"""
Simple and self-contained implementation of a basic autoencoder, and a naive 
variational autoencoder. Uses the NeuralNetworks module of nAIveML.jl. 

Copyright Dr. Philipp Emanuel Stelzig, 2023-2024
"""

module Encoder

include("neuralnetworks.jl")

import .NeuralNetworks as nn

using LinearAlgebra
using Statistics
using Distributions

abstract type AbstractEncoder end

mutable struct AutoEncoder <: AbstractEncoder
    dIn::Integer
    encIdx::Integer
    encDim::Integer
    network::nn.Network

    function AutoEncoder(d, nInLayersEnc, nInLayersDec, actFunc, dactFunc)
        if d != nInLayersDec[end]
            error("Output dimension of decoder $(nInLayersDec[end]) and input dimension $d do not match.")
        end

        nInLayers = [nInLayersEnc; nInLayersDec]
        lEnc = length(nInLayersEnc)
        loss = nn.Losses.square_error
        derx_loss = nn.Losses.derx_square_error

        network = nn.Network(d, nInLayers, actFunc, dactFunc, loss, derx_loss)

        new(d, lEnc, nInLayersEnc[end], network)
    end
end

mutable struct VarAutoEncoder <: AbstractEncoder
    dIn::Integer
    encIdx::Integer
    encDim::Integer  
    network::nn.Network

    mu::Vector{<:Real}
    Cov::Matrix{<:Real}
    encDistrib

    function VarAutoEncoder(d, nInLayersEnc, nInLayersDec, actFunc, dactFunc)
        if d != nInLayersDec[end]
            error("Output dimension of decoder $(nInLayersDec[end]) and input dimension $d do not match.")
        end

        nInLayers = [nInLayersEnc; nInLayersDec]
        lEnc = length(nInLayersEnc)
        loss = nn.Losses.square_error
        derx_loss = nn.Losses.derx_square_error

        network = nn.Network(d, nInLayers, actFunc, dactFunc, loss, derx_loss)

        new(d, lEnc, nInLayersEnc[end], network)
    end    
end

function encode(enc::AbstractEncoder, x_in::Array{<:Real})
    X = nn.forward(enc.network, x_in)

    return X[enc.encIdx][2:end]
end

function decode(enc::AbstractEncoder, x_enc::Array{<:Real})
    x_enc_ext = [1; x_enc]

    X = nn.forward(enc.network, x_enc_ext, enc.encIdx)

    return X[enc.network.nLayers][2:end]
end

function decode(enc::VarAutoEncoder, x_enc::Array{<:Real}, alpha::Real) 
    x_enc_var = x_enc + alpha*(rand(enc.encDistrib) - enc.mu)

    x_enc_var_ext = [1; x_enc_var]

    X = nn.forward(enc.network, x_enc_var_ext, enc.encIdx)

    return X[enc.network.nLayers][2:end]    
end

function generate(enc::VarAutoEncoder, alpha::Real=1.0)
    rand_enc = enc.mu + alpha*(rand(enc.encDistrib) - enc.mu)

    return decode(enc, rand_enc)
end

function trainingStep!(enc::AbstractEncoder, x::Matrix{<:Real}, h::Real)
    return nn.trainingStep!(enc.network, x, x, h)
end

function train!(enc::AutoEncoder,
    trainData::Matrix{<:Real},
    testData::Matrix{<:Real},
    maxSteps::Integer, tol::Real, sMax::Real, sMin::Real, nEarlyStopPeriod::Integer=10)
    
    # nn.forward returns the output of all hidden layers as a dict starting with index zero for the input layer. Also, its first entry is a bias term
    # The output of the output layer is nn.forward(netw, v)[netw.nLayers][2:end]
    relErr(netw::nn.Network, x::Matrix{Float64}, y::Matrix{Float64}) = sum([norm(nn.forward(netw, x[:,j])[netw.nLayers][2:end] - y[:,j])^2 for j in 1:size(x)[2]]) / (2*size(x)[2])

    nn.train!(enc.network, trainData, trainData, testData, testData, relErr, maxSteps, tol, sMax, sMin, nEarlyStopPeriod)
end

function train!(enc::VarAutoEncoder,
    trainData::Matrix{<:Real},
    testData::Matrix{<:Real},
    maxSteps::Integer, tol::Real, sMax::Real, sMin::Real, nEarlyStopPeriod::Integer=10)

    # nn.forward returns the output of all hidden layers as a dict starting with index zero for the input layer. Also, its first entry is a bias term
    # The output of the output layer is nn.forward(netw, v)[netw.nLayers][2:end]
    relErr(netw::nn.Network, x::Matrix{Float64}, y::Matrix{Float64}) = sum([norm(nn.forward(netw, x[:,j])[netw.nLayers][2:end] - y[:,j])^2 for j in 1:size(x)[2]]) / (2*size(x)[2])

    nn.train!(enc.network, trainData, trainData, testData, testData, relErr, maxSteps, tol, sMax, sMin, nEarlyStopPeriod)

    nTrain = size(trainData)[2]
    trainEnc = zeros(enc.encDim, nTrain)

    for i in 1:nTrain
        trainEnc[:,i] = encode(enc, trainData[:,i])
    end

    enc.mu = vec(mean(trainEnc, dims=2))
    enc.Cov = cov(transpose(trainEnc))

    enc.encDistrib = MvNormal(enc.mu, enc.Cov)
end

end