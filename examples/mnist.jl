"""
Using the nAIveML.jl implementation of feed forward neural networks 
to classify MNIST data in a naive way. 
Part of nAIveML.jl. 

Copyright Dr. Philipp Emanuel Stelzig, 2023-2024
"""

include("../src/naiveml.jl")
import .nAIveML.NeuralNetworks as nn
import .nAIveML.DimensionReduction as dr
using DelimitedFiles
using Plots
using Images
using Random

this_folder = splitdir(@__FILE__)[1]
println("this_folder=$(this_folder)")
# MNIST data ##################################################################
mnist_pxdata = DelimitedFiles.readdlm("$(this_folder)/mnist_pxdata_short.txt", ',', Int, '\n')
mnist_labels = DelimitedFiles.readdlm("$(this_folder)/mnist_labels_short.txt", ',', Int, '\n')

xData = copy(transpose(mnist_pxdata / 255.0))
nData = size(xData)[2]
yData = zeros(10, nData)
for i in 1:nData
    yData[mnist_labels[i]+1, i] = 1
end

# Do random permutation on data
perm = shuffle(1:nData)
xData = copy(xData[:,perm])
yData = copy(yData[:,perm])

# Use PCA to reduce input data dimension from 28*28 to 50
_, pca_vecs, pca_mean = dr.pca(xData, true)

# Plotting the first 20 PCA vectors
nPx = 28
img_px = zeros(nPx,nPx)
for v in 1:20
    for r in 1:nPx
        bright_max = maximum(pca_vecs[:,v])
        img_px[r,:] = pca_vecs[((r-1)*nPx+1):((r-1)*nPx+nPx), v] ./ bright_max
    end    
    plt = plot(colorview(Gray, img_px))
    display(plt)
end

nDimRed = 70
xData = transpose(pca_vecs[:,1:nDimRed])*xData


# Define activation and loss function
if false # Set to true for Mean Square Error as loss and ReLU activation
    theta(x) = nn.Activations.relu(x)
    dtheta(x) = nn.Activations.der_relu(x)
    errdist(x, y) = nn.Losses.square_error(x, y)
    derrdistx(x, y) = nn.Losses.derx_square_error(x,y)
else # Cross entropy loss and sigmoid activation
    theta(x) = nn.Activations.sigmoid(x)
    dtheta(x) = nn.Activations.der_sigmoid(x)
    logeps = 1e-8
    errdist(x, y) = nn.Losses.cross_entropy(x, y)
    derrdistx(x, y) = nn.Losses.derx_cross_entropy(x, y)
end

# Set up neural network
xDim = size(xData)[1]
ffnn = nn.Network(xDim, [500, 200, 10], theta, dtheta, errdist, derrdistx)

nTrainData = floor(Int64, 6*nData/8)

trainData = xData[:, 1:nTrainData]
trainLabels = yData[:, 1:nTrainData]
testData = xData[:, nTrainData+1:nData]
testLabels = yData[:, nTrainData+1:nData]

function classicationErr(netw::nn.Network, x::Matrix{Float64}, y::Matrix{Float64})
    err_count = 0
    nSamples = size(x)[2]
    for i in 1:nSamples
        pred = argmax(nn.forward(netw, x[:,i])[3][2:end])
        act = argmax(y[:,i])
        if pred != act
           err_count += 1
        end
    end

    return err_count / nSamples
end

nSteps = 200
nn.train!(ffnn, trainData, trainLabels, testData, testLabels, classicationErr, nSteps, 1e-7, 0.05, 0.001)

rel_err = classicationErr(ffnn, testData, testLabels)
println("\"Number of validation datasets\"=$(nData - nTrainData)")
println("\"Accuracy\"=$(1.0 - rel_err)")
