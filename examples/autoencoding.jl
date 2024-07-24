"""
Using nAIveML.jl autoencoders on MNIST data to generate some new numbers.
Part of nAIveML.jl. 

Copyright Dr. Philipp Emanuel Stelzig, 2023-2024
"""

include("../src/naiveml.jl")

import .nAIveML.Encoder as ae
import .nAIveML.NeuralNetworks as nn
using DelimitedFiles
using Random
using Plots
using Images

this_folder = splitdir(@__FILE__)[1]
println("this_folder=$(this_folder)")
# MNIST data ##################################################################
mnist_pxdata = DelimitedFiles.readdlm("$(this_folder)/mnist_pxdata_short.txt", ',', Int, '\n')

xData = copy(transpose(mnist_pxdata / 255.0))
nData = size(xData)[2]

# Do random permutation on data
perm = shuffle(1:nData)
xData = copy(xData[:,perm])

# Use auto-encoding to reduce input data dimension ############################
nPx = 28
nDimRed = 24

nTrainData = floor(Int64, 0.1*nData)
nTestData = floor(Int64, 0.025*nTrainData)

trainData = xData[:, 1:nTrainData]
testData = xData[:, nTrainData+1:nTrainData + 1 + nTestData]

enc = ae.VarAutoEncoder(nPx*nPx, [400, 200, nDimRed], [400, nPx*nPx], nn.Activations.sigmoid, nn.Activations.der_sigmoid)
ae.train!(enc, trainData, testData, 100, 1e-2, 0.1, 0.001, 101)

# Plotting original and encoding side by side in one image
function plot_single(img_flat, nx, ny)
    px = zeros(nx, ny)
    for r in 1:ny
        px[r, 1:nx] = img_flat[(r-1)*ny+1:(r-1)*ny+nx]
    end
    plt = plot(colorview(Gray, px))
    display(plt)
end

function plot_side_by_side(img_flat1, img_flat2, nx, ny)
    px = zeros(ny, 2*nx)
    for r in 1:ny
        px[r, 1:nx] = img_flat1[(r-1)*ny+1:(r-1)*ny+nx]
        px[r, nx+1:nx+nx] = img_flat2[(r-1)*ny+1:(r-1)*ny+nx]
    end
    plt = plot(colorview(Gray, px))
    display(plt)
end

for v in 1:40
    # Choose random picture
    img_idx = mod(rand(Int), nData) + 1

    x_orig = xData[:, img_idx]
    x_enc = ae.encode(enc, x_orig)
    x_dec = ae.decode(enc, x_enc, 0.05)

    bright_dec = maximum(x_dec)
    plot_side_by_side(x_orig, x_dec ./ bright_dec, nPx, nPx)
end
