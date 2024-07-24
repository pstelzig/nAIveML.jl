"""
Some basic experiments to check workings of nAIveML.jl. 
Part of nAIveML.jl. 

Copyright Dr. Philipp Emanuel Stelzig, 2023-2024
"""

include("../src/naiveml.jl")
import .nAIveML
using Plots
using LinearAlgebra
using Random

# Experiment on gradient gradient descent #####################################
parabula = x-> x^2
der_parabula = x -> 2*x
x_min, f_min = nAIveML.Optim.gdunconstr(parabula, der_parabula, 5.0, -100, 100, 1.25, 0.8, 1e-3, 50)
x_min, f_min = nAIveML.Optim.gdunconstr(parabula, der_parabula, 5.0, 0.0, 100, 1.25, 0.8, 1e-3, 50)
x_min, f_min = nAIveML.Optim.gdunconstr(sin, cos, 5.0, -10.0, 10.0, 1.25, 0.8, 1e-3, 50)
x_min, f_min = nAIveML.Optim.gdunconstr(log, x->1/x, 5.0, 1e-5, 10.0, 1.25, 0.8, 1e-5, 120)

# Minimizes sum((x.-[1; 2]).^2) = 1/2 * x'*2*I*x - 2*[1; 2]'*x + [1; 2]'*[1; 2] 
# under the constraint 3*x[1] - x[2] = 1.0. Has optimal solution [1; 2]
parabula2 = x -> (x.-[1; 2])'*I*(x.-[1; 2])
der_parabula2 = x -> 2*(x.-[1; 2])
x_0 = ones(2)
x_min, f_min = nAIveML.Optim.gdlinconstr(parabula2, der_parabula2, [3.0 -1.0], [1.0], x_0, 0.7, 0.8, 1e-7, 200)
x_min, f_min = nAIveML.Optim.qplinconstr([2.0 0; 0.0 2.0], -2*[1.0; 2.0], [3.0 -1.0; -3.0 1.0], [1.0; -1.0], 0.7, 0.8, 1e-4, 200)
x_min, f_min = nAIveML.Optim.gdfullconstr(parabula2, nothing, x -> 3.0*x[1] -1.0*x[2] - 1.0, [5.0, 14.0], 1e-5, 200)

# Minimizes x[1]^2 + x[2]^2 = x'x
# under the constraint x[2] = 1 + x[1]^2 which has the optimal solution [0; 1]
parabula3 = x -> x'*x
der_parabula3 = x -> x
constr1 = x -> [1 + x[1]^2 - x[2]]
der_constr1 = x -> [2*x[1] -1]
x_0 = [1; 2]

x_min, f_min = nAIveML.Optim.gdfullconstr(parabula3, nothing, constr1, x_0, 1e-3, 200)
x_min, f_min = nAIveML.Optim.gdnonlinconstr(parabula3, der_parabula3, constr1, der_constr1, x_0, 0.25, 0.8, 1e-5, 200)

# Minimizes x[1]^2 + x[2]^2 + x[3]^2 = x'x
# under the constraint x[3] = 1 + x[1]^2 + x[2]^2 and x[2] = 2 + x[1]^2 which has the optimal solution [0; 2; 5]
parabula4 = x -> x'*x
der_parabula4 = x -> x
constr2 = x -> [1 + x[1]^2 + x[2]^2 - x[3]; 2 + x[1]^2 - x[2]]
der_constr2 = x -> [2*x[1] 2*x[2] -1; 2*x[1] -1 0]
x_0 = [1; 3; 11]

x_min, f_min = nAIveML.Optim.gdfullconstr(parabula4, nothing, constr2, x_0, 1e-3, 200)
x_min, f_min = nAIveML.Optim.gdnonlinconstr(parabula4, der_parabula4, constr2, der_constr2, x_0, 0.25, 0.8, 1e-5, 200)

# Experiment on linear regression #############################################
# Use linear regression to reconstruct the linear function y=-7x + 2 + noise(x)
nData = 500
xData = rand(1,nData)
yData = -7*xData .+ 2.0 + 1.5*rand(1,nData)

(W,c) = nAIveML.Regression.Linear(xData, yData)
(W_R,c_R) = nAIveML.Regression.Ridge(xData, yData, 2.0)
yLinReg = W*xData .+ c
yRidgeReg = W_R*xData .+ c_R

println("(W,c) created with (-7.0,2.0) and noise. ")
println("Linear regression: (W,c)=($W,$c)")
println("Ridge regression: (W_R,c_R)=($W_R,$c_R)")

plt = scatter(xData[1,:], yData[1,:], label="original")
plot!(plt, xData[1,:], yLinReg[1,:], label="linear regression")
plot!(plt, xData[1,:], yRidgeReg[1,:], label="Ridge regression")
display(plt)

# Experiment on dimension reduction ###########################################
# Use pca to get the best approximating basis for the points (x, -7x + 2 + noise(x))
nData = 500
xVals = rand(1,nData)
yVals = -7*xVals .+ 2.0 + 1.5*rand(1,nData)
xData = [xVals;yVals]
println("Data main direction = [-7.0; 1.0]")

pca_vals_free, pca_vecs_free, pca_mean = nAIveML.DimensionReduction.pca(xData, true)
println("pca_vecs_free = $pca_vecs_free, mean=$pca_mean")
println("PCA main direction (mean value free)=[$(pca_vecs_free[:,1][2]/pca_vecs_free[:,1][1]), 1.0]")

pca_vals, pca_vecs = nAIveML.DimensionReduction.pca(xData, false)
println("pca_vecs = $pca_vecs")
println("PCA main direction=[$(pca_vecs[:,1][2]/pca_vecs[:,1][1]), 1.0]")

# Experiment on logistic regression ###########################################
# Split unit square into 3 disjoint sets U1, U2, U3: 
# U1: Triangle: below and including y=-x+1/2
# U2: Area between Triangle U1 and below and including parabula y=(1-x)^2
# U3: Area above parabula

nTrain = 300
xTrain = rand(2, nTrain)
yTrain = zeros(Int64, nTrain)

function binIntoBands(x::Vector{Float64})
    if x[2] <= -x[1] + 0.5
        return 0
    elseif (x[2] > -x[1] + 0.5) && (x[2] <= (1-x[1])^2)
        return 1
    else
        return 2
    end
end

for j in 1:nTrain
    yTrain[j] = binIntoBands(xTrain[:,j])
end

logit = nAIveML.Regression.Logistic(2, 3)
nAIveML.Regression.train!(logit, xTrain, yTrain, 1e-3, 500)

# Test with random points
nTest = 200
xTest = rand(2, nTest)
matches = 0
for j in 1:nTest
    p = nAIveML.Regression.forward(logit, xTest[:,j])
    if (argmax(p) == binIntoBands(xTest[:,j])+1)
        global matches += 1
    end
end
print("\"Accuracy of logistic regression\"=$(matches/nTest)")

# Experiment on neural networks ###############################################
# Use a neural network to approximate the function f : \mathbb{R}^3 -> \mathbb{R}^2
# f(x) := [1.2*(x[1] - 0.2)^2 + 0.1, x[1]*(x[2] + 1.6) - 1.1]
theta(x) = tanh(x)
dtheta(x) = 1 - (tanh(x))^2
errdist(x, y) = 1/2*norm(x - y)^2
derrdistx(x, y) = (x-y)

ffnn = nAIveML.NeuralNetworks.Network(3, [3, 8, 2], theta, dtheta, errdist, derrdistx)
X = nAIveML.NeuralNetworks.forward(ffnn, [0.2, 0.3, 0.4])
Delta = nAIveML.NeuralNetworks.backward(ffnn, X, [0, 1])

xData = rand(3,100)
yData = [1.2*(xData[1,:]' .- 0.2).^2 .+ 0.1; xData[1,:]' .* (xData[2,:]' .+ 1.6) .- 1.1]

for k in 1:100
    err1 = nAIveML.NeuralNetworks.trainingStep!(ffnn, xData, yData, 0.5)
    println("err1=$err1")
end

# Experiment on auto-encoding #################################################
# Use auto-encoding to learn the relevant parameters when encoding the 
# family of curvse f_k : y = sin(x + s_k) when the parameter s_k ranges 
# from 0, 1/N, 2/N,..., 1

nData = 1000
nTrain = 200
d_in = 50
d_enc = 10
xVals = collect(range(0.0, 1.0, length=d_in))
xData = zeros((d_in, nData))
for j in 1:nData
    s = j/nData
    xData[:,j] = sin.(s .+ xVals)
end

perm = shuffle(1:nData)
xData = copy(xData[:, perm])

enc = nAIveML.Encoder.VarAutoEncoder(d_in, [40, d_enc], [40, d_in], nAIveML.NeuralNetworks.Activations.sigmoid, nAIveML.NeuralNetworks.Activations.der_sigmoid)
nAIveML.Encoder.train!(enc, xData[:, 1:nTrain], xData[:, nTrain+1:end], 100, 1e-4, 0.1, 0.0001, nData)

x_example = sin.(xVals .+ 0.373)
x_enc = nAIveML.Encoder.encode(enc, x_example)
x_dec = nAIveML.Encoder.decode(enc, x_enc, 0.05)
plt = plot(xVals, x_example)
plot!(plt, xVals, x_dec)
display(plt)

# Experiments on Support Vector Machines ######################################
# Classifies two point clusters taken from the unit square and shifted, s.t.
# the "upper" cluster points with labels +1 are all above the line y=0.3, and
# the "lower" cluster points with labels -1 are all below the line y=-0.2, and
nData = 500
xDataUpper = rand(2, nData)
xDataUpper = [xDataUpper[1, :]'; xDataUpper[2, :]' .+ 0.3]
yDataUpper = ones(Integer, nData)

xDataLower = rand(2, nData)
xDataLower = [xDataLower[1, :]'; xDataLower[2, :]' .- 1.2]
yDataLower = -ones(Integer, nData)

xData = hcat(xDataUpper, xDataLower)
yData = [yDataUpper; yDataLower]
perm = shuffle(1:2*nData)
xData = copy(xData[:,perm])
yData = copy(yData[perm])

# Linear hard margin SVM
svm = nAIveML.SupportVectorMachines.LinHardMargin()
nAIveML.SupportVectorMachines.train!(svm, xData, yData, 1e-3, 500)

plt = scatter(xDataUpper[1,:], xDataUpper[2,:], label="y=1", color="red")
scatter!(plt, xDataLower[1,:], xDataLower[2,:], label="y=-1", color="blue")
xPlt = range(0,1,51)
plot!(plt, xPlt,  (1 .- svm.b .- svm.w[1]*xPlt)./svm.w[2], label="upper SVM margin")
plot!(plt, xPlt, (-1 .- svm.b .- svm.w[1]*xPlt)./svm.w[2], label="lower SVM margin")
display(plt)

# Kernel hard margin SVM
krnl = (x,y) -> x'*y
kersvm = nAIveML.SupportVectorMachines.KernelHardMargin(krnl)
nAIveML.SupportVectorMachines.train!(kersvm, xData, yData, 1e-3, 500)

yComp = [nAIveML.SupportVectorMachines.forward(kersvm, xData[:,j]) for j in 1:2*nData]
upperIdx = [j for j in 1:2*nData if yComp[j] > 0]
lowerIdx = [j for j in 1:2*nData if yComp[j] < 0]
plt2 = scatter(xData[1,upperIdx], xData[2,upperIdx], label="y_comp=1", color="green")
scatter!(plt2, xData[1,lowerIdx], xData[2,lowerIdx], label="y_comp=-1", color="yellow")
display(plt2)

# Classifies two point clusters. Constructed from a random point cloud in the unit square.
# The "inner" cluster points with labels +1 are those within the circle at origin and r=0.4.
# The "outer" cluster points with labels -1 are those between the origin-centered circles of r=0.6 and r=1.0.
xData = rand(2, 1000)

rInner = 0.5
idxInner = [j for j in 1:size(xData)[2] if norm(xData[:,j]) <= rInner]
xDataInner = xData[:,idxInner]
yDataInner = ones(Integer, size(xDataInner)[2])

rOuter1 = 0.55
rOuter2 = 1.0
idxOuter = [j for j in 1:size(xData)[2] if norm(xData[:,j]) >= rOuter1 && norm(xData[:,j]) <= rOuter2]
xDataOuter = xData[:,idxOuter]
yDataOuter = -ones(Integer, size(xDataInner)[2])

xData = hcat(xDataInner, xDataOuter)
yData = [yDataInner; yDataOuter]
nData = length(yData)
perm = shuffle(1:nData)
xData = copy(xData[:,perm])
yData = copy(yData[perm])

# Using a (more or less) nontrivial kernel
phi(x) = [x[1]; x[2]; x[1]^2 + x[2]^2]
krnl2(x,y) = x[1]*y[1] + x[2]*y[2] + (x[1]^2 + x[2]^2)*(y[1]^2 + y[2]^2)

kersvm = nAIveML.SupportVectorMachines.KernelHardMargin(krnl2)
nAIveML.SupportVectorMachines.train!(kersvm, xData, yData, 1e-3, 500)

yComp = [nAIveML.SupportVectorMachines.forward(kersvm, xData[:,j]) for j in 1:nData]
innerIdx = [j for j in 1:nData if yComp[j] > 0]
outerIdx = [j for j in 1:nData if yComp[j] < 0]
plt3 = scatter(xData[1,innerIdx], xData[2,innerIdx], label="y_comp=1", color="green")
scatter!(plt3, xData[1,outerIdx], xData[2,outerIdx], label="y_comp=-1", color="yellow")
display(plt3)
