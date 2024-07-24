"""
nAIveML.jl contains simple and self-contained implementations of some basic 
machine learning and artificial intelligence techniques for educational
purposes. 

Copyright Dr. Philipp Emanuel Stelzig, 2023-2024
"""

module nAIveML
include("regression.jl")
include("neuralnetworks.jl")
include("optim.jl")
end