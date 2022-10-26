using LinearAlgebra, ConvexOptimizationUtils, AbstractLinearOperators, Test, Random
Random.seed!(123)
include("test_utils.jl")

# Random input
T = Float64
x = randn(T, 2, 3, 4)
mask = x .> 0

# Zero set
C = zero_set(T, mask)

# Gradient test
g = indicator(C)
fun = proxy_objfun(1.0, g)
y = randn(T, 2, 3, 4)
test_grad(fun, y; step=1e-4, rtol=1e-3)