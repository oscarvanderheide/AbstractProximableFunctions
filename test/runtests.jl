using ConvexOptimizationUtils, Test

@testset "ConvexOptimizationUtils.jl" begin
    include("./test_differentiable.jl")
    include("./test_proximable_functions.jl")
end