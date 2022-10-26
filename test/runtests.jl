using ConvexOptimizationUtils, Test

@testset "ConvexOptimizationUtils.jl" begin
    include("./test_differentiable.jl")
    include("./test_constraints.jl")
    include("./test_proximable_functions.jl")
    include("./test_optimization_utils.jl")
end