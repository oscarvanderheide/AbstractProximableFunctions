module ConvexOptimizationUtils

using LinearAlgebra, AbstractLinearOperators, Roots

const RealOrComplex{T<:Real} = Union{T,Complex{T}}

include("./abstract_types.jl")
include("./optimization_utils.jl")
include("./linalg_utils.jl")
include("./differentiable_functions.jl")
include("./differentiable_functions_examples.jl")
include("./projectionable_sets.jl")
include("./projectionable_sets_examples.jl")
include("./proximable_functions.jl")
include("./proximable_functions_examples.jl")

end