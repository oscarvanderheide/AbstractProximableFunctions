module ConvexOptimizationUtils

using LinearAlgebra, Flux, AbstractLinearOperators, Roots

const RealOrComplex{T<:Real} = Union{T,Complex{T}}

include("./abstract_types.jl")
include("./type_utils.jl")
include("./optimization_utils.jl")
include("./differentiable_functions.jl")
include("./proximable_functions.jl")
include("./weighted_proximable_functions.jl")

end