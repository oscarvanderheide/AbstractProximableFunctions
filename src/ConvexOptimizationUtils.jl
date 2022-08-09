module ConvexOptimizationUtils

# Modules
using LinearAlgebra, Flux, AbstractLinearOperators, Roots

const RealOrComplex{T<:Real} = Union{T,Complex{T}}

include("./abstract_types.jl")
include("./type_utils.jl")
include("./optimization_utils.jl")
include("./differentiable_examples.jl")
include("./proximable_examples.jl")

end