module ConvexOptimizationUtils

using LinearAlgebra, AbstractLinearOperators, Roots

const RealOrComplex{T<:Real} = Union{T,Complex{T}}

include("./abstract_types.jl")
include("./linalg_utils.jl")
include("./differentiable_functions.jl")
include("./projectionable_sets.jl")
include("./proximable_functions.jl")
include("./weighted_proximable_functions.jl")
include("./prox_plus_indicator.jl")
include("./optimization_utils.jl")

end