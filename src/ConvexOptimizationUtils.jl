module ConvexOptimizationUtils

using LinearAlgebra, AbstractLinearOperators, Roots

const RealOrComplex{T<:Real} = Union{T,Complex{T}}

include("./abstract_types.jl")
include("./misc_utils.jl")
include("./linalg_utils.jl")
include("./argmin.jl")
include("./differentiable_functions.jl")
include("./projectionable_sets.jl")
include("./proximable_functions.jl")
include("./proximable_norms.jl")

end