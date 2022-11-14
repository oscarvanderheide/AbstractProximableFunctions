module AbstractProximableFunctions

using LinearAlgebra, AbstractLinearOperators, Roots

const RealOrComplex{T<:Real} = Union{T,Complex{T}}

include("./abstract_types.jl")
include("./abstract_type_utils.jl")
include("./argmin.jl")
include("./diff_functions.jl")
include("./proximable_functions.jl")
# include("./projectionable_sets.jl")
# include("./norm_utils.jl")
# include("./linalg_utils.jl")

end