#: Constraint set utilities

export no_constraints, zero_set, indicator, δ


# No constraints

struct NoConstraints{T,N}<:ProjectionableSet{T,N} end

no_constraints(T::DataType, N::Int64) = NoConstraints{T,N}()

project!(x::AbstractArray{T,N}, ::NoConstraints{T,N}, y::Array{T,N}) where {T,N} = (y .= x)

# Homogeneous constraints via masking grid function

struct ZeroSet{T,N}<:ProjectionableSet{T,N}
    mask::AbstractArray{Bool,N}
end

zeros_set(T::DataType, mask::AbstractArray{Bool,N}) = ZeroSet{T,N}(mask)

project!(x::AbstractArray{T,N}, C::ZeroSet{T,N}, y::AbstractArray{T,N}) where {T,N} = (y[C.mask] .= x[C.mask]; y[(!).(C.mask)] .= 0)

struct ZeroOperator{T,N}<:AbstractLinearOperator{T,N,N}
    C::ZeroSet{T,N}
end

zero_operator(C::ZeroSet{T,N}) where {T,N} = ZeroOperator{T,N}(C)

AbstractLinearOperators.domain_size(A::ZeroOperator) = size(A.C.mask)
AbstractLinearOperators.range_size(A::ZeroOperator) = domain_size(A)
AbstractLinearOperators.matvecprod(A::ZeroOperator{T,N}, u::AbstractArray{T,N}) where {T,N} = project(u, A.C)
AbstractLinearOperators.matvecprod_adj(A::ZeroOperator{T,N}, v::AbstractArray{T,N}) where {T,N} = A*v

# Sub-level sets with proximable functions

"""
Constraint set C = {x:g(x)<=ε}
"""
struct SubLevelSet{T,N}<:ProjectionableSet{T,N}
    g::ProximableFunction{T,N}
    ε::Real
end

Base.:≤(g::ProximableFunction{CT,N}, ε::T) where {T<:Real,N,CT<:RealOrComplex{T}} = SubLevelSet{CT,N}(g, ε)

project!(x::AbstractArray{T,N}, C::SubLevelSet{T,N}, y::AbstractArray{T,N}) where {T,N} = project!(x, C.ε, C.g, y)
project!(x::AbstractArray{T,N}, C::SubLevelSet{T,N}, y::AbstractArray{T,N}, opt::AbstractOptimizer) where {T,N} = project!(x, C.ε, C.g, y, opt)

# Indicator function

"""
Indicator function δ_C(x) = {0, if x ∈ C; ∞, otherwise} for convex sets C
"""
struct IndicatorFunction{T,N}<:ProximableFunction{T,N}
    C::ProjectionableSet{T,N}
end

indicator(C::ProjectionableSet{T,N}) where {T,N} = IndicatorFunction{T,N}(C)
δ(C::ProjectionableSet{T,N}) where {T,N} = IndicatorFunction{T,N}(C)

proxy!(y::AbstractArray{CT,N}, ::T, δ::IndicatorFunction{CT,N}, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = project!(y, δ.C, x)
proxy!(y::AbstractArray{CT,N}, ::T, δ::IndicatorFunction{CT,N}, x::AbstractArray{CT,N}, opt::AbstractOptimizer) where {T<:Real,N,CT<:RealOrComplex{T}} = project!(y, δ.C, x, opt)