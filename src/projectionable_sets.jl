#: Projectionable set utilities

export SublevelSet
export IndicatorFunction, indicator, δ


# Sub-level sets with proximable functions

struct SublevelSet{T,N,FT<:AbstractMinimizableFunction{T,N}}<:AbstractProjectionableSet{T,N}
    fun::FT
    level::Real
end

Base.:≤(g::FT, ε::T) where {T<:Real,N,CT<:RealOrComplex{T},FT<:AbstractMinimizableFunction{CT,N}} = SublevelSet{CT,N,FT}(g, ε)

Base.in(x::AbstractArray{T,N}, C::SublevelSet{T,N,FT}) where {T<:RealOrComplex,N,FT<:AbstractMinimizableFunction{T,N}} = C.fun(x) ≤ C.level

project!(x::AbstractArray{T,N}, C::SublevelSet{T,N,FT}, y::AbstractArray{T,N}; optimizer::Union{Nothing,AbstractConvexOptimizer}=nothing) where {T,N,FT<:AbstractProximableFunction{T,N}} = project!(x, C.level, C.fun, y; optimizer=optimizer)


# Indicator function

struct IndicatorFunction{T,N}<:AbstractProximableFunction{T,N}
    C::AbstractProjectionableSet{T,N}
end

indicator(C::AbstractProjectionableSet{T,N}) where {T,N} = IndicatorFunction{T,N}(C)
δ(C::AbstractProjectionableSet{T,N}) where {T,N} = IndicatorFunction{T,N}(C)

(δC::IndicatorFunction{CT,N})(x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = (x ∈ δC.C) ? T(0) : T(Inf)

proxy!(y::AbstractArray{CT,N}, ::T, δ::IndicatorFunction{CT,N}, x::AbstractArray{CT,N}; optimizer::Union{Nothing,AbstractConvexOptimizer}=nothing) where {T<:Real,N,CT<:RealOrComplex{T}} = project!(y, δ.C, x; optimizer=optimizer)
project!(y::AbstractArray{CT,N}, ::T, δ::IndicatorFunction{CT,N}, x::AbstractArray{CT,N}; optimizer::Union{Nothing,AbstractConvexOptimizer}=nothing) where {T<:Real,N,CT<:RealOrComplex{T}} = project!(y, δ.C, x; optimizer=optimizer)