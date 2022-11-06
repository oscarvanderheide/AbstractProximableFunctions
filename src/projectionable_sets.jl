#: Projectionable set utilities

export SublevelSet
export IndicatorFunction, indicator, δ


# Sub-level sets with proximable functions

struct SublevelSet{T,N}<:AbstractProjectionableSet{T,N}
    fun::AbstractProximableFunction{T,N}
    level::Real
end

Base.:≤(g::AbstractProximableFunction{CT,N}, ε::T) where {T<:Real,N,CT<:RealOrComplex{T}} = SublevelSet{CT,N}(g, ε)

Base.in(x::AbstractArray{T,N}, C::SublevelSet{T,N}) where {T<:RealOrComplex,N} = C.fun(x) ≤ C.level

project!(x::AbstractArray{T,N}, C::SublevelSet{T,N}, options::AbstractArgMinOptions, y::AbstractArray{T,N}) where {T,N} = project!(x, C.level, C.fun, options, y)


# Indicator function

struct IndicatorFunction{T,N}<:AbstractProximableFunction{T,N}
    C::AbstractProjectionableSet{T,N}
end

indicator(C::AbstractProjectionableSet{T,N}) where {T,N} = IndicatorFunction{T,N}(C)
δ(C::AbstractProjectionableSet{T,N}) where {T,N} = IndicatorFunction{T,N}(C)

fun_eval(δC::IndicatorFunction{CT,N}, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = (x ∈ δC.C) ? T(0) : T(Inf)

proxy!(y::AbstractArray{CT,N}, ::T, δ::IndicatorFunction{CT,N}, options::AbstractArgMinOptions, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = project!(y, δ.C, options, x)
project!(y::AbstractArray{CT,N}, ::T, δ::IndicatorFunction{CT,N}, options::AbstractArgMinOptions, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = project!(y, δ.C, options, x)