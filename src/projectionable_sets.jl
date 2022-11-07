#: Examples of projectionable sets

export SublevelSet
export IndicatorFunction, indicator, δ
export ZeroSet, zero_set


# Sub-level sets with proximable functions

struct SublevelSet{T,N}<:AbstractProjectionableSet{T,N}
    fun::AbstractProximableFunction{T,N}
    level::Real
end

Base.:≤(g::AbstractProximableFunction{CT,N}, ε::T) where {T<:Real,N,CT<:RealOrComplex{T}} = SublevelSet{CT,N}(g, ε)

Base.in(x::AbstractArray{T,N}, C::SublevelSet{T,N}) where {T<:RealOrComplex,N} = C.fun(x) ≤ C.level

project!(x::AbstractArray{T,N}, C::SublevelSet{T,N}, options::AbstractArgminMethod, y::AbstractArray{T,N}) where {T,N} = project!(x, C.level, C.fun, options, y)


# Indicator function

struct IndicatorFunction{T,N}<:AbstractIndicatorFunction{T,N}
    set::AbstractProjectionableSet{T,N}
end

indicator(C::AbstractProjectionableSet{T,N}) where {T,N} = IndicatorFunction{T,N}(C)
δ(C::AbstractProjectionableSet{T,N}) where {T,N} = IndicatorFunction{T,N}(C)

get_set(δ::IndicatorFunction) = δ.set

proxy!(y::AbstractArray{CT,N}, ::T, δ::IndicatorFunction{CT,N}, options::AbstractArgminMethod, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = project!(y, δ.set, options, x)
project!(y::AbstractArray{CT,N}, ::T, δ::IndicatorFunction{CT,N}, options::AbstractArgminMethod, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = project!(y, δ.set, options, x)


# Homogeneous constraints via masking grid function

struct ZeroSet{T,N}<:AbstractProjectionableSet{T,N}
    is_zero::AbstractArray{Bool,N}
end

zero_set(T::DataType, is_zero::AbstractArray{Bool,N}) where N = ZeroSet{T,N}(is_zero)

Base.in(x::AbstractArray{T,N}, C::ZeroSet{T,N}) where {T,N} = all(x[C.is_zero] .== 0)

function project!(x::AbstractArray{T,N}, C::ZeroSet{T,N}, ::ExactArgmin, y::AbstractArray{T,N}) where {T,N}
    idx_nonzero = (!).(C.is_zero)
    y[idx_nonzero] .= x[idx_nonzero]
    y[C.is_zero] .= 0
    return y
end