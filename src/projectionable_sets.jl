#: Projectionable set utilities

export ZeroSet, zero_set
export SublevelSet
export IndicatorFunction, indicator, δ


# Homogeneous constraints via masking grid function

struct ZeroSet{T,N}<:ProjectionableSet{T,N}
    is_zero::AbstractArray{Bool,N}
end

zero_set(T::DataType, is_zero::AbstractArray{Bool,N}) where N = ZeroSet{T,N}(is_zero)

Base.in(x::AbstractArray{T,N}, C::ZeroSet{T,N}) where {T,N} = all(x[C.is_zero] .== 0)

function project!(x::AbstractArray{T,N}, C::ZeroSet{T,N}, y::AbstractArray{T,N}; optimizer::Union{Nothing,Optimizer}=nothing) where {T,N}
    idx_nonzero = (!).(C.is_zero)
    y[idx_nonzero] .= x[idx_nonzero]
    y[C.is_zero] .= 0
    return y
end


# Sub-level sets with functions

struct SublevelSet{T,N,FT<:MinimizableFunction{T,N}}<:ProjectionableSet{T,N}
    fun::FT
    level::Real
end

Base.:≤(g::FT, ε::T) where {T<:Real,N,CT<:RealOrComplex{T},FT<:MinimizableFunction{CT,N}} = SublevelSet{CT,N,FT}(g, ε)

Base.in(x::AbstractArray{T,N}, C::SublevelSet{T,N,FT}) where {T<:RealOrComplex,N,FT<:MinimizableFunction{T,N}} = C.fun(x) ≤ C.level

project!(x::AbstractArray{T,N}, C::SublevelSet{T,N,FT}, y::AbstractArray{T,N}; optimizer::Union{Nothing,Optimizer}=nothing) where {T,N,FT<:ProximableFunction{T,N}} = project!(x, C.level, C.fun, y; optimizer=optimizer)


# Indicator function

struct IndicatorFunction{T,N}<:ProximableFunction{T,N}
    C::ProjectionableSet{T,N}
end

indicator(C::ProjectionableSet{T,N}) where {T,N} = IndicatorFunction{T,N}(C)
δ(C::ProjectionableSet{T,N}) where {T,N} = IndicatorFunction{T,N}(C)

(δC::IndicatorFunction{CT,N})(x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = (x ∈ δC.C) ? T(0) : T(Inf)

proxy!(y::AbstractArray{CT,N}, ::T, δ::IndicatorFunction{CT,N}, x::AbstractArray{CT,N}; optimizer::Union{Nothing,Optimizer}=nothing) where {T<:Real,N,CT<:RealOrComplex{T}} = project!(y, δ.C, x; optimizer=optimizer)
project!(y::AbstractArray{CT,N}, ::T, δ::IndicatorFunction{CT,N}, x::AbstractArray{CT,N}; optimizer::Union{Nothing,Optimizer}=nothing) where {T<:Real,N,CT<:RealOrComplex{T}} = project!(y, δ.C, x; optimizer=optimizer)