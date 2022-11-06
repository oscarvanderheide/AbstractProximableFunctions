#: Examples of projectionable sets

export ZeroSet, zero_set


# Homogeneous constraints via masking grid function

struct ZeroSet{T,N}<:AbstractProjectionableSet{T,N}
    is_zero::AbstractArray{Bool,N}
end

zero_set(T::DataType, is_zero::AbstractArray{Bool,N}) where N = ZeroSet{T,N}(is_zero)

Base.in(x::AbstractArray{T,N}, C::ZeroSet{T,N}) where {T,N} = all(x[C.is_zero] .== 0)

function project!(x::AbstractArray{T,N}, C::ZeroSet{T,N}, ::ExactArgMin, y::AbstractArray{T,N}) where {T,N}
    idx_nonzero = (!).(C.is_zero)
    y[idx_nonzero] .= x[idx_nonzero]
    y[C.is_zero] .= 0
    return y
end