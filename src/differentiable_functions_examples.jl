#: Examples of differentiable functions

export LeastSquaresMisfit, leastsquares_misfit


# Least-squares misfit

struct LeastSquaresMisfit{T,N1,N2}<:AbstractDifferentiableFunction{T,N1}
    linear_operator::AbstractLinearOperator{T,N1,N2}
    known_term::AbstractArray{T,N2}
end

leastsquares_misfit(A::AbstractLinearOperator{T,N1,N2}, y::AbstractArray{T,N2}) where {T,N1,N2} = LeastSquaresMisfit{T,N1,N2}(A, y)

fun_eval(f::LeastSquaresMisfit{T,N1,N2}, x::AbstractArray{T,N1}) where {T<:RealOrComplex,N1,N2} = norm(f.linear_operator*x-f.known_term)^2/2

function grad_eval!(f::LeastSquaresMisfit{T,N1,N2}, x::AbstractArray{T,N1}, gradient::AbstractArray{T,N1}) where {T<:RealOrComplex,N1,N2}
    r = f.linear_operator*x-f.known_term
    gradient .= f.linear_operator'*r
    return gradient
end

function fungrad_eval!(f::LeastSquaresMisfit{T,N1,N2}, x::AbstractArray{T,N1}, gradient::AbstractArray{T,N1}) where {T<:RealOrComplex,N1,N2}
    r = f.linear_operator*x-f.known_term
    gradient .= f.linear_operator'*r
    return norm(f.linear_operator*x-f.known_term)^2/2
end