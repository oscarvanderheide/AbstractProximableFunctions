#: Suite of differentiable functions for TV optimization

export LeastSquaresMisfit, leastsquares_misfit, funeval


# Least-squares misfit

"""
Least-squares misfit function associated to the problem
min_x 0.5*||A*x-y||^2
"""
struct LeastSquaresMisfit{T,N1,N2}<:DifferentiableFunction{T,N1}
    linear_operator::AbstractLinearOperator{T,N1,N2}
    known_term::AbstractArray{T,N2}
end

leastsquares_misfit(A::AbstractLinearOperator{T,N1,N2}, y::AbstractArray{T,N2}) where {T,N1,N2} = LeastSquaresMisfit{T,N1,N2}(A, y)

function funeval!(f::LeastSquaresMisfit{T,N1,N2}, x::AbstractArray{T,N1}; gradient::Union{Nothing,AbstractArray{T,N1}}=nothing, eval::Bool=true) where {T,N1,N2}
    r = f.linear_operator*x-f.known_term
    eval ? (fval = T(0.5)*norm(r)^2) : (fval = nothing)
    ~isnothing(gradient) && (gradient .= f.linear_operator'*r)
    return fval
end