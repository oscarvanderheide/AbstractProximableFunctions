#: Suite of differentiable functions for TV optimization

export LeastSquaresMisfit, leastsquares_misfit


# Least-squares misfit

"""
Least-squares misfit function associated to the problem
min_x 0.5*||A*x-y||^2
"""
struct LeastSquaresMisfit{T,N1,N2}<:DifferentiableFunction{T,N1}
    A::AbstractLinearOperator{T,N1,N2}
    y::AbstractArray{T,N2}
end

leastsquares_misfit(A::AbstractLinearOperator{T,N1,N2}, y::AbstractArray{T,N2}) where {T,N1,N2} = LeastSquaresMisfit{T,N1,N2}(A, y)

function grad!(f::LeastSquaresMisfit{T,N1,N2}, x::AbstractArray{T,N1}, g::Union{Nothing,AbstractArray{T,N1}}; eval::Bool=true) where {T,N1,N2}
    r = f.A*x-f.y
    eval ? (fval = T(0.5)*norm(r)^2) : (fval = nothing)
    g !== nothing && (g .= adjoint(f.A)*r)
    return fval
end