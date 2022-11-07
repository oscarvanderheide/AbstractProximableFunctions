#: Misc utilities

export leastsquares_solve, leastsquares_solve!
export spectral_radius


## Least-squares linear problem routines

leastsquares_solve!(A::AbstractLinearOperator{CT,N1,N2}, b::AbstractArray{CT,N2}, g::AbstractProximableFunction{CT,N1}, initial_estimate::AbstractArray{CT,N1}, options::AbstractArgminMethod, x::AbstractArray{CT,N1}) where {T<:Real,N1,N2,CT<:RealOrComplex{T}} = argmin!(leastsquares_misfit(A, b)+g, initial_estimate, options, x)

leastsquares_solve(A::AbstractLinearOperator{CT,N1,N2}, b::AbstractArray{CT,N2}, g::AbstractProximableFunction{CT,N1}, initial_estimate::AbstractArray{CT,N1}, options::AbstractArgminMethod) where {T<:Real,N1,N2,CT<:RealOrComplex{T}} = leastsquares_solve!(A, b, g, initial_estimate, options, similar(initial_estimate))


# Other utils

function spectral_radius(A::Union{AbstractMatrix{T},AbstractLinearOperator{T,N,N}}; x::Union{Nothing,AbstractArray{T,N}}=nothing, niter::Int64=10) where {T,N}
    if isnothing(x)
        A isa AbstractMatrix && (x = randn(T, size(A,2)))
        A isa AbstractLinearOperator && (x = randn(T, domain_size(A)))
    end
    x = x/norm(x)
    y = similar(x)
    ρ = real(T)(0)
    for _ = 1:niter
        y .= A*x
        ρ = norm(y)/norm(x)
        x .= y/norm(y)
    end
    return ρ
end