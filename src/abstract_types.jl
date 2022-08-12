#: Abstract functional types

export AbstractOptimizer, MinimizableFunction, DifferentiableFunction, ProximableFunction, ProjectionableSet, funeval, funeval!, minimize, minimize!, proxy, proxy!, project, project!


# Abstract type declarations

abstract type AbstractOptimizer<:Flux.Optimise.AbstractOptimiser end

"""Expected behavior for MinimizableFunction:
- minimize!(f::MinimizableFunction{T,N}, x0::AbstractArray{T,N}, opt::AbstractOptimizer, x::AbstractArray{T,N}) where {T,N}
It approximates the solution of the optimization problem: min_x f(x)
"""
abstract type MinimizableFunction{T,N} end

minimize(fun::MinimizableFunction{T,N}, x0::AbstractArray{T,N}, opt::AbstractOptimizer) where {T,N} = minimize!(fun, x0, opt, similar(x0))


"""Expected behavior for DifferentiableFunction:
- fval::T = funeval!(f::DifferentiableFunction{DT}, x::DT; gradient::DT, eval::Bool) where {T,N,DT<:AbstractArray{T,N}} 
"""
abstract type DifferentiableFunction{T,N}<:MinimizableFunction{T,N} end

function funeval(fun::DifferentiableFunction{T,N}, x::AbstractArray{T,N}; gradient::Bool=false, eval::Bool=true) where {T,N}
    gradient ? (g = similar(x)) : (g = nothing)
    fval = funeval!(fun, x; gradient=g, eval=eval)
    gradient ? (return (fval, g)) : (return fval)
end

(fun::DifferentiableFunction{T,N})(x::AbstractArray{T,N}; gradient::Bool=false, eval::Bool=true) where {T,N} = funeval(fun, x; gradient=gradient, eval=eval)

"""Expected behavior for ProximableFunction:
- proxy!(y::AbstractArray{T,N}, λ::T, g::ProximableFunction{T,N}, x::AbstractArray{T,N}) where {T,N}
- proxy!(y::AbstractArray{T,N}, λ::T, g::ProximableFunction{T,N}, x::AbstractArray{T,N}, opt::AbstractOptimizer) where {T,N}
It approximates the solution of the optimization problem: min_x 0.5*norm(x-y)^2+λ*g(x)
- project!(y::AbstractArray{T,N}, ε::T, g::ProximableFunction{T,N}, x::AbstractArray{T,N}) where {T,N}
- project!(y::AbstractArray{T,N}, ε::T, g::ProximableFunction{T,N}, x::AbstractArray{T,N}, opt::AbstractOptimizer) where {T,N}
It approximates the solution of the optimization problem: min_{g(x)<=ε} 0.5*norm(x-y)^2
"""
abstract type ProximableFunction{T,N} end

proxy(y::AbstractArray{CT,N}, λ::T, g::ProximableFunction{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = proxy!(y, λ, g, similar(y))
project(y::AbstractArray{CT,N}, ε::T, g::ProximableFunction{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = project!(y, ε, g, similar(y))


"""Projectional sets
Expected behavior for convex sets: y = project!(x, C, y), y = Π_C(x)
"""
abstract type ProjectionableSet{T,N} end

project(x::AbstractArray{T,N}, C::ProjectionableSet{T,N}) where {T,N} = project!(x, C, similar(x))