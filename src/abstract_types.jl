#: Abstract functional types

export AbstractConvexOptimizer, AbstractDiffPlusProxOptimizer
export AbstractMinimizableFunction, minimize, minimize!, fun_eval, fun_eval!
export AbstractDifferentiableFunction, grad_eval, grad_eval!, fungrad_eval, fungrad_eval!
export AbstractProximableFunction, proxy, proxy!, project, project!
export AbstractProjectionableSet


## Optimizers

abstract type AbstractConvexOptimizer end
abstract type AbstractDiffPlusProxOptimizer<:AbstractConvexOptimizer end


## Minimizable functions

abstract type AbstractMinimizableFunction{T,N} end

# minimize!(fun::AbstractMinimizableFunction{T,N}, x0::AbstractArray{T,N}, x::AbstractArray{T,N}; optimizer::Optimizer) where {T,N} = ...
# fun_eval(fun::AbstractMinimizableFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = ...

minimize(fun::AbstractMinimizableFunction{T,N}, x0::AbstractArray{T,N}; optimizer::Union{Nothing,AbstractConvexOptimizer}=nothing) where {T,N} = minimize!(fun, x0, similar(x0); optimizer=optimizer)

(fun::AbstractMinimizableFunction{T,N})(x::AbstractArray{T,N}) where {T,N} = fun_eval(fun, x)


## Differentiable functions

abstract type AbstractDifferentiableFunction{T,N}<:AbstractMinimizableFunction{T,N} end

# fun_eval(fun::AbstractDifferentiableFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = ...
# grad_eval!(fun::AbstractDifferentiableFunction{T,N}, x::AbstractArray{T,N}, gradient::AbstractArray{T,N}) where {T,N} = ...
# fungrad_eval!(fun::AbstractDifferentiableFunction{T,N}, x::AbstractArray{T,N}, gradient::AbstractArray{T,N}) where {T,N} = ...

grad_eval(fun::AbstractDifferentiableFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = (g = similar(x); grad_eval!(fun, x, g); return g)

fungrad_eval(fun::AbstractDifferentiableFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = (g = similar(x); fval = fungrad_eval!(fun, x, g); return (fval, g))


## Proximable functions

abstract type AbstractProximableFunction{T,N}<:AbstractMinimizableFunction{T,N} end

# fun_eval(fun::AbstractProximableFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = ...
# proxy!(y::AbstractArray{CT,N}, λ::T, g::AbstractProximableFunction{CT,N}; optimizer::Optimizer) where {T<:Real,N,CT<:RealOrComplex{T}} = ...
# get_optimizer(g::AbstractProximableFunction) = ...

proxy(y::AbstractArray{CT,N}, λ::T, g::AbstractProximableFunction{CT,N}; optimizer::Union{Nothing,AbstractConvexOptimizer}=nothing) where {T<:Real,N,CT<:RealOrComplex{T}} = proxy!(y, λ, g, similar(y); optimizer=optimizer)
project(y::AbstractArray{CT,N}, ε::T, g::AbstractProximableFunction{CT,N}; optimizer::Union{Nothing,AbstractConvexOptimizer}=nothing) where {T<:Real,N,CT<:RealOrComplex{T}} = project!(y, ε, g, similar(y); optimizer=optimizer)

function get_optimizer(optimizer::Union{Nothing,AbstractConvexOptimizer,AbstractProximableFunction}...)
    @inbounds for opt = optimizer
        if ~isnothing(opt)
            (opt isa AbstractConvexOptimizer) && (return opt)
            (opt isa AbstractProximableFunction) && ~isnothing(get_optimizer(opt)) && (return get_optimizer(opt))
        end
    end
    return nothing
end

is_specified(opt::Union{Nothing,AbstractConvexOptimizer}) = isnothing(opt) && error("The requested routines needs the specification of an optimizer")

abstract type ProxPlusIndicator{T,N}<:AbstractProximableFunction{T,N} end


## Convex sets

abstract type AbstractProjectionableSet{T,N} end

## Base.in(x::AbstractArray{T,N}, C::AbstractProjectionableSet{T,N}) where {T,N} = ...
## project!(x::AbstractArray{T,N}, C::AbstractProjectionableSet{T,N}, y::AbstractArray{T,N}; optimizer::Optimizer) = ...

project(x::AbstractArray{T,N}, C::AbstractProjectionableSet{T,N}; optimizer::Union{Nothing,AbstractConvexOptimizer}=nothing) where {T,N} = project!(x, C, similar(x); optimizer=optimizer)