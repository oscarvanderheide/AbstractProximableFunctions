#: Abstract functional types

export DiffPlusProxOptimizer
export MinimizableFunction, minimize, minimize!, fun_eval, fun_eval!
export DifferentiableFunction, grad_eval, grad_eval!, fungrad_eval, fungrad_eval!
export ProximableFunction, proxy, proxy!, project, project!
export ProjectionableSet


## Optimizers

abstract type Optimizer end
abstract type DiffPlusProxOptimizer<:Optimizer end


## Minimizable functions

abstract type MinimizableFunction{T,N} end

# minimize!(fun::MinimizableFunction{T,N}, x0::AbstractArray{T,N}, x::AbstractArray{T,N}; optimizer::Optimizer) where {T,N} = ...
# fun_eval(fun::MinimizableFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = ...

minimize(fun::MinimizableFunction{T,N}, x0::AbstractArray{T,N}; optimizer::Union{Nothing,Optimizer}=nothing) where {T,N} = minimize!(fun, x0, similar(x0); optimizer=optimizer)

(fun::MinimizableFunction{T,N})(x::AbstractArray{T,N}) where {T,N} = fun_eval(fun, x)


## Differentiable functions

abstract type DifferentiableFunction{T,N}<:MinimizableFunction{T,N} end

# fun_eval(fun::DifferentiableFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = ...
# grad_eval!(fun::DifferentiableFunction{T,N}, x::AbstractArray{T,N}, gradient::AbstractArray{T,N}) where {T,N} = ...
# fungrad_eval!(fun::DifferentiableFunction{T,N}, x::AbstractArray{T,N}, gradient::AbstractArray{T,N}) where {T,N} = ...

grad_eval(fun::DifferentiableFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = (g = similar(x); grad_eval!(fun, x, g); return g)

fungrad_eval(fun::DifferentiableFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = (g = similar(x); fval = fungrad_eval!(fun, x, g); return (fval, g))


## Proximable functions

abstract type ProximableFunction{T,N}<:MinimizableFunction{T,N} end

# fun_eval(fun::ProximableFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = ...
# proxy!(y::AbstractArray{CT,N}, λ::T, g::ProximableFunction{CT,N}; optimizer::Optimizer) where {T<:Real,N,CT<:RealOrComplex{T}} = ...
# get_optimizer(g::ProximableFunction) = ...

proxy(y::AbstractArray{CT,N}, λ::T, g::ProximableFunction{CT,N}; optimizer::Union{Nothing,Optimizer}=nothing) where {T<:Real,N,CT<:RealOrComplex{T}} = proxy!(y, λ, g, similar(y); optimizer=optimizer)
project(y::AbstractArray{CT,N}, ε::T, g::ProximableFunction{CT,N}; optimizer::Union{Nothing,Optimizer}=nothing) where {T<:Real,N,CT<:RealOrComplex{T}} = project!(y, ε, g, similar(y); optimizer=optimizer)

function get_optimizer(optimizer::Union{Nothing,Optimizer,ProximableFunction}...)
    @inbounds for opt = optimizer
        if ~isnothing(opt)
            (opt isa Optimizer) && (return opt)
            (opt isa ProximableFunction) && ~isnothing(get_optimizer(opt)) && (return get_optimizer(opt))
        end
    end
    return nothing
end

is_specified(opt::Union{Nothing,Optimizer}) = isnothing(opt) && error("The requested routines needs the specification of an optimizer")


## Convex sets

abstract type ProjectionableSet{T,N} end

## Base.in(x::AbstractArray{T,N}, C::ProjectionableSet{T,N}) where {T,N} = ...
## project!(x::AbstractArray{T,N}, C::ProjectionableSet{T,N}, y::AbstractArray{T,N}; optimizer::Optimizer) = ...

project(x::AbstractArray{T,N}, C::ProjectionableSet{T,N}; optimizer::Union{Nothing,Optimizer}=nothing) where {T,N} = project!(x, C, similar(x); optimizer=optimizer)