#: Abstract functional types

export AbstractArgMinOptions, ExactArgMin
export AbstractMinimizableFunction, argmin!, fun_eval
export AbstractDifferentiableFunction, grad_eval, grad_eval!, fungrad_eval, fungrad_eval!
export AbstractProximableFunction, proxy, proxy!, project, project!
export AbstractProjectionableSet


## Argmin option types

abstract type AbstractArgMinOptions end
struct ExactArgMin<:AbstractArgMinOptions end

not_implemented(::ExactArgMin) = error("Exact minimization has not been implemented!")


## Minimizable functions

abstract type AbstractMinimizableFunction{T,N} end

# argmin!(fun::AbstractMinimizableFunction{T,N}, x0::AbstractArray{T,N}, options::AbstractArgMinOptions, x::AbstractArray{T,N}) where {T,N} = ...
# fun_eval(fun::AbstractMinimizableFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = ...

Base.argmin(fun::AbstractMinimizableFunction{T,N}, x0::AbstractArray{T,N}, options::AbstractArgMinOptions) where {T,N} = argmin!(fun, x0, options, similar(x0))

(fun::AbstractMinimizableFunction{T,N})(x::AbstractArray{T,N}) where {T,N} = fun_eval(fun, x)


## Differentiable functions

abstract type AbstractDifferentiableFunction{T,N} end

# fun_eval(fun::AbstractDifferentiableFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = ...
# grad_eval!(fun::AbstractDifferentiableFunction{T,N}, x::AbstractArray{T,N}, gradient::AbstractArray{T,N}) where {T,N} = ...
# fungrad_eval!(fun::AbstractDifferentiableFunction{T,N}, x::AbstractArray{T,N}, gradient::AbstractArray{T,N}) where {T,N} = ...

(fun::AbstractDifferentiableFunction{T,N})(x::AbstractArray{T,N}) where {T,N} = fun_eval(fun, x)

grad_eval(fun::AbstractDifferentiableFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = (g = similar(x); grad_eval!(fun, x, g); return g)

fungrad_eval(fun::AbstractDifferentiableFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = (g = similar(x); fval = fungrad_eval!(fun, x, g); return (fval, g))


## Proximable functions

abstract type AbstractProximableFunction{T,N} end

# fun_eval(fun::AbstractProximableFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = ...
# proxy!(y::AbstractArray{CT,N}, λ::T, g::AbstractProximableFunction{CT,N}, options::AbstractArgMinOptions, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = ...

(fun::AbstractProximableFunction{T,N})(x::AbstractArray{T,N}) where {T,N} = fun_eval(fun, x)

proxy(y::AbstractArray{CT,N}, λ::T, g::AbstractProximableFunction{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = proxy!(y, λ, g, ExactArgMin(), similar(y))
proxy!(y::AbstractArray{CT,N}, λ::T, g::AbstractProximableFunction{CT,N}, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = proxy!(y, λ, g, ExactArgMin(), x)
proxy(y::AbstractArray{CT,N}, λ::T, g::AbstractProximableFunction{CT,N}, options::AbstractArgMinOptions) where {T<:Real,N,CT<:RealOrComplex{T}} = proxy!(y, λ, g, options, similar(y))
project(y::AbstractArray{CT,N}, ε::T, g::AbstractProximableFunction{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = project!(y, ε, g, ExactArgMin(), similar(y))
project!(y::AbstractArray{CT,N}, ε::T, g::AbstractProximableFunction{CT,N}, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = project!(y, ε, g, ExactArgMin(), x)
project(y::AbstractArray{CT,N}, ε::T, g::AbstractProximableFunction{CT,N}, options::AbstractArgMinOptions) where {T<:Real,N,CT<:RealOrComplex{T}} = project!(y, ε, g, options, similar(y))


## Convex sets

abstract type AbstractProjectionableSet{T,N} end

## Base.in(x::AbstractArray{T,N}, C::AbstractProjectionableSet{T,N}) where {T,N} = ...
## project!(x::AbstractArray{T,N}, C::AbstractProjectionableSet{T,N}, options::AbstractArgMinOptions, y::AbstractArray{T,N}) = ...

project(x::AbstractArray{T,N}, C::AbstractProjectionableSet{T,N}) where {T,N} = project!(x, C, ExactArgMin(), similar(x))
project!(x::AbstractArray{T,N}, C::AbstractProjectionableSet{T,N}, y::AbstractArray{T,N}) where {T,N} = project!(x, C, ExactArgMin(), y)
project(x::AbstractArray{T,N}, C::AbstractProjectionableSet{T,N}, options::AbstractArgMinOptions) where {T,N} = project!(x, C, options, similar(x))