#: Abstract functional types

export AbstractMinimizableFunction, fun_eval
export AbstractDiffPlusProxFunction
export AbstractArgminMethod, argmin, argmin!
export AbstractDifferentiableFunction, grad_eval, grad_eval!, fungrad_eval, fungrad_eval!
export AbstractProximableFunction, proxy, proxy!, project, project!
export AbstractWeightedProximableFunction
export AbstractProjectionableSet
export AbstractIndicatorFunction
export AbstractProxPlusIndicator
export AbstractWeightedProxPlusIndicator


## Minimizable functions

abstract type AbstractMinimizableFunction{T,N} end

# fun_eval(fun::AbstractMinimizableFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = ...

(fun::AbstractMinimizableFunction{T,N})(x::AbstractArray{T,N}) where {T,N} = fun_eval(fun, x)

abstract type AbstractDiffPlusProxFunction{T,N}<:AbstractMinimizableFunction{T,N} end

# get_diff(fun::AbstractDiffPlusProxFunction{T,N}) where {T,N} = ...
# get_prox(fun::AbstractDiffPlusProxFunction{T,N}) where {T,N} = ...

fun_eval(fun::AbstractDiffPlusProxFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = get_diff(fun)(x)+get_prox(fun)(x)


## Argmin option types

abstract type AbstractArgminMethod end

# argmin!(fun::AbstractMinimizableFunction{CT,N}, initial_estimate::AbstractArray{CT,N}, options::AbstractArgminMethod, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = ...

Base.argmin(fun::AbstractMinimizableFunction{CT,N}, initial_estimate::AbstractArray{CT,N}, options::AbstractArgminMethod) where {T<:Real,N,CT<:RealOrComplex{T}} = argmin!(fun, initial_estimate, options, similar(initial_estimate))


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
# proxy!(y::AbstractArray{CT,N}, λ::T, g::AbstractProximableFunction{CT,N}, options::AbstractArgminMethod, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = ...

(fun::AbstractProximableFunction{T,N})(x::AbstractArray{T,N}) where {T,N} = fun_eval(fun, x)

proxy(y::AbstractArray{CT,N}, λ::T, g::AbstractProximableFunction{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = proxy!(y, λ, g, exact_argmin(), similar(y))
proxy!(y::AbstractArray{CT,N}, λ::T, g::AbstractProximableFunction{CT,N}, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = proxy!(y, λ, g, exact_argmin(), x)
proxy(y::AbstractArray{CT,N}, λ::T, g::AbstractProximableFunction{CT,N}, options::AbstractArgminMethod) where {T<:Real,N,CT<:RealOrComplex{T}} = proxy!(y, λ, g, options, similar(y))
project(y::AbstractArray{CT,N}, ε::T, g::AbstractProximableFunction{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = project!(y, ε, g, exact_argmin(), similar(y))
project!(y::AbstractArray{CT,N}, ε::T, g::AbstractProximableFunction{CT,N}, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = project!(y, ε, g, exact_argmin(), x)
project(y::AbstractArray{CT,N}, ε::T, g::AbstractProximableFunction{CT,N}, options::AbstractArgminMethod) where {T<:Real,N,CT<:RealOrComplex{T}} = project!(y, ε, g, options, similar(y))


## Weighted proximable functions

abstract type AbstractWeightedProximableFunction{T,N}<:AbstractProximableFunction{T,N} end

# get_prox(g::AbstractWeightedProximableFunction{T,N}) where {T,N} = ...
# get_linear_operator(g::AbstractWeightedProximableFunction{T,N}) where {T,N} = ...

fun_eval(g::AbstractWeightedProximableFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = get_prox(g)(get_linear_operator(g)*x)


## Projectionable sets

abstract type AbstractProjectionableSet{T,N} end

# Base.in(x::AbstractArray{T,N}, C::AbstractProjectionableSet{T,N}) where {T,N} = ...
# project!(x::AbstractArray{T,N}, C::AbstractProjectionableSet{T,N}, options::AbstractArgminMethod, y::AbstractArray{T,N}) = ...

project(x::AbstractArray{T,N}, C::AbstractProjectionableSet{T,N}) where {T,N} = project!(x, C, exact_argmin(), similar(x))
project!(x::AbstractArray{T,N}, C::AbstractProjectionableSet{T,N}, y::AbstractArray{T,N}) where {T,N} = project!(x, C, exact_argmin(), y)
project(x::AbstractArray{T,N}, C::AbstractProjectionableSet{T,N}, options::AbstractArgminMethod) where {T,N} = project!(x, C, options, similar(x))


## Indicator functions

abstract type AbstractIndicatorFunction{T,N}<: AbstractProximableFunction{T,N} end

# get_set(g::AbstractIndicatorFunction{T,N}) where {T,N} = ...

fun_eval(δC::AbstractIndicatorFunction{CT,N}, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = (x ∈ get_set(δC)) ? T(0) : T(Inf)


## Proximable + indicator functions

abstract type AbstractProxPlusIndicator{T,N}<:AbstractProximableFunction{T,N} end

# get_prox(g::AbstractProxPlusIndicator{T,N}) where {T,N} = ...
# get_indicator(g::AbstractProxPlusIndicator{T,N}) where {T,N} = ...

fun_eval(g::AbstractProxPlusIndicator{CT,N}, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = (x ∈ get_set(get_indicator(g))) ? get_prox(g)(x) : T(Inf)


## Weighted proximable + indicator functions

abstract type AbstractWeightedProxPlusIndicator{T,N}<:AbstractProxPlusIndicator{T,N} end