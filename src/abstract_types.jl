#: Abstract functional types

export AbstractEvaluableFunction, fun_eval
export AbstractMinimizableFunction
export AbstractMinOptions, argmin, argmin!
export AbstractProximableFunction, proxy, proxy!, project, project!, argmin_options
export AbstractDiffPlusProxFunction
export AbstractProxyOptions
export AbstractDifferentiableFunction, grad_eval, grad_eval!, fungrad_eval, fungrad_eval!
export AbstractWeightedProximableFunction
export AbstractProjectionableSet
export AbstractIndicatorFunction
export AbstractProxPlusIndicator
export AbstractWeightedProxPlusIndicator


## Evaluable functions

abstract type AbstractEvaluableFunction{T,N} end
# fun_eval(fun::AbstractEvaluableFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = ...

(fun::AbstractEvaluableFunction{T,N})(x::AbstractArray{T,N}) where {T,N} = fun_eval(fun, x)


## Minimizable functions

abstract type AbstractMinimizableFunction{T,N}<:AbstractEvaluableFunction{T,N} end
abstract type AbstractMinOptions{MT<:AbstractMinimizableFunction} end
# argmin!(fun::MT, initial_estimate::AT, options::AbstractMinOptions{MT}, x::AT) where {T<:RealOrComplex,N,AT<:AbstractArray{T,N},MT<:AbstractMinimizableFunction{T,N}} = ...
# min_options(fun::AbstractMinimizableFunction{T,N}) where {T,N} = ...

Base.argmin(fun::MT, initial_estimate::AbstractArray{T,N}, options::AbstractMinOptions{MT}) where {T<:RealOrComplex,N,MT<:AbstractMinimizableFunction{T,N}} = argmin!(fun, initial_estimate, options, similar(initial_estimate))
Base.argmin(fun::AbstractMinimizableFunction{T,N}, initial_estimate::AbstractArray{CT,N}) where {T<:RealOrComplex,N} = argmin!(fun, initial_estimate, min_options(fun), similar(initial_estimate))
argmin!(fun::MT, initial_estimate::AT, x::AT) where {T<:Real,N,CT<:RealOrComplex{T},AT<:AbstractArray{CT,N},MT<:AbstractMinimizableFunction{CT,N}} = argmin!(fun, initial_estimate, min_options(fun), x)

min_options(::MT) where {MT<:AbstractMinimizableFunction} = exact_min(MT)


## Proximable functions

abstract type AbstractProximableFunction{T,N}<:AbstractEvaluableFunction{T,N} end
abstract type AbstractProxyOptions{PT<:AbstractProximableFunction} end
# proxy!(y::AT, λ::T, g::PT, options::AbstractProxyOptions{PT}, x::AT) where {T<:Real,N,CT<:RealOrComplex{T},AT<:AbstractArray{CT,N},PT<:AbstractProximableFunction{CT,N}} = ...
# proxy_options(g::AbstractProximableFunction) = ...

proxy(y::AbstractArray{CT,N}, λ::T, g::AbstractProximableFunction{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = proxy!(y, λ, g, proxy_options(g), similar(y))
proxy(y::AbstractArray{CT,N}, λ::T, g::PT, options::AbstractProxyOptions{PT}) where {T<:Real,N,CT<:RealOrComplex{T},PT<:AbstractProximableFunction{CT,N}} = proxy!(y, λ, g, options, similar(y))
proxy!(y::AT, λ::T, g::AbstractProximableFunction{CT,N}, x::AT) where {T<:Real,N,CT<:RealOrComplex{T},AT<:AbstractArray{CT,N}} = proxy!(y, λ, g, proxy_options(g), x)
project(y::AbstractArray{CT,N}, ε::T, g::AbstractProximableFunction{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = project!(y, ε, g, proxy_options(g), similar(y))
project(y::AbstractArray{CT,N}, ε::T, g::PT, options::AbstractProxyOptions{PT}) where {T<:Real,N,CT<:RealOrComplex{T},PT<:AbstractProximableFunction{CT,N}} = project!(y, ε, g, options, similar(y))
project!(y::AT, ε::T, g::AbstractProximableFunction{CT,N}, x::AT) where {T<:Real,N,CT<:RealOrComplex{T},AT<:AbstractArray{CT,N}} = project!(y, ε, g, proxy_options(g), x)

proxy_options(::PT) where {PT<:AbstractProximableFunction} = exact_proxy(PT)


## Differentiable functions

abstract type AbstractDifferentiableFunction{T,N} end

# fun_eval(fun::AbstractDifferentiableFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = ...
# grad_eval!(fun::AbstractDifferentiableFunction{T,N}, x::AbstractArray{T,N}, gradient::AbstractArray{T,N}) where {T,N} = ...
# fungrad_eval!(fun::AbstractDifferentiableFunction{T,N}, x::AbstractArray{T,N}, gradient::AbstractArray{T,N}) where {T,N} = ...

(fun::AbstractDifferentiableFunction{T,N})(x::AbstractArray{T,N}) where {T,N} = fun_eval(fun, x)

grad_eval(fun::AbstractDifferentiableFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = (g = similar(x); grad_eval!(fun, x, g); return g)

fungrad_eval(fun::AbstractDifferentiableFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = (g = similar(x); fval = fungrad_eval!(fun, x, g); return (fval, g))


## Abstract linear algebra utils

abstract type AbstractDiffPlusProxFunction{T,N}<:AbstractMinimizableFunction{T,N} end

# get_diff(fun::AbstractDiffPlusProxFunction{T,N}) where {T,N} = ...
# get_prox(fun::AbstractDiffPlusProxFunction{T,N}) where {T,N} = ...

fun_eval(fun::AbstractDiffPlusProxFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = get_diff(fun)(x)+get_prox(fun)(x)


## Weighted proximable functions

abstract type AbstractWeightedProximableFunction{T,N}<:AbstractProximableFunction{T,N} end

# get_prox(g::AbstractWeightedProximableFunction{T,N}) where {T,N} = ...
# get_linear_operator(g::AbstractWeightedProximableFunction{T,N}) where {T,N} = ...

fun_eval(g::AbstractWeightedProximableFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = get_prox(g)(get_linear_operator(g)*x)


## Projectionable sets

abstract type AbstractProjectionableSet{T,N} end

# Base.in(x::AbstractArray{T,N}, C::AbstractProjectionableSet{T,N}) where {T,N} = ...
# project!(x::AbstractArray{T,N}, C::AbstractProjectionableSet{T,N}, options::AbstractMinOptions, y::AbstractArray{T,N}) = ...

project(x::AbstractArray{T,N}, C::AbstractProjectionableSet{T,N}) where {T,N} = project!(x, C, argmin_options(C), similar(x))
project!(x::AbstractArray{T,N}, C::AbstractProjectionableSet{T,N}, y::AbstractArray{T,N}) where {T,N} = project!(x, C, argmin_options(C), y)
project(x::AbstractArray{T,N}, C::AbstractProjectionableSet{T,N}, options::AbstractMinOptions) where {T,N} = project!(x, C, options, similar(x))

argmin_options(::AbstractProjectionableSet) = exact_argmin()


## Indicator functions

abstract type AbstractIndicatorFunction{T,N}<:AbstractProximableFunction{T,N} end

# get_set(g::AbstractIndicatorFunction{T,N}) where {T,N} = ...

fun_eval(δC::AbstractIndicatorFunction{CT,N}, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = (x ∈ get_set(δC)) ? T(0) : T(Inf)

proxy!(y::AbstractArray{CT,N}, ::T, δ::AbstractIndicatorFunction{CT,N}, options::AbstractMinOptions, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = project!(y, get_set(δ), options, x)
project!(y::AbstractArray{CT,N}, ::T, δ::AbstractIndicatorFunction{CT,N}, options::AbstractMinOptions, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = project!(y, get_set(δ), options, x)

argmin_options(δ::AbstractIndicatorFunction) = argmin_options(get_set(δ))


## Sum of proximable functions

abstract type AbstractPlusProximableFunction{T,N}<:AbstractProximableFunction{T,N} end

# first_addend(f::AbstractPlusProximableFunction) = ...
# second_addend(f::AbstractPlusProximableFunction) = ...

fun_eval(g::AbstractPlusProximableFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = first_addend(g)(x)+second_addend(g)

argmin_options(g::AbstractPlusProximableFunction) = (argmin_options(first_addend(g)), argmin_options(second_addend(g)))


## Proximable + indicator functions

abstract type AbstractProxPlusIndicator{T,N}<:AbstractPlusProximableFunction{T,N} end

# get_prox(g::AbstractProxPlusIndicator{T,N}) where {T,N} = ...
# get_indicator(g::AbstractProxPlusIndicator{T,N}) where {T,N} = ...

fun_eval(g::AbstractProxPlusIndicator{CT,N}, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = (x ∈ get_set(get_indicator(g))) ? get_prox(g)(x) : T(Inf)

argmin_options(g::AbstractProxPlusIndicator) = (argmin_options(get_prox(g)), argmin_options(get_indicator(g)))


## Weighted proximable + indicator functions

abstract type AbstractWeightedProxPlusIndicator{T,N}<:AbstractProxPlusIndicator{T,N} end