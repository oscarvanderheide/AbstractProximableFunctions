#: Abstract type utils


## Indicator functions

abstract type AbstractIndicatorFunction{T,N}<:AbstractProximableFunction{T,N} end
# indicator_set(g::AbstractIndicatorFunction{T,N}) where {T,N} = ...

funeval(δC::AbstractIndicatorFunction{CT,N}, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = (x ∈ indicator_set(δC)) ? T(0) : T(Inf)

prox!(y::AT, ::T, δ::AbstractIndicatorFunction{CT,N}, options::AbstractProjectionableSetOptions, x::AT) where {T<:Real,N,CT<:RealOrComplex{T},AT<:AbstractArray{CT,N}} = proj!(y, indicator_set(δ), options, x)
proj!(y::AT, ::T, δ::AbstractIndicatorFunction{CT,N}, options::AbstractProjectionableSetOptions, x::AT) where {T<:Real,N,CT<:RealOrComplex{T},AT<:AbstractArray{CT,N}} = proj!(y, indicator_set(δ), options, x)

options(δ::AbstractIndicatorFunction) = options(indicator_set(δ))


## Weighted proximable functions

abstract type AbstractWeightedProximableFunction{T,N}<:AbstractProximableFunction{T,N} end
# prox_fun(g::AbstractWeightedProximableFunction{T,N}) where {T,N} = ...
# linear_operator(g::AbstractWeightedProximableFunction{T,N}) where {T,N} = ...

funeval(g::AbstractWeightedProximableFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = prox_fun(g)(linear_operator(g)*x)

function prox!(y::AT, λ::T, g::AbstractWeightedProximableFunction{CT,N}, options::WeightedProximableFISTA, x::AT) where {T<:Real,N,CT<:RealOrComplex{T},AT<:AbstractArray{CT,N}}

    # Objective function (dual problem)
    f = leastsquares_misfit(λ*linear_operator(g)', y)+λ*conjugate(prox_fun(g))

    # Minimization (dual variable)
    FISTA_options = set_Lipschitz_constant(options.FISTA_options, λ^2*Lipschitz_constant(options.FISTA_options))
    p0 = similar(y, range_size(get_linear_operator(g))); p0 .= 0
    p = argmin(f, p0, FISTA_options)

    # Dual to primal solution
    return x .= y-λ*(linear_operator(g)'*p)

end

function proj!(y::AT, ε::T, g::AbstractWeightedProximableFunction{CT,N}, options::WeightedProximableFISTA, x::AT) where {T<:Real,N,CT<:RealOrComplex{T},AT<:AbstractArray{CT,N}}

    # Objective function (dual problem)
    f = leastsquares_misfit(linear_operator(g)', y)+conjugate(indicator(prox_fun(g) ≤ ε))

    # Minimization (dual variable)
    p0 = similar(y, range_size(linear_operator(g))); p0 .= 0
    p = argmin(f, p0, options.FISTA_options)

    # Dual to primal solution
    return x .= y-linear_operator(g)'*p

end


## Prox/Proj-able + indicator functions

abstract type AbstractProximablePlusIndicator{T,N}<:AbstractProximableFunction{T,N} end
# prox_term(g::AbstractProximablePlusIndicator{T,N}) where {T,N} = ...
# indicator_term(g::AbstractProximablePlusIndicator{T,N}) where {T,N} = ...

first_term(g::AbstractProximablePlusIndicator) = prox_term(g)
second_term(g::AbstractProximablePlusIndicator) = indicator_term(g)


## Weighted proximable + indicator functions

abstract type AbstractWeightedProximablePlusIndicator{T,N}<:AbstractProximablePlusIndicator{T,N} end


## f+g, f differentiable, g prox/proj-able

abstract type AbstractDifferentiablePlusProximableFunction{T,N}<:AbstractMinimizableFunction{T,N} end
# diff_term(fun::AbstractDifferentiablePlusProximableFunction{T,N}) where {T,N} = ...
# prox_term(fun::AbstractDifferentiablePlusProximableFunction{T,N}) where {T,N} = ...

funeval(fun::AbstractDifferentiablePlusProximableFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = diff_term(fun)(x)+prox_term(fun)(x)