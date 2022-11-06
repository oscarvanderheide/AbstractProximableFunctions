#: Utilities for proximable functions

export WeightedProximableFunction, weighted_prox, get_prox, get_linear_operator


# Proximable + linear operator

struct WeightedProximableFunction{T,N1,N2}<:AbstractWeightedProximableFunction{T,N1}
    prox::AbstractProximableFunction{T,N2}
    linear_operator::AbstractLinearOperator{T,N1,N2}
end

weighted_prox(g::AbstractProximableFunction{T,N2}, A::AbstractLinearOperator{T,N1,N2}) where {T,N1,N2} = WeightedProximableFunction{T,N1,N2}(g, A)
Base.:∘(g::AbstractProximableFunction{T,N2}, A::AbstractLinearOperator{T,N1,N2}) where {T,N1,N2} = weighted_prox(g, A)

get_prox(g::WeightedProximableFunction) = g.prox
get_linear_operator(g::WeightedProximableFunction) = g.linear_operator

fun_eval(g::AbstractWeightedProximableFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = get_prox(g)(get_linear_operator(g)*x)

function proxy!(y::AbstractArray{CT,N}, λ::T, g::AbstractWeightedProximableFunction{CT,N}, options::ConjugateAndFISTA{T}, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}}

    # Objective function (dual problem)
    f = leastsquares_misfit(λ*get_linear_operator(g)', y)+λ*conjugate(get_prox(g))

    # Minimization (dual variable)
    options_FISTA = set_Lipschitz_constant(options.options_FISTA, λ^2*Lipschitz_constant(options.options_FISTA))
    p0 = similar(y, range_size(get_linear_operator(g))); p0 .= 0
    p = argmin(f, p0, options_FISTA)

    # Dual to primal solution
    return x .= y-λ*(get_linear_operator(g)'*p)

end

proxy!(::AbstractArray{CT,N}, ::T, ::AbstractWeightedProximableFunction{CT,N}, options::ExactArgMin, ::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = not_implemented(options)

function project!(y::AbstractArray{CT,N}, ε::T, g::AbstractWeightedProximableFunction{CT,N}, options::ConjugateAndFISTA{T}, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}}

    # Objective function (dual problem)
    f = leastsquares_misfit(get_linear_operator(g)', y)+conjugate(indicator(get_prox(g) ≤ ε))

    # Minimization (dual variable)
    p0 = similar(y, range_size(get_linear_operator(g))); p0 .= 0
    p = argmin(f, p0, options.options_FISTA)

    # Dual to primal solution
    return x .= y-get_linear_operator(g)'*p

end

project!(::AbstractArray{CT,N}, ::T, ::AbstractWeightedProximableFunction{CT,N}, options::ExactArgMin, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = not_implemented(options)


# Proximable + indicator

struct ProxPlusIndicator{T,N}<:AbstractProximableFunction{T,N}
    prox::AbstractProximableFunction{T,N}
    indicator::IndicatorFunction{T,N}
end

Base.:+(g::AbstractProximableFunction{T,N}, δ::IndicatorFunction{T,N}) where {T,N} = ProxPlusIndicator{T,N}(g, δ)
Base.:+(δ::IndicatorFunction{T,N}, g::AbstractProximableFunction{T,N}) where {T,N} = g+δ

fun_eval(g::ProxPlusIndicator{T,N}, x::AbstractArray{T,N}) where {T,N} = (x ∈ g.indicator.C) ? g.prox(x) : T(Inf)

struct ProxPlusIndicatorProxObj{T,N}<:AbstractDifferentiableFunction{T,N}
    y::AbstractArray{T,N}
    λ::Real
    C::AbstractProjectionableSet{T,N}
end

prox_plus_indicator_proxobj(y::AbstractArray{CT,N}, λ::T, C::AbstractProjectionableSet{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = ProxPlusIndicatorProxObj{CT,N}(y, λ, C)

function fun_eval(f::ProxPlusIndicatorProxObj{CT,N}, p::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}}
    r = f.y-f.λ*p
    xC = project(r, f.C)
    return T(0.5)*norm(r)^2-T(0.5)*norm(r-xC)^2
end

function grad_eval!(f::ProxPlusIndicatorProxObj{CT,N}, p::AbstractArray{CT,N}, g::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}}
    r = f.y-f.λ*p
    xC = project(r, f.C)
    return g .= -f.λ*xC
end

function fungrad_eval!(f::ProxPlusIndicatorProxObj{CT,N}, p::AbstractArray{CT,N}, g::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}}
    r = f.y-f.λ*p
    xC = project(r, f.C)
    g .= -f.λ*xC
    return T(0.5)*norm(r)^2-T(0.5)*norm(r-xC)^2, g
end

function proxy!(y::AbstractArray{CT,N}, λ::T, g::ProxPlusIndicator{CT,N}, options::ConjugateProjectAndFISTA{T}, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}}

    # Objective function (dual problem)
    f = prox_plus_indicator_proxobj(y, λ, g.indicator.C)+λ*conjugate(g.prox)

    # Minimization (dual variable)
    options_FISTA = set_Lipschitz_constant(options.options_FISTA, λ^2*Lipschitz_constant(options.options_FISTA))
    p0 = similar(y); p0 .= 0
    p = argmin(f, p0, options_FISTA)

    # Dual to primal solution
    return project!(y-λ*p, g.indicator.C, x)

end

proxy!(::AbstractArray{CT,N}, ::T, ::ProxPlusIndicator{CT,N}, options::ExactArgMin, ::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = not_implemented(options)

function project!(y::AbstractArray{CT,N}, ε::T, g::ProxPlusIndicator{CT,N}, options::ConjugateProjectAndFISTA{T}, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}}

    # Objective function (dual problem)
    f = prox_plus_indicator_proxobj(y, T(1), g.indicator.C)+conjugate(indicator(g.prox ≤ ε))

    # Minimization (dual variable)
    p0 = similar(y); p0 .= 0
    p = argmin(f, p0, options.options_FISTA)

    # Dual to primal solution
    return project!(y-p, g.indicator.C, x)

end

project!(::AbstractArray{CT,N}, ::T, ::ProxPlusIndicator{CT,N}, options::ExactArgMin, ::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = not_implemented(options)


# Weighted proximable + indicator

struct WeightedProxPlusIndicator{T,N}<:AbstractProximableFunction{T,N}
    wprox::AbstractWeightedProximableFunction{T,N}
    indicator::IndicatorFunction{T,N}
end

Base.:+(g::AbstractWeightedProximableFunction{T,N}, δ::IndicatorFunction{T,N}) where {T,N} = WeightedProxPlusIndicator{T,N}(g, δ)
Base.:+(δ::IndicatorFunction{T,N}, g::AbstractWeightedProximableFunction{T,N}) where {T,N} = g+δ

fun_eval(g::WeightedProxPlusIndicator{T,N}, x::AbstractArray{T,N}) where {T,N} = (x ∈ g.indicator.C) ? g.wprox(x) : T(Inf)

struct WeightedProxPlusIndicatorProxObj{T,N1,N2}<:AbstractDifferentiableFunction{T,N2}
    linear_operator::AbstractLinearOperator{T,N1,N2}
    y::AbstractArray{T,N1}
    λ::Real
    C::AbstractProjectionableSet{T,N1}
end

wprox_plus_indicator_proxobj(A::AbstractLinearOperator{CT,N1,N2}, y::AbstractArray{CT,N1}, λ::T, C::AbstractProjectionableSet{CT,N1}) where {T<:Real,N1,N2,CT<:RealOrComplex{T}} = WeightedProxPlusIndicatorProxObj{CT,N1,N2}(A, y, λ, C)

function fun_eval(f::WeightedProxPlusIndicatorProxObj{CT,N1,N2}, p::AbstractArray{CT,N2}) where {T<:Real,CT<:RealOrComplex{T},N1,N2}
    r = f.y-f.λ*f.linear_operator'*p
    xC = project(r, f.C)
    return T(0.5)*norm(r)^2-T(0.5)*norm(r-xC)^2
end

function grad_eval!(f::WeightedProxPlusIndicatorProxObj{CT,N1,N2}, p::AbstractArray{CT,N2}, g::AbstractArray{CT,N2}) where {T<:Real,CT<:RealOrComplex{T},N1,N2}
    r = f.y-f.λ*f.linear_operator'*p
    xC = project(r, f.C)
    return g .= -f.λ*(f.linear_operator*xC)
end

function fungrad_eval!(f::WeightedProxPlusIndicatorProxObj{CT,N1,N2}, p::AbstractArray{CT,N2}, g::AbstractArray{CT,N2}) where {T<:Real,CT<:RealOrComplex{T},N1,N2}
    r = f.y-f.λ*f.linear_operator'*p
    xC = project(r, f.C)
    g .= -f.λ*(f.linear_operator*xC)
    return T(0.5)*norm(r)^2-T(0.5)*norm(r-xC)^2, g
end

function proxy!(y::AbstractArray{CT,N}, λ::T, g::WeightedProxPlusIndicator{CT,N}, options::ConjugateProjectAndFISTA{T}, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}}

    # Objective function (dual problem)
    f = wprox_plus_indicator_proxobj(get_linear_operator(g.wprox), y, λ, g.indicator.C)+λ*conjugate(get_prox(g.wprox))

    # Minimization (dual variable)
    options_FISTA = set_Lipschitz_constant(options.options_FISTA, λ^2*Lipschitz_constant(options.options_FISTA))
    p0 = similar(y, range_size(get_linear_operator(g.wprox))); p0 .= 0
    p = argmin(f, p0, options_FISTA)

    # Dual to primal solution
    return project!(y-λ*(get_linear_operator(g.wprox)'*p), g.indicator.C, x)

end

proxy!(::AbstractArray{CT,N}, ::T, ::WeightedProxPlusIndicator{CT,N}, options::ExactArgMin, ::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = not_implemented(options)

function project!(y::AbstractArray{CT,N}, ε::T, g::WeightedProxPlusIndicator{CT,N}, options::ConjugateProjectAndFISTA{T}, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}}

    # Objective function (dual problem)
    f = wprox_plus_indicator_proxobj(get_linear_operator(g.wprox), y, T(1), g.indicator.C)+conjugate(indicator(get_prox(g.wprox) ≤ ε))

    # Minimization (dual variable)
    p0 = similar(y, range_size(get_linear_operator(g.wprox))); p0 .= 0
    p = argmin(f, p0, options.options_FISTA)

    # Dual to primal solution
    return project!(y-get_linear_operator(g.wprox)'*p, g.indicator.C, x)

end

project!(::AbstractArray{CT,N}, ::T, ::WeightedProxPlusIndicator{CT,N}, options::ExactArgMin, ::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = not_implemented(options)