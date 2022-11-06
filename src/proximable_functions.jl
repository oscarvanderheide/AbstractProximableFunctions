#: Utilities for proximable functions

export WeightedProximableFunction, weighted_prox


# Proximable + linear operator

struct WeightedProximableFunction{T,N1,N2}<:AbstractProximableFunction{T,N1}
    prox::AbstractProximableFunction{T,N2}
    linear_operator::AbstractLinearOperator{T,N1,N2}
end

weighted_prox(g::AbstractProximableFunction{T,N2}, A::AbstractLinearOperator{T,N1,N2}) where {T,N1,N2} = WeightedProximableFunction{T,N1,N2}(g, A)
Base.:∘(g::AbstractProximableFunction{T,N2}, A::AbstractLinearOperator{T,N1,N2}) where {T,N1,N2} = weighted_prox(g, A)

fun_eval(g::WeightedProximableFunction{T,N1,N2}, x::AbstractArray{T,N1}) where {T,N1,N2} = g.prox(g.linear_operator*x)

function proxy!(y::AbstractArray{CT,N1}, λ::T, g::WeightedProximableFunction{CT,N1,N2}, options::ConjugateAndFISTA{T}, x::AbstractArray{CT,N1}) where {T<:Real,N1,N2,CT<:RealOrComplex{T}}

    # Objective function (dual problem)
    f = leastsquares_misfit(λ*g.linear_operator', y)+λ*conjugate(g.prox)

    # Minimization (dual variable)
    options_FISTA = set_Lipschitz_constant(options.options_FISTA, λ^2*Lipschitz_constant(options.options_FISTA))
    p0 = similar(y, range_size(g.linear_operator)); p0 .= 0
    p = argmin(f, p0, options_FISTA)

    # Dual to primal solution
    return x .= y-λ*(g.linear_operator'*p)

end

proxy!(::AbstractArray{CT,N1}, ::T, ::WeightedProximableFunction{CT,N1,N2}, options::ExactArgMin, ::AbstractArray{CT,N1}) where {T<:Real,N1,N2,CT<:RealOrComplex{T}} = not_implemented(options)

function project!(y::AbstractArray{CT,N1}, ε::T, g::WeightedProximableFunction{CT,N1,N2}, options::ConjugateAndFISTA{T}, x::AbstractArray{CT,N1}) where {T<:Real,N1,N2,CT<:RealOrComplex{T}}

    # Objective function (dual problem)
    f = leastsquares_misfit(g.linear_operator', y)+conjugate(indicator(g.prox ≤ ε))

    # Minimization (dual variable)
    p0 = similar(y, range_size(g.linear_operator)); p0 .= 0
    p = argmin(f, p0, options.options_FISTA)

    # Dual to primal solution
    return x .= y-g.linear_operator'*p

end

project!(::AbstractArray{CT,N1}, ::T, ::WeightedProximableFunction{CT,N1,N2}, options::ExactArgMin, x::AbstractArray{CT,N1}) where {T<:Real,N1,N2,CT<:RealOrComplex{T}} = not_implemented(options)


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

struct WeightedProxPlusIndicator{T,N1,N2}<:AbstractProximableFunction{T,N1}
    wprox::WeightedProximableFunction{T,N1,N2}
    indicator::IndicatorFunction{T,N1}
end

Base.:+(g::WeightedProximableFunction{T,N1,N2}, δ::IndicatorFunction{T,N1}) where {T,N1,N2} = WeightedProxPlusIndicator{T,N1,N2}(g, δ)
Base.:+(δ::IndicatorFunction{T,N1}, g::WeightedProximableFunction{T,N1,N2}) where {T,N1,N2} = g+δ

fun_eval(g::WeightedProxPlusIndicator{T,N1,N2}, x::AbstractArray{T,N1}) where {T,N1,N2} = (x ∈ g.indicator.C) ? g.wprox(x) : T(Inf)

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

function proxy!(y::AbstractArray{CT,N1}, λ::T, g::WeightedProxPlusIndicator{CT,N1,N2}, options::ConjugateProjectAndFISTA{T}, x::AbstractArray{CT,N1}) where {T<:Real,N1,N2,CT<:RealOrComplex{T}}

    # Objective function (dual problem)
    f = wprox_plus_indicator_proxobj(g.wprox.linear_operator, y, λ, g.indicator.C)+λ*conjugate(g.wprox.prox)

    # Minimization (dual variable)
    options_FISTA = set_Lipschitz_constant(options.options_FISTA, λ^2*Lipschitz_constant(options.options_FISTA))
    p0 = similar(y, range_size(g.wprox.linear_operator)); p0 .= 0
    p = argmin(f, p0, options_FISTA)

    # Dual to primal solution
    return project!(y-λ*(g.wprox.linear_operator'*p), g.indicator.C, x)

end

proxy!(::AbstractArray{CT,N1}, ::T, ::WeightedProxPlusIndicator{CT,N1,N2}, options::ExactArgMin, ::AbstractArray{CT,N1}) where {T<:Real,N1,N2,CT<:RealOrComplex{T}} = not_implemented(options)

function project!(y::AbstractArray{CT,N1}, ε::T, g::WeightedProxPlusIndicator{CT,N1,N2}, options::ConjugateProjectAndFISTA{T}, x::AbstractArray{CT,N1}) where {T<:Real,N1,N2,CT<:RealOrComplex{T}}

    # Objective function (dual problem)
    f = wprox_plus_indicator_proxobj(g.wprox.linear_operator, y, T(1), g.indicator.C)+conjugate(indicator(g.wprox.prox ≤ ε))

    # Minimization (dual variable)
    p0 = similar(y, range_size(g.wprox.linear_operator)); p0 .= 0
    p = argmin(f, p0, options.options_FISTA)

    # Dual to primal solution
    return project!(y-g.wprox.linear_operator'*p, g.indicator.C, x)

end

project!(::AbstractArray{CT,N1}, ::T, ::WeightedProxPlusIndicator{CT,N1,N2}, options::ExactArgMin, ::AbstractArray{CT,N1}) where {T<:Real,N1,N2,CT<:RealOrComplex{T}} = not_implemented(options)