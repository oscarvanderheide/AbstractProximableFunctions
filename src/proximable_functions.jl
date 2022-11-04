#: Utilities for proximable functions

export WeightedProximableFunction, weighted_prox


# Proximable + linear operator

struct WeightedProximableFunction{T,N1,N2}<:AbstractWeightedProximableFunction{T,N1,N2}
    prox::AbstractProximableFunction{T,N2}
    linear_operator::AbstractLinearOperator{T,N1,N2}
    optimizer::Union{Nothing,AbstractConvexOptimizer}
end

weighted_prox(g::AbstractProximableFunction{T,N2}, A::AbstractLinearOperator{T,N1,N2}; optimizer::Union{Nothing,AbstractConvexOptimizer}=nothing) where {T,N1,N2} = WeightedProximableFunction{T,N1,N2}(g, A, optimizer)
Base.:∘(g::AbstractProximableFunction{T,N2}, A::AbstractLinearOperator{T,N1,N2}; optimizer::Union{Nothing,AbstractConvexOptimizer}=nothing) where {T,N1,N2} = weighted_prox(g, A; optimizer=optimizer)

fun_eval(g::WeightedProximableFunction{T,N1,N2}, x::AbstractArray{T,N1}) where {T,N1,N2} = g.prox(g.linear_operator*x)

get_optimizer(g::WeightedProximableFunction) = get_optimizer(g.optimizer, get_optimizer(g.prox))

function proxy!(y::AbstractArray{CT,N1}, λ::T, g::WeightedProximableFunction{CT,N1,N2}, x::AbstractArray{CT,N1}; optimizer::Union{Nothing,AbstractConvexOptimizer}=nothing) where {T<:Real,N1,N2,CT<:RealOrComplex{T}}

    # Objective function (dual problem)
    f = leastsquares_misfit(λ*g.linear_operator', y)+λ*conjugate(g.prox)

    # Minimization (dual variable)
    optimizer = get_optimizer(optimizer, g); is_specified(optimizer)
    optimizer = set_Lipschitz_constant(optimizer, λ^2*Lipschitz_constant(optimizer))
    p0 = similar(y, range_size(g.linear_operator)); p0 .= 0
    p = minimize(f, p0, optimizer)

    # Dual to primal solution
    return x .= y-λ*(g.linear_operator'*p)

end

function project!(y::AbstractArray{CT,N1}, ε::T, g::WeightedProximableFunction{CT,N1,N2}, x::AbstractArray{CT,N1}; optimizer::Union{Nothing,AbstractConvexOptimizer}=nothing) where {T<:Real,N1,N2,CT<:RealOrComplex{T}}

    # Objective function (dual problem)
    f = leastsquares_misfit(g.linear_operator', y)+conjugate(indicator(g.prox ≤ ε))

    # Minimization (dual variable)
    optimizer = get_optimizer(optimizer, g); is_specified(optimizer)
    p0 = similar(y, range_size(g.linear_operator)); p0 .= 0
    p = minimize(f, p0, optimizer)

    # Dual to primal solution
    return x .= y-g.linear_operator'*p

end


# Weighted proximable + indicator

struct WeightedProxPlusIndicator{T,N1,N2}<:ProxPlusIndicator{T,N1}
    wprox::AbstractWeightedProximableFunction{T,N1,N2}
    indicator::IndicatorFunction{T,N1}
end

Base.:+(g::AbstractWeightedProximableFunction{T,N1,N2}, δ::IndicatorFunction{T,N1}) where {T,N1,N2} = WeightedProxPlusIndicator{T,N1,N2}(g, δ)
Base.:+(δ::IndicatorFunction{T,N1}, g::AbstractWeightedProximableFunction{T,N1,N2}) where {T,N1,N2} = g+δ

fun_eval(g::WeightedProxPlusIndicator{T,N1,N2}, x::AbstractArray{T,N1}) where {T,N1,N2} = (x ∈ g.indicator.C) ? g.wprox(x) : T(Inf)

get_optimizer(g::WeightedProxPlusIndicator) = get_optimizer(g.wprox)

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

function proxy!(y::AbstractArray{CT,N1}, λ::T, g::WeightedProxPlusIndicator{CT,N1,N2}, x::AbstractArray{CT,N1}; optimizer::Union{Nothing,AbstractConvexOptimizer}=nothing) where {T<:Real,N1,N2,CT<:RealOrComplex{T}}

    # Objective function (dual problem)
    f = wprox_plus_indicator_proxobj(g.wprox.linear_operator, y, λ, g.indicator.C)+λ*conjugate(g.wprox.prox)

    # Minimization (dual variable)
    optimizer = get_optimizer(optimizer, g); is_specified(optimizer)
    optimizer = set_Lipschitz_constant(optimizer, λ^2*Lipschitz_constant(optimizer))
    p0 = similar(y, range_size(g.wprox.linear_operator)); p0 .= 0
    p = minimize(f, p0, optimizer)

    # Dual to primal solution
    return project!(y-λ*(g.wprox.linear_operator'*p), g.indicator.C, x)

end

function project!(y::AbstractArray{CT,N1}, ε::T, g::WeightedProxPlusIndicator{CT,N1,N2}, x::AbstractArray{CT,N1}; optimizer::Union{Nothing,AbstractConvexOptimizer}=nothing) where {T<:Real,N1,N2,CT<:RealOrComplex{T}}

    # Objective function (dual problem)
    f = wprox_plus_indicator_proxobj(g.wprox.linear_operator, y, T(1), g.indicator.C)+conjugate(indicator(g.wprox.prox ≤ ε))

    # Minimization (dual variable)
    optimizer = get_optimizer(optimizer, g); is_specified(optimizer)
    p0 = similar(y, range_size(g.wprox.linear_operator)); p0 .= 0
    p = minimize(f, p0, optimizer)

    # Dual to primal solution
    return project!(y-g.wprox.linear_operator'*p, g.indicator.C, x)

end