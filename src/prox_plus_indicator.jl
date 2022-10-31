export ProxPlusIndicator, WeightedProxPlusIndicator


# Weighted proximable + indicator

abstract type ProxPlusIndicator{T,N}<:ProximableFunction{T,N} end

struct WeightedProxPlusIndicator{T,N1,N2}<:ProxPlusIndicator{T,N1}
    wprox::WeightedProximableFun{T,N1,N2}
    indicator::IndicatorFunction{T,N1}
end

Base.:+(g::WeightedProximableFun{T,N1,N2}, δ::IndicatorFunction{T,N1}) where {T,N1,N2} = WeightedProxPlusIndicator{T,N1,N2}(g, δ)
Base.:+(δ::IndicatorFunction{T,N1}, g::WeightedProximableFun{T,N1,N2}) where {T,N1,N2} = g+δ

fun_eval(g::WeightedProxPlusIndicator{T,N1,N2}, x::AbstractArray{T,N1}) where {T,N1,N2} = (x ∈ g.indicator.C) ? g.wprox(x) : T(Inf)

get_optimizer(g::WeightedProxPlusIndicator) = get_optimizer(g.wprox)

struct WeightedProxPlusIndicatorProxObj{T,N1,N2}<:DifferentiableFunction{T,N2}
    linear_operator::AbstractLinearOperator{T,N1,N2}
    y::AbstractArray{T,N1}
    λ::Real
    C::ProjectionableSet{T,N1}
end

wprox_plus_indicator_proxobj(A::AbstractLinearOperator{CT,N1,N2}, y::AbstractArray{CT,N1}, λ::T, C::ProjectionableSet{CT,N1}) where {T<:Real,N1,N2,CT<:RealOrComplex{T}} = WeightedProxPlusIndicatorProxObj{CT,N1,N2}(A, y, λ, C)

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

function proxy!(y::AbstractArray{CT,N1}, λ::T, g::WeightedProxPlusIndicator{CT,N1,N2}, x::AbstractArray{CT,N1}; optimizer::Union{Nothing,Optimizer}=nothing) where {T<:Real,N1,N2,CT<:RealOrComplex{T}}

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

function project!(y::AbstractArray{CT,N1}, ε::T, g::WeightedProxPlusIndicator{CT,N1,N2}, x::AbstractArray{CT,N1}; optimizer::Union{Nothing,Optimizer}=nothing) where {T<:Real,N1,N2,CT<:RealOrComplex{T}}

    # Objective function (dual problem)
    f = wprox_plus_indicator_proxobj(g.wprox.linear_operator, y, T(1), g.indicator.C)+conjugate(indicator(g.wprox.prox ≤ ε))

    # Minimization (dual variable)
    optimizer = get_optimizer(optimizer, g); is_specified(optimizer)
    p0 = similar(y, range_size(g.wprox.linear_operator)); p0 .= 0
    p = minimize(f, p0, optimizer)

    # Dual to primal solution
    return project!(y-g.wprox.linear_operator'*p, g.indicator.C, x)

end