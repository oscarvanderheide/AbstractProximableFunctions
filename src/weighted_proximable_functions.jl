export set_optimizer!, WeightedProximableFun, weighted_prox, ProxPlusIndicator, WeightedProxPlusIndicator


# Proximable + linear operator

mutable struct WeightedProximableFun{T,N1,N2}<:ProximableFunction{T,N1}
    prox::ProximableFunction{T,N2}
    linear_operator::AbstractLinearOperator{T,N1,N2}
    opt::AbstractOptimizer
end

weighted_prox(g::ProximableFunction{T,N2}, A::AbstractLinearOperator{T,N1,N2}, opt::AbstractOptimizer) where {T,N1,N2} = WeightedProximableFun{T,N1,N2}(g, A, opt)
set_optimizer!(g::WeightedProximableFun, opt::AbstractOptimizer) = (g.opt = opt)

function proxy!(y::AbstractArray{CT,N1}, λ::T, g::WeightedProximableFun{CT,N1,N2}, x::AbstractArray{CT,N1}) where {T<:Real,N1,N2,CT<:RealOrComplex{T}}

    # Objective function (dual problem)
    f = leastsquares_misfit(adjoint(g.linear_operator), y/λ)+conjugate(g.prox)/λ

    # Minimization (dual variable)
    p0 = similar(y, range_size(g.linear_operator)); fill!(p0, 0)
    p = minimize(f, p0, g.opt)

    # Dual to primal solution
    return x .= y-λ*(adjoint(g.linear_operator)*p)

end

function project!(y::AbstractArray{CT,N1}, ε::T, g::WeightedProximableFun{CT,N1,N2}, x::AbstractArray{CT,N1}) where {T<:Real,N1,N2,CT<:RealOrComplex{T}}

    # Objective function (dual problem)
    f = leastsquares_misfit(adjoint(g.linear_operator), y)+conjugate(indicator(g.prox ≤ ε))

    # Minimization (dual variable)
    p0 = similar(y, range_size(g.linear_operator)); fill!(p0, 0)
    p = minimize(f, p0, g.opt)

    # Dual to primal solution
    return x .= y-adjoint(g.linear_operator)*p

end

(g::WeightedProximableFun{T,N1,N2})(x::AbstractArray{T,N1}) where {T,N1,N2} = g.prox(g.linear_operator*x)

Flux.gpu(g::WeightedProximableFun{T,N1,N2}) where {T,N1,N2} = WeightedProximableFun{T,N1,N2}(g.prox, gpu(g.linear_operator), g.opt)
Flux.cpu(g::WeightedProximableFun{T,N1,N2}) where {T,N1,N2} = WeightedProximableFun{T,N1,N2}(g.prox, cpu(g.linear_operator), g.opt)


# Weighted proximable + indicator

abstract type ProxPlusIndicator{T,N}<:ProximableFunction{T,N} end

struct WeightedProxPlusIndicator{T,N1,N2}<:ProxPlusIndicator{T,N1}
    wprox::WeightedProximableFun{T,N1,N2}
    indicator::IndicatorFunction{T,N1}
end

Base.:+(g::WeightedProximableFun{T,N1,N2}, δ::IndicatorFunction{T,N1}) where {T,N1,N2} = WeightedProxPlusIndicator{T,N1,N2}(g, δ)
Base.:+(δ::IndicatorFunction{T,N1}, g::WeightedProximableFun{T,N1,N2}) where {T,N1,N2} = g+δ

struct WeightedProxPlusIndicatorProxObj{T,N1,N2}<:DifferentiableFunction{T,N2}
    linear_operator::AbstractLinearOperator{T,N1,N2}
    y::AbstractArray{T,N1}
    λ::T
    C::ProjectionableSet{T,N1}
end

wprox_plus_indicator_proxobj(A::AbstractLinearOperator{CT,N1,N2}, y::AbstractArray{CT,N1}, λ::T, C::ProjectionableSet{T,N1}) where {T<:Real,N1,N2,CT<:RealOrComplex{T}} = WeightedProxPlusIndicatorProxObj{T,N1,N2}(A, y, λ, C)

function funeval!(f::WeightedProxPlusIndicatorProxObj{T,N1,N2}, p::AbstractArray{CT,N2}; gradient::Union{Nothing,AbstractArray{CT,N2}}=nothing, eval::Bool=true) where {T<:Real,N1,N2,CT<:RealOrComplex{T}}
    r = f.y-f.λ*f.linear_operator'*p
    xC = project(r, f.C)
    eval ? (fval = T(0.5)*norm(r)^2-T(0.5)*norm(xC-r)^2) : (fval = nothing)
    ~isnothing(gradient) && (gradient .= -f.λ*f.linear_operator*xC)
    return fval
end

function proxy!(y::AbstractArray{CT,N1}, λ::T, g::WeightedProxPlusIndicator{CT,N1,N2}, x::AbstractArray{CT,N1}) where {T<:Real,N1,N2,CT<:RealOrComplex{T}}

    # Objective function (dual problem)
    f = wprox_plus_indicator_proxobj(g.wprox.linear_operator, y, λ, g.indicator.C)+λ*conjugate(g.prox)

    # Minimization (dual variable)
    p0 = similar(y, range_size(g.linear_operator)); fill!(p0, 0)
    p = minimize(f, p0, g.opt)

    # Dual to primal solution
    return project!(y-λ*(adjoint(g.linear_operator)*p), g.indicator.C, x)

end