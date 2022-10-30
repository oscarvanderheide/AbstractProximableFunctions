export ProxPlusIndicator, WeightedProxPlusIndicator


# Weighted proximable + indicator

abstract type ProxPlusIndicator{T,N}<:ProximableFunction{T,N} end

struct WeightedProxPlusIndicator{T,N1,N2}<:ProxPlusIndicator{T,N1}
    wprox::WeightedProximableFun{T,N1,N2}
    indicator::IndicatorFunction{T,N1}
end

(g::WeightedProxPlusIndicator{T,N1,N2})(x::AbstractArray{T,N1}) where {T,N1,N2} = (x ∈ g.indicator.C) ? g.wprox(x) : T(Inf)

Base.:+(g::WeightedProximableFun{T,N1,N2}, δ::IndicatorFunction{T,N1}) where {T,N1,N2} = WeightedProxPlusIndicator{T,N1,N2}(g, δ)
Base.:+(δ::IndicatorFunction{T,N1}, g::WeightedProximableFun{T,N1,N2}) where {T,N1,N2} = g+δ

struct WeightedProxPlusIndicatorProxObj{T,N1,N2}<:DifferentiableFunction{T,N2}
    linear_operator::AbstractLinearOperator{T,N1,N2}
    y::AbstractArray{T,N1}
    λ::T
    C::ProjectionableSet{T,N1}
end

wprox_plus_indicator_proxobj(A::AbstractLinearOperator{CT,N1,N2}, y::AbstractArray{CT,N1}, λ::T, C::ProjectionableSet{CT,N1}) where {T<:Real,N1,N2,CT<:RealOrComplex{T}} = WeightedProxPlusIndicatorProxObj{CT,N1,N2}(A, y, λ, C)

function funeval!(f::WeightedProxPlusIndicatorProxObj{CT,N1,N2}, p::AbstractArray{CT,N2}; gradient::Union{Nothing,AbstractArray{CT,N2}}=nothing, eval::Bool=true) where {T<:Real,CT<:RealOrComplex{T},N1,N2}
    r = f.y-f.λ*f.linear_operator'*p
    xC = project(r, f.C)
    eval ? (fval = (T(0.5)*norm(r)^2-T(0.5)*norm(xC-r)^2)/f.λ^2) : (fval = nothing)
    ~isnothing(gradient) && (gradient .= -(f.linear_operator*xC)/f.λ)
    return fval
end

function proxy!(y::AbstractArray{CT,N1}, λ::T, g::WeightedProxPlusIndicator{CT,N1,N2}, x::AbstractArray{CT,N1}) where {T<:Real,N1,N2,CT<:RealOrComplex{T}}

    # Objective function (dual problem)
    f = wprox_plus_indicator_proxobj(g.wprox.linear_operator, y, λ, g.indicator.C)+conjugate(g.wprox.prox)/λ

    # Minimization (dual variable)
    p0 = similar(y, range_size(g.wprox.linear_operator)); fill!(p0, 0)
    p = minimize(f, p0, g.wprox.opt)

    # Dual to primal solution
    return project!(y-λ*(g.wprox.linear_operator'*p), g.indicator.C, x)

end