export WeightedProximableFun, weighted_prox, set_optimizer!


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

Flux.gpu(g::WeightedProximableFun{T,N1,N2}) where {T,N1,N2} = WeightedProximableFun{T,N1,N2}(g.prox, gpu(g.linear_operator))
Flux.cpu(g::WeightedProximableFun{T,N1,N2}) where {T,N1,N2} = WeightedProximableFun{T,N1,N2}(g.prox, cpu(g.linear_operator))