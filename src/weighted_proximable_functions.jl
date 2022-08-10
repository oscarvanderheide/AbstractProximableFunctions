export WeightedProximableFun, weighted_prox


# Proximable + linear operator

struct WeightedProximableFun{T,N1,N2}<:ProximableFunction{T,N1}
    prox::ProximableFunction{T,N2}
    linear_operator::AbstractLinearOperator{T,N1,N2}
end

weighted_prox(g::ProximableFunction{T,N2}, A::AbstractLinearOperator{T,N1,N2}) where {T,N1,N2} = WeightedProximableFun{T,N1,N2}(g, A)
Base.:∘(g::ProximableFunction{T,N2}, A::AbstractLinearOperator{T,N1,N2}) where {T,N1,N2} = weighted_prox(g, A)

function proxy!(y::AbstractArray{CT,N1}, λ::T, g::WeightedProximableFun{CT,N1,N2}, x::AbstractArray{CT,N1}, opt::AbstractOptimizer) where {T<:Real,N1,N2,CT<:RealOrComplex{T}}

    # Objective function (dual problem)
    f = leastsquares_misfit(adjoint(g.linear_operator), y/λ)+conjugate(g.prox)/λ

    # Minimization (dual variable)
    p0 = similar(y, range_size(g.linear_operator)); fill!(p0, 0)
    opt.verbose ? ((fval, p) = minimize(f, p0, opt)) : (p = minimize(f, p0, opt))

    # Dual to primal solution
    x .= y-λ*(adjoint(g.linear_operator)*p)
    opt.verbose ? (return (fval, x)) : (return x)

end

function project!(y::AbstractArray{CT,N1}, ε::T, g::WeightedProximableFun{CT,N1,N2}, x::AbstractArray{CT,N1}, opt::AbstractOptimizer) where {T<:Real,N1,N2,CT<:RealOrComplex{T}}

    # Objective function (dual problem)
    f = leastsquares_misfit(adjoint(g.linear_operator), y)+conjugate(indicator(g.prox ≤ ε))

    # Minimization (dual variable)
    p0 = similar(y, range_size(g.linear_operator)); fill!(p0, 0)
    opt.verbose ? ((fval, p) = minimize(f, p0, opt)) : (p = minimize(f, p0, opt))

    # Dual to primal solution
    x .= y-adjoint(g.linear_operator)*p
    opt.verbose ? (return (fval, x)) : (return x)

end

(g::WeightedProximableFun{T,N1,N2})(x::AbstractArray{T,N1}) where {T,N1,N2} = g.prox(g.linear_operator*x)

Flux.gpu(g::WeightedProximableFun{T,N1,N2}) where {T,N1,N2} = WeightedProximableFun{T,N1,N2}(g.prox, gpu(g.linear_operator))
Flux.cpu(g::WeightedProximableFun{T,N1,N2}) where {T,N1,N2} = WeightedProximableFun{T,N1,N2}(g.prox, cpu(g.linear_operator))