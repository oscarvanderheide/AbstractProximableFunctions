export WeightedProximableFun, weighted_prox


# Proximable + linear operator

struct WeightedProximableFun{T,N1,N2}<:ProximableFunction{T,N1}
    prox::ProximableFunction{T,N2}
    linear_operator::AbstractLinearOperator{T,N1,N2}
    optimizer::Union{Nothing,Optimizer}
end

weighted_prox(g::ProximableFunction{T,N2}, A::AbstractLinearOperator{T,N1,N2}; optimizer::Union{Nothing,Optimizer}=nothing) where {T,N1,N2} = WeightedProximableFun{T,N1,N2}(g, A, optimizer)
Base.:∘(g::ProximableFunction{T,N2}, A::AbstractLinearOperator{T,N1,N2}; optimizer::Union{Nothing,Optimizer}=nothing) where {T,N1,N2} = weighted_prox(g, A; optimizer=optimizer)

fun_eval(g::WeightedProximableFun{T,N1,N2}, x::AbstractArray{T,N1}) where {T,N1,N2} = g.prox(g.linear_operator*x)

get_optimizer(g::WeightedProximableFun) = get_optimizer(g.optimizer, get_optimizer(g.prox))

function proxy!(y::AbstractArray{CT,N1}, λ::T, g::WeightedProximableFun{CT,N1,N2}, x::AbstractArray{CT,N1}; optimizer::Union{Nothing,Optimizer}=nothing) where {T<:Real,N1,N2,CT<:RealOrComplex{T}}

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

function project!(y::AbstractArray{CT,N1}, ε::T, g::WeightedProximableFun{CT,N1,N2}, x::AbstractArray{CT,N1}; optimizer::Union{Nothing,Optimizer}=nothing) where {T<:Real,N1,N2,CT<:RealOrComplex{T}}

    # Objective function (dual problem)
    f = leastsquares_misfit(g.linear_operator', y)+conjugate(indicator(g.prox ≤ ε))

    # Minimization (dual variable)
    optimizer = get_optimizer(optimizer, g); is_specified(optimizer)
    p0 = similar(y, range_size(g.linear_operator)); p0 .= 0
    p = minimize(f, p0, optimizer)

    # Dual to primal solution
    return x .= y-g.linear_operator'*p

end