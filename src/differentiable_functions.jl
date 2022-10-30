#: Suite of differentiable functions for TV optimization

export LeastSquaresMisfit, leastsquares_misfit
export ProxyObjFun, ProjObjFun, proxy_objfun, proj_objfun


# Least-squares misfit

struct LeastSquaresMisfit{T,N1,N2}<:DifferentiableFunction{T,N1}
    linear_operator::AbstractLinearOperator{T,N1,N2}
    known_term::AbstractArray{T,N2}
end

leastsquares_misfit(A::AbstractLinearOperator{T,N1,N2}, y::AbstractArray{T,N2}) where {T,N1,N2} = LeastSquaresMisfit{T,N1,N2}(A, y)

fun_eval(f::LeastSquaresMisfit{T,N1,N2}, x::AbstractArray{T,N1}) where {T<:RealOrComplex,N1,N2} = norm(f.linear_operator*x-f.known_term)^2/2

function grad_eval!(f::LeastSquaresMisfit{T,N1,N2}, x::AbstractArray{T,N1}, gradient::AbstractArray{T,N1}) where {T<:RealOrComplex,N1,N2}
    r = f.linear_operator*x-f.known_term
    gradient .= f.linear_operator'*r
    return gradient
end

function fungrad_eval!(f::LeastSquaresMisfit{T,N1,N2}, x::AbstractArray{T,N1}, gradient::AbstractArray{T,N1}) where {T<:RealOrComplex,N1,N2}
    r = f.linear_operator*x-f.known_term
    gradient .= f.linear_operator'*r
    return norm(f.linear_operator*x-f.known_term)^2/2
end


# Proxy objective function

struct ProxyObjFun{T,N}<:DifferentiableFunction{T,N}
    prox::ProximableFunction{T,N}
    weight::Real
end

proxy_objfun(g::ProximableFunction{CT,N}, λ::T) where {T<:Real,N,CT<:RealOrComplex{T}} = ProxyObjFun{CT,N}(g, λ)

function fun_eval(fun::ProxyObjFun{T,N}, y::AbstractArray{T,N}) where {T,N}
    x̄ = proxy(y, fun.weight, fun.prox)
    return norm(x̄-y)^2/2+fun.weight*fun.prox(x̄)
end

function grad_eval!(fun::ProxyObjFun{T,N}, y::AbstractArray{T,N}, g::AbstractArray{T,N}) where {T,N}
    g .= y-proxy(y, fun.weight, fun.prox)
    return g
end

function fungrad_eval!(fun::ProxyObjFun{T,N}, y::AbstractArray{T,N}, g::AbstractArray{T,N}) where {T,N}
    x̄ = proxy(y, fun.weight, fun.prox)
    g .= y-x̄
    return norm(g)^2/2+fun.weight*fun.prox(x̄)
end


# Projection objective function

struct ProjObjFun{T,N}<:DifferentiableFunction{T,N}
    prox::ProximableFunction{T,N}
    level::Real
end

proj_objfun(g::ProximableFunction{CT,N}, ε::T) where {T<:Real,N,CT<:RealOrComplex{T}} = ProjObjFun{CT,N}(g, ε)

function fun_eval(fun::ProjObjFun{T,N}, y::AbstractArray{T,N}) where {T,N}
    return norm(project(y, fun.level, fun.prox)-y)^2/2
end

function grad_eval!(fun::ProjObjFun{T,N}, y::AbstractArray{T,N}, g::AbstractArray{T,N}) where {T,N}
    g .= y-project(y, fun.level, fun.prox)
    return g
end

function fungrad_eval!(fun::ProjObjFun{T,N}, y::AbstractArray{T,N}, g::AbstractArray{T,N}) where {T,N}
    g .= y-project(y, fun.weight, fun.prox)
    return norm(g)^2/2
end