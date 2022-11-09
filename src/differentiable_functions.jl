#: General routines of differentiable functions

export ProxyObjFun, ProjObjFun, proxy_objfun, proj_objfun
export LeastSquaresMisfit, leastsquares_misfit
export test_grad


# Proxy objective function

struct ProxyObjFun{T,N}<:AbstractDifferentiableFunction{T,N}
    prox::AbstractProximableFunction{T,N}
    weight::Real
    options::AbstractMinOptions
end

proxy_objfun(g::AbstractProximableFunction{CT,N}, λ::T; options::AbstractMinOptions=ExactArgmin()) where {T<:Real,N,CT<:RealOrComplex{T}} = ProxyObjFun{CT,N}(g, λ, options)

function fun_eval(fun::ProxyObjFun{T,N}, y::AbstractArray{T,N}) where {T,N}
    x̄ = proxy(y, fun.weight, fun.prox, fun.options)
    return norm(x̄-y)^2/2+fun.weight*fun.prox(x̄)
end

function grad_eval!(fun::ProxyObjFun{T,N}, y::AbstractArray{T,N}, g::AbstractArray{T,N}) where {T,N}
    g .= y-proxy(y, fun.weight, fun.prox, fun.options)
    return g
end

function fungrad_eval!(fun::ProxyObjFun{T,N}, y::AbstractArray{T,N}, g::AbstractArray{T,N}) where {T,N}
    x̄ = proxy(y, fun.weight, fun.prox, fun.options)
    g .= y-x̄
    return norm(g)^2/2+fun.weight*fun.prox(x̄)
end


# Projection objective function

struct ProjObjFun{T,N}<:AbstractDifferentiableFunction{T,N}
    prox::AbstractProximableFunction{T,N}
    level::Real
    options::AbstractMinOptions
end

proj_objfun(g::AbstractProximableFunction{CT,N}, ε::T; options::AbstractMinOptions=ExactArgmin()) where {T<:Real,N,CT<:RealOrComplex{T}} = ProjObjFun{CT,N}(g, ε, options)

function fun_eval(fun::ProjObjFun{T,N}, y::AbstractArray{T,N}) where {T,N}
    return norm(project(y, fun.level, fun.prox, fun.options)-y)^2/2
end

function grad_eval!(fun::ProjObjFun{T,N}, y::AbstractArray{T,N}, g::AbstractArray{T,N}) where {T,N}
    g .= y-project(y, fun.level, fun.prox, fun.options)
    return g
end

function fungrad_eval!(fun::ProjObjFun{T,N}, y::AbstractArray{T,N}, g::AbstractArray{T,N}) where {T,N}
    g .= y-project(y, fun.weight, fun.prox, fun.options)
    return norm(g)^2/2
end


# Least-squares misfit

struct LeastSquaresMisfit{T,N1,N2}<:AbstractDifferentiableFunction{T,N1}
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


# Test util

function test_grad(fun::AbstractDifferentiableFunction{CT,N}, x::AbstractArray{CT,N}; step::T=T(1e-4), rtol::T=eps(T)) where {T<:Real,N,CT<:Union{T,Complex{T}}}

    dx = convert(typeof(x), randn(CT, size(x))); dx *= norm(x)/norm(dx)
    Δx = grad_eval(fun, x)
    fp1 = fun_eval(fun, x+T(0.5)*step*dx)
    fm1 = fun_eval(fun, x-T(0.5)*step*dx)
    return isapprox((fp1-fm1)/step, real(dot(dx, Δx)); rtol=rtol)

end