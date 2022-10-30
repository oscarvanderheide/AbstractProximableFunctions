#: Utils

export conjugate


# Differentiable function linear algebra

struct ScaledDifferentiableFunction{T,N}<:DifferentiableFunction{T,N}
    scale::Real
    fun::DifferentiableFunction{T,N}
end

Base.:*(scale::T, fun::DifferentiableFunction{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = ScaledDifferentiableFunction{CT,N}(scale, fun)
Base.:*(fun::DifferentiableFunction{CT,N}, scale::T) where {T<:Real,N,CT<:RealOrComplex{T}} = scale*fun
Base.:/(fun::DifferentiableFunction{CT,N}, scale::T) where {T<:Real,N,CT<:RealOrComplex{T}} = (1/scale)*fun

fun_eval(fun::ScaledDifferentiableFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = fun.scale*fun.fun(x)
grad_eval!(fun::ScaledDifferentiableFunction{T,N}, x::AbstractArray{T,N}, g::AbstractArray{T,N}) where {T,N} = (grad_eval!(fun.fun, x, g); g .*= fun.scale; return g)
fungrad_eval!(fun::ScaledDifferentiableFunction{T,N}, x::AbstractArray{T,N}, g::AbstractArray{T,N}) where {T,N} = (fval = fungrad_eval!(fun.fun, x, g); g .*= fun.scale; return fun.scale*fval)

struct PlusDifferentiableFunction{T,N}<:DifferentiableFunction{T,N}
    fun1::DifferentiableFunction{T,N}
    fun2::DifferentiableFunction{T,N}
end

Base.:+(fun1::DifferentiableFunction{T,N}, fun2::DifferentiableFunction{T,N}) where {T,N} = PlusDifferentiableFunction{T,N}(fun1, fun2)

fun_eval(fun::PlusDifferentiableFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = fun.fun1(x)+fun.fun2(x)
grad_eval!(fun::PlusDifferentiableFunction{T,N}, x::AbstractArray{T,N}, g::AbstractArray{T,N}) where {T,N} = (grad_eval!(fun.fun1, x, g); g .+= grad_eval(fun.fun2, x); return g)
function fungrad_eval!(fun::PlusDifferentiableFunction{T,N}, x::AbstractArray{T,N}, g::AbstractArray{T,N}) where {T,N}
    fval = fungrad_eval!(fun.fun1, x, g)
    fval2, g2 = fungrad_eval(fun.fun2, x)
    g .+= g2
    return fval+fval2
end

struct MinusDifferentiableFunction{T,N}<:DifferentiableFunction{T,N}
    fun1::DifferentiableFunction{T,N}
    fun2::DifferentiableFunction{T,N}
end

Base.:-(fun1::DifferentiableFunction{T,N}, fun2::DifferentiableFunction{T,N}) where {T,N} = MinusDifferentiableFunction{T,N}(fun1, fun2)

fun_eval(fun::MinusDifferentiableFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = fun.fun1(x)-fun.fun2(x)
grad_eval!(fun::MinusDifferentiableFunction{T,N}, x::AbstractArray{T,N}, g::AbstractArray{T,N}) where {T,N} = (grad_eval!(fun.fun1, x, g); g .-= grad_eval(fun.fun2, x); return g)
function fungrad_eval!(fun::MinusDifferentiableFunction{T,N}, x::AbstractArray{T,N}, g::AbstractArray{T,N}) where {T,N}
    fval = fungrad_eval!(fun.fun1, x, g)
    fval2, g2 = fungrad_eval(fun.fun2, x)
    g .-= g2
    return fval-fval2
end


# Proximable linear algebra

struct ScaledProximableFunction{T,N}<:ProximableFunction{T,N}
    scale::Real
    fun::ProximableFunction{T,N}
end

Base.:*(scale::T, fun::ProximableFunction{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = ScaledProximableFunction{CT,N}(scale, fun)
Base.:*(fun::ProximableFunction{CT,N}, scale::T) where {T<:Real,N,CT<:RealOrComplex{T}} = scale*fun
Base.:/(fun::ProximableFunction{CT,N}, scale::T) where {T<:Real,N,CT<:RealOrComplex{T}} = (1/scale)*fun

fun_eval(fun::ScaledProximableFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = fun.scale*fun.fun(x)

proxy!(y::AbstractArray{CT,N}, λ::T, g::ScaledProximableFunction{CT,N}, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = proxy!(y, λ*g.scale, g.fun, x)
project!(y::AbstractArray{CT,N}, ε::T, g::ScaledProximableFunction{CT,N}, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = project!(y, g.scale/ε, g.fun, x)


# Conjugation of proximable functions

struct ConjugateProximableFunction{T,N}<:ProximableFunction{T,N}
    fun::ProximableFunction{T,N}
end

conjugate(g::ProximableFunction{T,N}) where {T,N} = ConjugateProximableFunction{T,N}(g)
conjugate(g::ConjugateProximableFunction) = g.fun

function proxy!(y::AbstractArray{CT,N}, λ::T, g::ConjugateProximableFunction{CT,N}, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}}
    proxy!(y/λ, 1/λ, g.fun, x)
    return x .= y-λ*x
end


# Mixed algebra

struct DiffPlusProxFunction{T,N}<:MinimizableFunction{T,N}
    diff::DifferentiableFunction{T,N}
    prox::ProximableFunction{T,N}
end

Base.:+(f::DifferentiableFunction{T,N}, g::ProximableFunction{T,N}) where {T,N} = DiffPlusProxFunction{T,N}(f, g)
Base.:+(g::ProximableFunction{T,N}, f::DifferentiableFunction{T,N}) where {T,N} = f+g