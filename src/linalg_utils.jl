#: Algebraic utils

export conjugate


# Proximable linear algebra

struct ScaledProxAndProjFunction{T,N}<:AbstractProximableFunction{T,N}
    scale::Real
    fun::AbstractProximableFunction{T,N}
end

Base.:*(scale::T, fun::AbstractProximableFunction{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = ScaledProxAndProjFunction{CT,N}(scale, fun)
Base.:*(fun::AbstractProximableFunction{CT,N}, scale::T) where {T<:Real,N,CT<:RealOrComplex{T}} = scale*fun
Base.:/(fun::AbstractProximableFunction{CT,N}, scale::T) where {T<:Real,N,CT<:RealOrComplex{T}} = (1/scale)*fun

funeval(fun::ScaledProxAndProjFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = fun.scale*fun.fun(x)

prox!(y::AbstractArray{CT,N}, λ::T, g::ScaledProxAndProjFunction{CT,N}, options::AbstractArgminOptions, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = prox!(y, λ*g.scale, g.fun, options, x)
proj!(y::AbstractArray{CT,N}, ε::T, g::ScaledProxAndProjFunction{CT,N}, options::AbstractArgminOptions, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = proj!(y, g.scale/ε, g.fun, options, x)


# Conjugation of proximable functions

struct ConjugateProxAndProjFunction{T,N}<:AbstractProximableFunction{T,N}
    fun::AbstractProximableFunction{T,N}
end

conjugate(g::AbstractProximableFunction{T,N}) where {T,N} = ConjugateProxAndProjFunction{T,N}(g)
conjugate(g::ConjugateProxAndProjFunction) = g.fun

function prox!(y::AbstractArray{CT,N}, λ::T, g::ConjugateProxAndProjFunction{CT,N}, options::AbstractArgminOptions, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}}
    prox!(y/λ, 1/λ, g.fun, options, x)
    return x .= y-λ*x
end


# Mixed diff/prox algebra

struct DiffPlusProxAndProjFunction{T,N}<:AbstractDifferentiablePlusProximableFunction{T,N}
    diff::AbstractDifferentiableFunction{T,N}
    prox::AbstractProximableFunction{T,N}
end

Base.:+(f::AbstractDifferentiableFunction{T,N}, g::AbstractProximableFunction{T,N}) where {T,N} = DiffPlusProxAndProjFunction{T,N}(f, g)
Base.:+(g::AbstractProximableFunction{T,N}, f::AbstractDifferentiableFunction{T,N}) where {T,N} = f+g

get_diff(fun::DiffPlusProxAndProjFunction) = fun.diff
get_prox(fun::DiffPlusProxAndProjFunction) = fun.prox