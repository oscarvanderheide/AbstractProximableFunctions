#: Utils

export conjugate, proxy_objfun, proj_objfun, weighted_prox
export no_constraints, indicator


# Scaled version of proximable/projectionable functions

struct ScaledProximableFun{T,N}<:ProximableFunction{T,N}
    c::Real
    prox::ProximableFunction{T,N}
end

proxy!(y::AbstractArray{CT,N}, λ::T, g::ScaledProximableFun{CT,N}, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = proxy!(y, λ*g.c, g.prox, x)
project!(y::AbstractArray{CT,N}, ε::T, g::ScaledProximableFun{CT,N}, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = project!(y, g.c/ε, g.prox, x)


# LinearAlgebra

Base.:*(c::T, g::ProximableFunction{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = ScaledProximableFun{CT,N}(c, g)
Base.:*(c::T, g::ScaledProximableFun{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = ScaledProximableFun{CT,N}(c*g.c, g.prox)
Base.:/(g::ProximableFunction{CT,N}, c::T) where {T<:Real,N,CT<:RealOrComplex{T}} = ScaledProximableFun{CT,N}(1/c, g)
Base.:/(g::ScaledProximableFun{CT,N}, c::T) where {T<:Real,N,CT<:RealOrComplex{T}} = ScaledProximableFun{CT,N}(g.c/c, g.prox)


# Conjugation of proximable functions

struct ConjugateProxFun{T,N}<:ProximableFunction{T,N}
    prox::ProximableFunction{T,N}
end

function proxy!(y::AbstractArray{CT,N}, λ::T, g::ConjugateProxFun{CT,N}, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}}
    proxy!(y/λ, 1/λ, g.prox, x)
    x .= y-λ*x
    return x
end

conjugate(g::ProximableFunction{T,N}) where {T,N} = ConjugateProxFun{T,N}(g)
conjugate(g::ConjugateProxFun) = g.g


# Constraint sets

## No constraints

struct NoConstraints{T,N}<:ProjectionableSet{T,N} end

no_constraints(T::DataType, N::Int64) = NoConstraints{T,N}()

project!(x::AbstractArray{T,N}, ::NoConstraints{T,N}, y::Array{T,N}) where {T,N} = (y .= x)

## Sub-level sets with proximable functions

"""
Constraint set C = {x:g(x)<=ε}
"""
struct SubLevelSet{T,N}<:ProjectionableSet{T,N}
    g::ProximableFunction{T,N}
    ε::Real
end

Base.:≤(g::ProximableFunction{CT,N}, ε::T) where {T<:Real,N,CT<:RealOrComplex{T}} = SubLevelSet{CT,N}(g, ε)

project!(x::AbstractArray{T,N}, C::SubLevelSet{T,N}, y::AbstractArray{T,N}) where {T,N} = project!(x, C.ε, C.g, y)
project!(x::AbstractArray{T,N}, C::SubLevelSet{T,N}, y::AbstractArray{T,N}, opt::AbstractOptimizer) where {T,N} = project!(x, C.ε, C.g, y, opt)

## Indicator function

"""
Indicator function δ_C(x) = {0, if x ∈ C; ∞, otherwise} for convex sets C
"""
struct IndicatorFunction{T,N}<:ProximableFunction{T,N}
    C::ProjectionableSet{T,N}
end

indicator(C::ProjectionableSet{T,N}) where {T,N} = IndicatorFunction{T,N}(C)

proxy!(y::AbstractArray{CT,N}, ::T, δ::IndicatorFunction{CT,N}, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = project!(y, δ.C, x)
proxy!(y::AbstractArray{CT,N}, ::T, δ::IndicatorFunction{CT,N}, x::AbstractArray{CT,N}, opt::AbstractOptimizer) where {T<:Real,N,CT<:RealOrComplex{T}} = project!(y, δ.C, x, opt)


# Proximable function evaluation

struct ProxyObjFun{T,N} <: DifferentiableFunction{T,N}
    λ::Real
    g::ProximableFunction{T,N}
    opt::Union{Nothing,AbstractOptimizer}
end

proxy_objfun(λ::T, g::ProximableFunction{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = ProxyObjFun{CT,N}(λ, g, nothing)
proxy_objfun(λ::T, g::ProximableFunction{CT,N}, opt::AbstractOptimizer) where {T<:Real,N,CT<:RealOrComplex{T}} = ProxyObjFun{CT,N}(λ, g, opt)

function funeval!(f::ProxyObjFun{CT,N}, y::AbstractArray{CT,N}; gradient::Union{Nothing,AbstractArray{CT,N}}=nothing, eval::Bool=false) where {T<:Real,N,CT<:RealOrComplex{T}}
    f.opt === nothing ? (x = proxy(y, f.λ, f.g)) : (x = proxy(y, f.λ, f.g, f.opt))
    ~isnothing(gradient) && (gradient .= y-x)
    eval ? (return T(0.5)*norm(x-y)^2+f.λ*f.g(x)) : (return nothing)
end


struct ProjObjFun{T,N}<:DifferentiableFunction{T,N}
    ε::Real
    g::ProximableFunction{T,N}
    opt::Union{Nothing,AbstractOptimizer}
end

proj_objfun(ε::T, g::ProximableFunction{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = ProjObjFun{CT,N}(ε, g, nothing)
proj_objfun(ε::T, g::ProximableFunction{CT,N}, opt::AbstractOptimizer) where {T<:Real,N,CT<:RealOrComplex{T}} = ProjObjFun{CT,N}(ε, g, opt)

function funeval!(f::ProjObjFun{CT,N}, y::AbstractArray{CT,N}; gradient::Union{Nothing,AbstractArray{CT,N}}=nothing, eval::Bool=false) where {T<:Real,N,CT<:RealOrComplex{T}}
    f.opt === nothing ? (x = project(y, f.ε, f.g)) : (x = project(y, f.ε, f.g, f.opt))
    ~isnothing(gradient) && (gradient .= y-x)
    eval ? (return T(0.5)*norm(x-y)^2) : (return nothing)
end


# Minimizable type utils

struct DiffPlusProxFun{T,N}<:MinimizableFunction{T,N}
    diff::DifferentiableFunction{T,N}
    prox::ProximableFunction{T,N}
end

Base.:+(f::DifferentiableFunction{T,N}, g::ProximableFunction{T,N}) where {T,N} = DiffPlusProxFun{T,N}(f, g)
Base.:+(g::ProximableFunction{T,N}, f::DifferentiableFunction{T,N}) where {T,N} = f+g