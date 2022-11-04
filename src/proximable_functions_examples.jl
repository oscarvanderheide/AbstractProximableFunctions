#: Examples of proximable functions

export ZeroProx, zero_prox, Norm, MixedNorm, mixed_norm


# Zero function

struct ZeroProx{T,N}<:AbstractProximableFunction{T,N} end

zero_prox(T::DataType, N::Number) = ZeroProx{T,N}()

fun_eval(::ZeroProx{CT,N}, ::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = T(0)

get_optimizer(::ZeroProx) = nothing

proxy!(p::AbstractArray{CT,N}, ::T, ::ZeroProx{CT,N}, q::AbstractArray{CT,N}; optimizer::Union{Nothing,AbstractConvexOptimizer}=nothing) where {T<:Real,N,CT<:RealOrComplex{T}} = (return q .= p)

project!(p::AbstractArray{CT,N}, ::T, ::ZeroProx{CT,N}, q::AbstractArray{CT,N}; optimizer::Union{Nothing,AbstractConvexOptimizer}=nothing) where {T<:Real,N,CT<:RealOrComplex{T}} = (return q .= p)


# Norms

struct Norm{T,N,P}<:AbstractProximableFunction{T,N}
    pareto_tol::Union{Nothing,Real}
end

LinearAlgebra.norm(T::DataType, N::Number, P::Number; pareto_tol::Union{Nothing,Real}=nothing) = Norm{T,N,P}(pareto_tol)

fun_eval(::Norm{T,N,P}, x::AbstractArray{T,N}) where {T,N,P} = norm(x, P)

get_optimizer(::Norm) = nothing


## L2 norm

function proxy!(y::AbstractArray{CT,N}, λ::T, g::Norm{CT,N,2}, x::AbstractArray{CT,N}; optimizer::Union{Nothing,AbstractConvexOptimizer}=nothing) where {T<:Real,N,CT<:RealOrComplex{T}}
    ny = g(y)
    ny <= λ ? (x .= 0) : (x .= (1-λ/ny)*y)
    return x
end

function project!(y::AbstractArray{CT,N}, ε::T, g::Norm{CT,N,2}, x::AbstractArray{CT,N}; optimizer::Union{Nothing,AbstractConvexOptimizer}=nothing) where {T<:Real,N,CT<:RealOrComplex{T}}
    ny = g(y)
    ny <= ε ? (x .= y) : (x .= ε*y/ny)
    return x
end


## L1 norm

function proxy!(y::AbstractArray{CT,N}, λ::T, ::Norm{CT,N,1}, x::AbstractArray{CT,N}; optimizer::Union{Nothing,AbstractConvexOptimizer}=nothing, abs_y::Union{Nothing,AbstractArray{T,N}}=nothing) where {T<:Real,N,CT<:RealOrComplex{T}}
    isnothing(abs_y) && (abs_y = abs.(y))
    return x .= (1 .-λ./abs_y).*(abs_y .>= λ).*y
end

function project!(y::AbstractArray{CT,N}, ε::T, g::Norm{CT,N,1}, x::AbstractArray{CT,N}; optimizer::Union{Nothing,AbstractConvexOptimizer}=nothing) where {T<:Real,N,CT<:RealOrComplex{T}}
    abs_y = abs.(y)
    sum(abs_y) <= ε && (return x .= y)
    λ = pareto_search_projL1(abs_y, ε; xrtol=g.pareto_tol)
    return proxy!(y, λ, g, x; abs_y=abs_y)
end


## LInf norm

function proxy!(y::AbstractArray{CT,N}, λ::T, ::Norm{CT,N,Inf}, x::AbstractArray{CT,N}; optimizer::Union{Nothing,AbstractConvexOptimizer}=nothing) where {T<:Real,N,CT<:RealOrComplex{T}}
    project!(y, λ, norm(CT, N, 1), x)
    return x .= y-x
end

function project!(y::AbstractArray{CT,N}, ε::T, ::Norm{CT,N,Inf}, x::AbstractArray{CT,N}; optimizer::Union{Nothing,AbstractConvexOptimizer}=nothing) where {T<:Real,N,CT<:RealOrComplex{T}}
    idx = abs.(y) .> ε
    x[idx] .= ε*y[idx]./abs.(y[idx])
    x[(!).(idx)] .= y[(!).(idx)]
    return x
end


# Mixed norms

struct MixedNorm{T,D,N1,N2}<:AbstractProximableFunction{T,D}
    pareto_tol::Union{Nothing,Real}
end

mixed_norm(T::DataType, D::Number, N1::Number, N2::Number; pareto_tol::Union{Nothing,Real}=nothing) = MixedNorm{T,D+1,N1,N2}(pareto_tol)

get_optimizer(::MixedNorm) = nothing


## L22 norm

fun_eval(::MixedNorm{T,N,2,2}, p::AbstractArray{T,N}) where {T,N} = norm22(p)

function proxy!(p::AbstractArray{CT,N}, λ::T, ::MixedNorm{CT,N,2,2}, q::AbstractArray{CT,N}; optimizer::Union{Nothing,AbstractConvexOptimizer}=nothing) where {T<:Real,N,CT<:RealOrComplex{T}}
    np = norm22(p)
    np <= λ ? (q .= 0) : (q .= (1-λ/np)*p)
    return q
end

function project!(p::AbstractArray{CT,N}, ε::T, ::MixedNorm{CT,N,2,2}, q::AbstractArray{CT,N}; optimizer::Union{Nothing,AbstractConvexOptimizer}=nothing) where {T<:Real,N,CT<:RealOrComplex{T}}
    np = norm22(p)
    np <= ε ? (q .= p) : (q .= ε*p/np)
    return q
end


## L21 norm

fun_eval(::MixedNorm{T,N,2,1}, p::AbstractArray{T,N}) where {T,N} = norm21(p)

function proxy!(p::AbstractArray{CT,N}, λ::T, ::MixedNorm{CT,N,2,1}, q::AbstractArray{CT,N}; ptn::Union{AbstractArray{T,N},Nothing}=nothing, optimizer::Union{Nothing,AbstractConvexOptimizer}=nothing) where {T<:Real,N,CT<:RealOrComplex{T}}
    ptn === nothing && (ptn = ptnorm2(p; η=eps(T)))
    return q .= (1 .-λ./ptn).*(ptn .>= λ).*p
end

function project!(p::AbstractArray{CT,N}, ε::T, g::MixedNorm{CT,N,2,1}, q::AbstractArray{CT,N}; optimizer::Union{Nothing,AbstractConvexOptimizer}=nothing) where {T<:Real,N,CT<:RealOrComplex{T}}
    ptn = ptnorm2(p; η=eps(T))
    sum(ptn) <= ε && (return q .= p)
    λ = pareto_search_projL1(ptn, ε; xrtol=g.pareto_tol)
    return proxy!(p, λ, g, q; ptn=ptn, optimizer=optimizer)
end


## L2Inf norm

fun_eval(::MixedNorm{T,N,2,Inf}, p::AbstractArray{T,N}) where {T,N} = norm2Inf(p)

function proxy!(p::AbstractArray{CT,N}, λ::T, ::MixedNorm{CT,N,2,Inf}, q::AbstractArray{CT,N}; optimizer::Union{Nothing,AbstractConvexOptimizer}=nothing) where {T<:Real,N,CT<:RealOrComplex{T}}
    project!(p, λ, mixed_norm(CT, N-1, 2, 1), q; optimizer=optimizer)
    return q .= p.-q
end

function project!(p::AbstractArray{CT,N}, ε::T, ::MixedNorm{CT,N,2,Inf}, q::AbstractArray{CT,N}; optimizer::Union{Nothing,AbstractConvexOptimizer}=nothing) where {T<:Real,N,CT<:RealOrComplex{T}}
    ptn = ptnorm2(p; η=eps(T))
    val = ptn .>= ε
    q .= p.*(ε*val./ptn+(!).(val))
    return q
end


# Pareto-search routines

pareto_search_projL1(ptn::AbstractArray{T,N}, ε::T; xrtol::Union{Nothing,T}=nothing) where {T<:Real,N} = T(solve(ZeroProblem(λ -> obj_pareto_search_projL1(λ, ptn, ε), (T(0), maximum(ptn))), Roots.Brent(); xreltol=isnothing(xrtol) ? eps(T) : xrtol))

obj_pareto_search_projL1(λ::T, ptn::AbstractArray{T,N}, ε::T) where {T<:Real,N} = sum(relu(ptn.-λ))-ε

relu(x::AbstractArray{T,N}) where {T<:Real,N} = x.*(x .> 0)


# Norm utils

ptdot(v1::AbstractArray{T,N}, v2::AbstractArray{T,N}) where {T,N} = sum(v1.*conj.(v2); dims=N)
ptnorm1(p::AbstractArray{CT,N}; η::T=T(0)) where {T<:Real,N,CT<:RealOrComplex{T}} = sum(abs.(p).+η; dims=N)
ptnorm2(p::AbstractArray{CT,N}; η::T=T(0)) where {T<:Real,N,CT<:RealOrComplex{T}} = sqrt.(sum(abs.(p).^2 .+η^2; dims=N))
ptnormInf(p::AbstractArray{CT,N}; η::T=T(0)) where {T<:Real,N,CT<:RealOrComplex{T}} = sqrt.(maximum(abs.(p).+η; dims=N))
norm21(v::AbstractArray{CT,N}; η::T=T(0)) where {T<:Real,N,CT<:RealOrComplex{T}} = sum(ptnorm2(v; η=η))
norm22(v::AbstractArray{CT,N}; η::T=T(0)) where {T<:Real,N,CT<:RealOrComplex{T}} = sqrt(sum(ptnorm2(v; η=η).^2))
norm2Inf(v::AbstractArray{CT,N}; η::T=T(0)) where {T<:Real,N,CT<:RealOrComplex{T}} = maximum(ptnorm2(v; η=η))