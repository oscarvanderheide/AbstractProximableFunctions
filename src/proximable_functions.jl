#: Examples of proximable functions

export ZeroProx, zero_prox, MixedNorm, MixedNormBatch, ptdot, ptnorm1, ptnorm2, ptnormInf, mixed_norm


# Mixed norms

struct ZeroProx{T,N}<:ProximableFunction{T,N} end

zero_prox(T::DataType, N::Number) = ZeroProx{T,N}()

fun_eval(::ZeroProx{CT,N}, ::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = T(0)

get_optimizer(::ZeroProx) = nothing

proxy!(p::AbstractArray{CT,N}, ::T, ::ZeroProx{CT,N}, q::AbstractArray{CT,N}; optimizer::Union{Nothing,Optimizer}=nothing) where {T<:Real,N,CT<:RealOrComplex{T}} = (return q .= p)

project!(p::AbstractArray{CT,N}, ::T, ::ZeroProx{CT,N}, q::AbstractArray{CT,N}; optimizer::Union{Nothing,Optimizer}=nothing) where {T<:Real,N,CT<:RealOrComplex{T}} = (return q .= p)


# Mixed norms

struct MixedNorm{T,D,N1,N2}<:ProximableFunction{T,D}
    pareto_tol::Union{Nothing,Real}
end

mixed_norm(T::DataType, D::Number, N1::Number, N2::Number; pareto_tol::Union{Nothing,Real}=nothing) = MixedNorm{T,D+1,N1,N2}(pareto_tol)

get_optimizer(::MixedNorm) = nothing


# L22 norm

fun_eval(::MixedNorm{T,N,2,2}, p::AbstractArray{T,N}) where {T,N} = norm22(p)

function proxy!(p::AbstractArray{CT,N}, λ::T, ::MixedNorm{CT,N,2,2}, q::AbstractArray{CT,N}; optimizer::Union{Nothing,Optimizer}=nothing) where {T<:Real,N,CT<:RealOrComplex{T}}
    np = norm22(p)
    np <= λ ? (q .= 0) : (q .= (1-λ/np)*p)
    return q
end

function project!(p::AbstractArray{CT,N}, ε::T, ::MixedNorm{CT,N,2,2}, q::AbstractArray{CT,N}; optimizer::Union{Nothing,Optimizer}=nothing) where {T<:Real,N,CT<:RealOrComplex{T}}
    np = norm22(p)
    np <= ε ? (q .= p) : (q .= ε*p/np)
    return q
end


# L21 norm

fun_eval(::MixedNorm{T,N,2,1}, p::AbstractArray{T,N}) where {T,N} = norm21(p)

function proxy!(p::AbstractArray{CT,N}, λ::T, ::MixedNorm{CT,N,2,1}, q::AbstractArray{CT,N}; ptn::Union{AbstractArray{T,N},Nothing}=nothing, optimizer::Union{Nothing,Optimizer}=nothing) where {T<:Real,N,CT<:RealOrComplex{T}}
    ptn === nothing && (ptn = ptnorm2(p; η=eps(T)))
    return q .= (1 .-λ./ptn).*(ptn .>= λ).*p
end

function project!(p::AbstractArray{CT,N}, ε::T, g::MixedNorm{CT,N,2,1}, q::AbstractArray{CT,N}; optimizer::Union{Nothing,Optimizer}=nothing) where {T<:Real,N,CT<:RealOrComplex{T}}
    ptn = ptnorm2(p; η=eps(T))
    sum(ptn) <= ε && (return q .= p)
    λ = pareto_search_proj21(ptn, ε; xrtol=g.pareto_tol)
    return proxy!(p, λ, g, q; ptn=ptn, optimizer=optimizer)
end


# L2Inf norm

fun_eval(::MixedNorm{T,N,2,Inf}, p::AbstractArray{T,N}) where {T,N} = norm2Inf(p)

function proxy!(p::AbstractArray{CT,N}, λ::T, ::MixedNorm{CT,N,2,Inf}, q::AbstractArray{CT,N}; optimizer::Union{Nothing,Optimizer}=nothing) where {T<:Real,N,CT<:RealOrComplex{T}}
    project!(p, λ, mixed_norm(CT, N-1, 2, 1), q; optimizer=optimizer)
    return q .= p.-q
end

function project!(p::AbstractArray{CT,N}, ε::T, ::MixedNorm{CT,N,2,Inf}, q::AbstractArray{CT,N}; optimizer::Union{Nothing,Optimizer}=nothing) where {T<:Real,N,CT<:RealOrComplex{T}}
    ptn = ptnorm2(p; η=eps(T))
    val = ptn .>= ε
    q .= p.*(ε*val./ptn+(!).(val))
    return q
end


# Pareto-search routines

pareto_search_proj21(ptn::AbstractArray{T,N}, ε::T; xrtol::Union{Nothing,T}=nothing) where {T<:Real,N} = T(solve(ZeroProblem(λ -> obj_pareto_search_proj21(λ, ptn, ε), (T(0), maximum(ptn))), Roots.Brent(); xreltol=isnothing(xrtol) ? eps(T) : xrtol))

obj_pareto_search_proj21(λ::T, ptn::AbstractArray{T,N}, ε::T) where {T<:Real,N} = sum(relu(ptn.-λ))-ε

relu(x::AbstractArray{T,N}) where {T<:Real,N} = x.*(x .> 0)


# Norm utils

ptdot(v1::AbstractArray{T,N}, v2::AbstractArray{T,N}) where {T,N} = sum(v1.*conj.(v2); dims=N)
ptnorm1(p::AbstractArray{CT,N}; η::T=T(0)) where {T<:Real,N,CT<:RealOrComplex{T}} = sum(abs.(p).+η; dims=N)
ptnorm2(p::AbstractArray{CT,N}; η::T=T(0)) where {T<:Real,N,CT<:RealOrComplex{T}} = sqrt.(sum(abs.(p).^2 .+η^2; dims=N))
ptnormInf(p::AbstractArray{CT,N}; η::T=T(0)) where {T<:Real,N,CT<:RealOrComplex{T}} = sqrt.(maximum(abs.(p).+η; dims=N))
norm21(v::AbstractArray{CT,N}; η::T=T(0)) where {T<:Real,N,CT<:RealOrComplex{T}} = sum(ptnorm2(v; η=η))
norm22(v::AbstractArray{CT,N}; η::T=T(0)) where {T<:Real,N,CT<:RealOrComplex{T}} = sqrt(sum(ptnorm2(v; η=η).^2))
norm2Inf(v::AbstractArray{CT,N}; η::T=T(0)) where {T<:Real,N,CT<:RealOrComplex{T}} = maximum(ptnorm2(v; η=η))