#: Utilities for norm functions

export NullProx, null_prox, MixedNorm, MixedNormBatch, ptdot, ptnorm1, ptnorm2, ptnormInf, mixed_norm


# Null proximable function

struct NullProx{T,N}<:ProximableFunction{T,N} end

null_prox(T::DataType, N::Integer) = NullProx{T,N}()

proxy!(p::AbstractArray{CT,N}, ::T, ::NullProx{CT,N}, q::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = (q .= p)

project!(p::AbstractArray{CT,N}, ::T, ::NullProx{CT,N}, q::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = (q .= p)


# Mixed norm

struct MixedNorm{T,D,N1,N2}<:ProximableFunction{T,D}
    pareto_tol::Union{Nothing,Real}
end

mixed_norm(T::DataType, D::Number, N1::Number, N2::Number; pareto_tol::Union{Nothing,Real}=nothing) = MixedNorm{T,D+1,N1,N2}(pareto_tol)


# Mixed norm (batch)

struct MixedNormBatch{T,D,N1,N2}<:ProximableFunction{T,D}
    pareto_tol::Union{Nothing,Real}
end

mixed_norm_batch(T::DataType, D::Number, N1::Number, N2::Number; pareto_tol::Union{Nothing,Real}=nothing) = MixedNormBatch{T,D+2,N1,N2}(pareto_tol)


# L22 norm

function proxy!(p::AbstractArray{CT,N}, λ::T, ::MixedNorm{CT,N,2,2}, q::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}}
    np = norm22(p)
    np <= λ ? (return q .= CT(0)) : (return q .= (1-λ/np)*p)
end

function project!(p::AbstractArray{CT,N}, ε::T, ::MixedNorm{CT,N,2,2}, q::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}}
    np = norm22(p)
    np <= ε ? (return q .= p) : (return q .= ε*p/np)
end

(::MixedNorm{T,N,2,2})(p::AbstractArray{T,N}) where {T,N} = norm22(p)


# L21 norm

function proxy!(p::AbstractArray{CT,N}, λ::T, ::MixedNorm{CT,N,2,1}, q::AbstractArray{CT,N}; ptn::Union{AbstractArray{T,N},Nothing}=nothing) where {T<:Real,N,CT<:RealOrComplex{T}}
    ptn === nothing && (ptn = ptnorm2(p; η=eps(T)))
    return q .= (1 .-λ./ptn).*(ptn .>= λ).*p
end

function project!(p::AbstractArray{CT,N}, ε::T, g::MixedNorm{CT,N,2,1}, q::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}}
    ptn = ptnorm2(p; η=eps(T))
    sum(ptn) <= ε && (return q .= p)
    λ = pareto_search_proj21(ptn, ε; xrtol=g.pareto_tol)
    return proxy!(p, λ, g, q; ptn=ptn)
end

(::MixedNorm{T,N,2,1})(p::AbstractArray{T,N}) where {T,N} = norm21(p)


# L21 norm (batch, 2-D)

function proxy!(p::AbstractArray{CT,4}, λ::T, ::MixedNormBatch{CT,4,2,1}, q::AbstractArray{CT,4}; ptn::Union{AbstractArray{CT,4},Nothing}=nothing) where {T<:Real,CT<:RealOrComplex{T}}
    ptn === nothing && (ptn = ptnorm2_batch(p; η=eps(T)))
    nx,ny,nc,nb = size(p)
    p = reshape(p, nx,ny,2,div(nc,2)*nb)
    q = reshape(q, nx,ny,2,div(nc,2)*nb)
    ptn = reshape(ptn, nx,ny,1,div(nc,2)*nb)
    q .= (CT(1).-λ./ptn).*(ptn .>= λ).*p
    return reshape(q, nx,ny,nc,nb)
end

(::MixedNormBatch{T,4,2,1})(p::AbstractArray{T,4}) where T = norm21_batch(p)


# L2Inf norm

function proxy!(p::AbstractArray{CT,N}, λ::T, ::MixedNorm{CT,N,2,Inf}, q::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}}
    project!(p, λ, mixed_norm(CT, N-1, 2, 1), q)
    return q .= p.-q
end

function project!(p::AbstractArray{CT,N}, ε::T, ::MixedNorm{CT,N,2,Inf}, q::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}}
    ptn = ptnorm2(p; η=eps(T))
    val = ptn .>= ε
    q .= p.*(ε*val./ptn+(!).(val))
    return q
end

(::MixedNorm{T,N,2,Inf})(p::AbstractArray{T,N}) where {T,N} = norm2Inf(p)


# Pareto-search routines

pareto_search_proj21(ptn::AbstractArray{T,N}, ε::T; xrtol::Union{Nothing,T}=nothing) where {T<:Real,N} = T(solve(ZeroProblem(λ -> obj_pareto_search_proj21(λ, ptn, ε), (T(0), maximum(ptn))), Roots.Brent(); xreltol=isnothing(xrtol) ? eps(T) : xrtol))

obj_pareto_search_proj21(λ::T, ptn::AbstractArray{T,N}, ε::T) where {T<:Real,N} = sum(Flux.relu.(ptn.-λ))-ε


# Algebraic utils

ptdot(v1::AbstractArray{T,N}, v2::AbstractArray{T,N}) where {T,N} = sum(v1.*conj.(v2); dims=N)
ptnorm1(p::AbstractArray{CT,N}; η::T=T(0)) where {T<:Real,N,CT<:RealOrComplex{T}} = sum(abs.(p).+η; dims=N)
ptnorm2(p::AbstractArray{CT,N}; η::T=T(0)) where {T<:Real,N,CT<:RealOrComplex{T}} = sqrt.(sum(abs.(p).^2 .+η^2; dims=N))
ptnormInf(p::AbstractArray{CT,N}; η::T=T(0)) where {T<:Real,N,CT<:RealOrComplex{T}} = sqrt.(maximum(abs.(p).+η; dims=N))
function ptnorm2_batch(p::AbstractArray{CT,4}; η::T=T(0)) where {T<:Real,CT<:RealOrComplex{T}}
    nx,ny,nc,nb = size(p)
    p = reshape(p, nx,ny,2,div(nc,2)*nb)
    return reshape(sqrt.(abs.(p[:,:,1:1,:]).^2+abs.(p[:,:,2:2,:]).^2 .+η^2), nx,ny,div(nc,2),nb)
end
function ptnorm2_batch(p::AbstractArray{CT,5}; η::T=T(0)) where {T<:Real,CT<:RealOrComplex{T}}
    nx,ny,nz,nc,nb = size(p)
    p = reshape(p, nx,ny,nz,2,div(nc,2)*nb)
    return reshape(sqrt.(abs.(p[:,:,1:1,:]).^2+abs.(p[:,:,2:2,:]).^2+abs.(p[:,:,3:3,:]).^2 .+η^2), nx,ny,div(nc,2),nb)
end
norm21(v::AbstractArray{CT,N}; η::T=T(0)) where {T<:Real,N,CT<:RealOrComplex{T}} = sum(ptnorm2(v; η=η))
norm22(v::AbstractArray{CT,N}; η::T=T(0)) where {T<:Real,N,CT<:RealOrComplex{T}} = sqrt(sum(ptnorm2(v; η=η).^2))
norm2Inf(v::AbstractArray{CT,N}; η::T=T(0)) where {T<:Real,N,CT<:RealOrComplex{T}} = maximum(ptnorm2(v; η=η))
function norm21_batch(v::AbstractArray{CT,4}; η::T=T(0)) where {T<:Real,CT<:RealOrComplex{T}}
    _,_,nc,nb = size(v)
    return reshape(sum(ptnorm2_batch(v; η=η); dims=(1,2)), div(nc,2), nb)
end
function norm21_batch(v::AbstractArray{CT,5}; η::T=T(0)) where {T<:Real,CT<:RealOrComplex{T}}
    _,_,_,nc,nb = size(v)
    return reshape(sum(ptnorm2_batch(v; η=η); dims=(1,2,3)), div(nc,2), nb)
end