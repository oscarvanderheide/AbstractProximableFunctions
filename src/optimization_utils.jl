# Optimization utilities

export OptionsFISTA, ConjugateAndFISTA, ConjugateProjectAndFISTA, FISTA, conj_FISTA, conjproj_FISTA, Lipschitz_constant, set_Lipschitz_constant, fun_history, verbose
export leastsquares_solve, leastsquares_solve!
export spectral_radius


## FISTA options

mutable struct OptionsFISTA{T<:Real}<:AbstractArgMinOptions
    Lipschitz_constant::T
    Nesterov::Bool
    reset_counter::Union{Nothing,Integer}
    niter::Union{Nothing,Integer}
    verbose::Bool
    fun_history::Union{Nothing,AbstractVector{T}}
end

struct ConjugateAndFISTA{T<:Real}<:AbstractArgMinOptions
    options_FISTA::OptionsFISTA{T}
end

struct ConjugateProjectAndFISTA{T<:Real}<:AbstractArgMinOptions
    options_FISTA::OptionsFISTA{T}
end

function FISTA(L::T;
               Nesterov::Bool=true,
               reset_counter::Union{Nothing,Integer}=nothing,
               niter::Union{Nothing,Integer}=nothing,
               verbose::Bool=false,
               fun_history::Bool=false) where {T<:Real}
    (fun_history && ~isnothing(niter)) ? (fval = Array{T,1}(undef,niter)) : (fval = nothing)
    options = OptionsFISTA{T}(L, Nesterov, reset_counter, niter, verbose, fval)
    return options
end

conj_FISTA(args...; kwargs...) = ConjugateAndFISTA(FISTA(args...; kwargs...))
conjproj_FISTA(args...; kwargs...) = ConjugateProjectAndFISTA(FISTA(args...; kwargs...))


## Setter/Getter

set_Lipschitz_constant(options::OptionsFISTA{T}, L::T) where {T<:Real} = OptionsFISTA{T}(L, options.Nesterov, options.reset_counter, options.niter, options.verbose, options.fun_history)
Lipschitz_constant(options::OptionsFISTA) = options.Lipschitz_constant
verbose(options::OptionsFISTA) = options.verbose
fun_history(options::OptionsFISTA) = options.fun_history


## Least-squares linear problem routines

leastsquares_solve!(A::AbstractLinearOperator{CT,N1,N2}, b::AbstractArray{CT,N2}, g::AbstractProximableFunction{CT,N1}, initial_estimate::AbstractArray{CT,N1}, options::OptionsFISTA{T}, x::AbstractArray{CT,N1}) where {T<:Real,N1,N2,CT<:RealOrComplex{T}} = argmin!(leastsquares_misfit(A, b)+g, initial_estimate, options, x)

leastsquares_solve(A::AbstractLinearOperator{CT,N1,N2}, b::AbstractArray{CT,N2}, g::AbstractProximableFunction{CT,N1}, initial_estimate::AbstractArray{CT,N1}, options::OptionsFISTA{T}) where {T<:Real,N1,N2,CT<:RealOrComplex{T}} = leastsquares_solve!(A, b, g, initial_estimate, options, similar(initial_estimate))


# Other utils

function spectral_radius(A::Union{AbstractMatrix{T},AbstractLinearOperator{T,N,N}}; x::Union{Nothing,AbstractArray{T,N}}=nothing, niter::Int64=10) where {T,N}
    if isnothing(x)
        A isa AbstractMatrix && (x = randn(T, size(A,2)))
        A isa AbstractLinearOperator && (x = randn(T, domain_size(A)))
    end
    x = x/norm(x)
    y = similar(x)
    ρ = real(T)(0)
    for _ = 1:niter
        y .= A*x
        ρ = norm(y)/norm(x)
        x .= y/norm(y)
    end
    return ρ
end