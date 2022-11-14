#: Optimization utilities

export ExactArgMin, ExactProxAndProj, ExactProjSet, exact_argmin, exact_proxproj, exact_projset
export ArgMinFISTA, FISTA_options, set_Lipschitz_constant


## Fallback behavior

not_implemented() = error("This method has not been implemented for this function!")

argmin!(::AbstractMinimizableFunction{T,N}, ::AT, ::AbstractArgminOptions, ::AT) where {T,N,AT<:AbstractArray{T,N}} = not_implemented()
prox!(::AT, ::T, ::AbstractProximableFunction{CT,N}, ::AbstractProximableOptions, ::AT) where {T<:Real,N,CT<:RealOrComplex{T},AT<:AbstractArray{CT,N}} = not_implemented()
proj!(::AT, ::T, ::AbstractProximableFunction{CT,N}, ::AbstractProximableOptions, ::AT) where {T<:Real,N,CT<:RealOrComplex{T},AT<:AbstractArray{CT,N}} = not_implemented()
proj!(::AT, ::AbstractProjectionableSet{T,N}, ::AbstractProjectionableSetOptions, ::AT) where {T,N,AT<:AbstractArray{T,N}} = not_implemented()


## Exact options

struct ExactArgMin<:AbstractArgminOptions{MT} end
struct ExactProxAndProj<:AbstractProximableOptions end
struct ExactProjSet<:AbstractProjectionableSetOptions end

exact_argmin() = ExactArgMin()
exact_proxproj() = ExactProxAndProj()
exact_projset() = ExactProjSet()


## FISTA options

mutable struct ArgMinFISTA<:AbstractArgminOptions
    Lipschitz_constant::Real
    Nesterov::Bool
    reset_counter::Union{Nothing,Integer}
    niter::Union{Nothing,Integer}
    verbose::Bool
    fun_history::Union{Nothing,AbstractVector{<:Real}}
end

function FISTA_options(L::T;
               Nesterov::Bool=true,
               reset_counter::Union{Nothing,Integer}=nothing,
               niter::Union{Nothing,Integer}=nothing,
               verbose::Bool=false,
               fun_history::Bool=false) where {T<:Real}
    (fun_history && ~isnothing(niter)) ? (fval = Array{T,1}(undef,niter)) : (fval = nothing)
    options = ArgMinFISTA(L, Nesterov, reset_counter, niter, verbose, fval)
    return options
end

Lipschitz_constant(options::ArgMinFISTA) = options.Lipschitz_constant
set_Lipschitz_constant(options::ArgMinFISTA, L::Real) = ArgMinFISTA(L, options.Nesterov, options.reset_counter, options.niter, options.verbose, options.fun_history)
Nesterov(options::ArgMinFISTA) = options.Nesterov
reset_counter(options::ArgMinFISTA) = options.reset_counter
niter(options::ArgMinFISTA) = options.niter
verbose(options::ArgMinFISTA) = options.verbose
fun_history(options::ArgMinFISTA) = options.fun_history

function argmin!(fun::AbstractDifferentiablePlusProximableFunction{CT,N}, initial_estimate::AT, options::ArgMinFISTA, x::AT) where {T<:Real,N,CT<:RealOrComplex{T},AT<:AbstractArray{CT,N}}
    # - FISTA: Beck, A., and Teboulle, M., 2009, A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems

    # Initialization
    x .= initial_estimate
    x_ = similar(x)
    g  = similar(x)
    L = T(Lipschitz_constant(options))
    isnothing(reset_counter(options)) ? (counter = nothing) : (counter = 0)
    t0 = T(1)
    diff_fun = diff_term(fun)
    prox_fun = prox_term(fun)

    # Optimization loop
    @inbounds for i = 1:niter(options)

        # Compute gradient
        if verbose(options) || ~isnothing(fun_history(options))
            fval_i = fungradeval!(diff_fun, x, g)
        else
            gradeval!(diff_fun, x, g)
        end
        ~isnothing(fun_history(options)) && (fun_history(options)[i] = fval_i)

        # Print current iteration
        verbose(options) && (@info string("Iter: ", i, ", fval: ", fval_i))

        # Update
        prox!(x-g/L, 1/L, prox_fun, x_)

        # Nesterov acceleration
        if Nesterov(options)
            t = (1+sqrt(1+4*t0^2))/2
            x .= x_+(t0-1)/t*(x_-x)
            t0 = t
            ~isnothing(counter) && (counter += 1)

            # Reset momentum
            ~isnothing(reset_counter(options)) && (counter >= reset_counter(options)) && (t0 = T(1))
        end

    end

    return x

end


## Specialized FISTA for prox/proj

struct WeightedProximableFISTA<:AbstractProximableOptions
    FISTA_options::ArgMinFISTA
end

FISTA_options(::Type{<:AbstractWeightedProximableFunction}, L::T; Nesterov::Bool=true, reset_counter::Union{Nothing,Integer}=nothing, niter::Union{Nothing,Integer}=nothing, verbose::Bool=false, fun_history::Bool=false) where {T<:Real} = ProxAndProjFISTA(FISTA_proxproj_options(L; Nesterov=Nesterov, reset_counter=reset_counter, niter=niter, verbose=verbose, fun_history=fun_history))