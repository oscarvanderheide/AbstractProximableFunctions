#: Optimization utilities

export ExactArgmin, exact_argmin
export ArgminFISTA, FISTA, Lipschitz_constant, set_Lipschitz_constant, fun_history, verbose
export ConjugateAndFISTA, conjugate_FISTA, ConjugateProjectAndFISTA, conjugateproject_FISTA

not_implemented() = error("This minimization method has not been implemented for this function!")

argmin!(::AbstractMinimizableFunction{T,N}, ::AbstractArray{T,N}, ::AbstractMinOptions, ::AbstractArray{T,N}) where {T,N} = not_implemented()
proxy!(::AbstractArray{CT,N}, ::T, ::AbstractProximableFunction{CT,N}, ::AbstractMinOptions, ::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = not_implemented()
project!(::AbstractArray{CT,N}, ::T, ::AbstractProximableFunction{CT,N}, ::AbstractMinOptions, ::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = not_implemented()
project!(::AbstractArray{T,N}, ::AbstractProjectionableSet{T,N}, ::AbstractMinOptions, ::AbstractArray{T,N}) where {T,N} = not_implemented()


## Exact min/proxy options

struct ExactMin{MT<:AbstractMinimizableFunction}<:AbstractMinOptions{MT} end
struct ExactProxy{PT<:AbstractProximableFunction}<:AbstractProxyOptions{PT} end

exact_min(MT::DataType) = ExactMin{MT}
exact_proxy(PT::DataType) = ExactMin{PT}


## FISTA options

mutable struct ArgminFISTA{T<:Real}<:AbstractMinOptions
    Lipschitz_constant::T
    Nesterov::Bool
    reset_counter::Union{Nothing,Integer}
    niter::Union{Nothing,Integer}
    verbose::Bool
    fun_history::Union{Nothing,AbstractVector{T}}
end

function FISTA(L::T;
               Nesterov::Bool=true,
               reset_counter::Union{Nothing,Integer}=nothing,
               niter::Union{Nothing,Integer}=nothing,
               verbose::Bool=false,
               fun_history::Bool=false) where {T<:Real}
    (fun_history && ~isnothing(niter)) ? (fval = Array{T,1}(undef,niter)) : (fval = nothing)
    options = ArgminFISTA{T}(L, Nesterov, reset_counter, niter, verbose, fval)
    return options
end

set_Lipschitz_constant(options::ArgminFISTA{T}, L::T) where {T<:Real} = ArgminFISTA{T}(L, options.Nesterov, options.reset_counter, options.niter, options.verbose, options.fun_history)
Lipschitz_constant(options::ArgminFISTA) = options.Lipschitz_constant
verbose(options::ArgminFISTA) = options.verbose
fun_history(options::ArgminFISTA) = options.fun_history

function argmin!(fun::AbstractDiffPlusProxFunction{CT,N}, initial_estimate::AbstractArray{CT,N}, options::ArgminFISTA{T}, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}}
    # - FISTA: Beck, A., and Teboulle, M., 2009, A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems

    # Initialization
    x .= initial_estimate
    x_ = similar(x)
    g  = similar(x)
    L = Lipschitz_constant(options)
    isnothing(options.reset_counter) ? (counter = nothing) : (counter = 0)
    t0 = T(1)
    diff_fun = get_diff(fun)
    prox_fun = get_prox(fun)

    # Optimization loop
    @inbounds for i = 1:options.niter

        # Compute gradient
        if verbose(options) || ~isnothing(options.fun_history)
            fval_i = fungrad_eval!(diff_fun, x, g)
        else
            grad_eval!(diff_fun, x, g)
        end
        ~isnothing(options.fun_history) && (options.fun_history[i] = fval_i)

        # Print current iteration
        verbose(options) && (@info string("Iter: ", i, ", fval: ", fval_i))

        # Update
        proxy!(x-g/L, 1/L, prox_fun, x_)

        # Nesterov acceleration
        if options.Nesterov
            t = (1+sqrt(1+4*t0^2))/2
            x .= x_+(t0-1)/t*(x_-x)
            t0 = t
            ~isnothing(counter) && (counter += 1)

            # Reset momentum
            ~isnothing(options.reset_counter) && (counter >= options.reset_counter) && (t0 = T(1))
        end

    end

    return x

end


## Conjugate-and-FISTA method

struct ConjugateAndFISTA{T<:Real}<:AbstractMinOptions
    options_FISTA::ArgminFISTA{T}
end

conjugate_FISTA(args...; kwargs...) = ConjugateAndFISTA(FISTA(args...; kwargs...))

get_FISTA_options(method::ConjugateAndFISTA) = method.options_FISTA


## Conjugate-project-and-FISTA method

struct ConjugateProjectAndFISTA{T<:Real}<:AbstractMinOptions
    options_FISTA::ArgminFISTA{T}
end

conjugateproject_FISTA(args...; kwargs...) = ConjugateProjectAndFISTA(FISTA(args...; kwargs...))

get_FISTA_options(method::ConjugateProjectAndFISTA) = method.options_FISTA