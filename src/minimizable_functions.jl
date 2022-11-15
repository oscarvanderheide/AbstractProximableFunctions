#: Optimization utilities

export ExactArgmin, exact_argmin
export ArgminFISTA, FISTA_options, set_Lipschitz_constant, fun_history


## Fallback behavior for argmin

not_implemented() = error("This method has not been implemented for this function!")

argmin!(::AbstractMinimizableFunction{T,N}, ::AT, ::AbstractArgminOptions, ::AT) where {T,N,AT<:AbstractArray{T,N}} = not_implemented()
prox!(::AT, ::T, ::AbstractProximableFunction{CT,N}, ::AbstractArgminOptions, ::AT) where {T<:Real,N,CT<:RealOrComplex{T},AT<:AbstractArray{CT,N}} = not_implemented()
proj!(::AT, ::T, ::AbstractProximableFunction{CT,N}, ::AbstractArgminOptions, ::AT) where {T<:Real,N,CT<:RealOrComplex{T},AT<:AbstractArray{CT,N}} = not_implemented()
proj!(::AT, ::AbstractProjectionableSet{T,N}, ::AbstractArgminOptions, ::AT) where {T,N,AT<:AbstractArray{T,N}} = not_implemented()


## Exact options

struct ExactArgmin<:AbstractArgminOptions end

exact_argmin() = ExactArgmin()


# Differentiable+proximable functions

struct DiffPlusProxFunction{T,N}<:AbstractMinimizableFunction{T,N}
    diff::AbstractDifferentiableFunction{T,N}
    prox::AbstractProximableFunction{T,N}
    options::AbstractArgminOptions
end

Base.:+(f::AbstractDifferentiableFunction{T,N}, g::AbstractProximableFunction{T,N}; options::AbstractArgminOptions=exact_argmin()) where {T,N} = DiffPlusProxFunction{T,N}(f, g, options)
Base.:+(g::AbstractProximableFunction{T,N}, f::AbstractDifferentiableFunction{T,N}; options::AbstractArgminOptions=exact_argmin()) where {T,N} = +(f, g; options=options)


## FISTA options

struct ArgminFISTA<:AbstractArgminOptions
    Lipschitz_constant::Union{Nothing,Real}
    Nesterov::Bool
    reset_counter::Union{Nothing,Integer}
    niter::Union{Nothing,Integer}
    verbose::Bool
    fun_history::Union{Nothing,AbstractVector{<:Real}}
end

function FISTA_options(L::Union{Nothing,Real};
               Nesterov::Bool=true,
               reset_counter::Union{Nothing,Integer}=nothing,
               niter::Union{Nothing,Integer}=nothing,
               verbose::Bool=false,
               fun_history::Bool=false)
    (fun_history && ~isnothing(niter)) ? (fval = Array{typeof(L),1}(undef,niter)) : (fval = nothing)
    return ArgminFISTA(L, Nesterov, reset_counter, niter, verbose, fval)
end

Lipschitz_constant(options::ArgminFISTA) = options.Lipschitz_constant
set_Lipschitz_constant(options::ArgminFISTA, L::Real) = ArgminFISTA(L, options.Nesterov, options.reset_counter, options.niter, options.verbose, options.fun_history)
fun_history(options::ArgminFISTA) = options.fun_history

function argmin!(fun::DiffPlusProxFunction{CT,N}, initial_estimate::AT, options::ArgminFISTA, x::AT) where {T<:Real,N,CT<:RealOrComplex{T},AT<:AbstractArray{CT,N}}
    # - FISTA: Beck, A., and Teboulle, M., 2009, A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems

    # Initialization
    x .= initial_estimate
    x_ = similar(x)
    g  = similar(x)
    L = T(Lipschitz_constant(options))
    isnothing(options.reset_counter) ? (counter = nothing) : (counter = 0)
    t0 = T(1)
    diff_fun = fun.diff
    prox_fun = fun.prox

    # Optimization loop
    @inbounds for i = 1:options.niter

        # Compute gradient
        if options.verbose || ~isnothing(fun_history(options))
            fval_i = fungradeval!(diff_fun, x, g)
        else
            gradeval!(diff_fun, x, g)
        end
        ~isnothing(fun_history(options)) && (fun_history(options)[i] = fval_i)

        # Print current iteration
        options.verbose && (@info string("Iter: ", i, ", fval: ", fval_i))

        # Update
        prox!(x-g/L, 1/L, prox_fun, x_)

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