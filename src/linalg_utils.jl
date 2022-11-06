#: Utils

export conjugate


# Differentiable function linear algebra

struct ScaledDifferentiableFunction{T,N}<:AbstractDifferentiableFunction{T,N}
    scale::Real
    fun::AbstractDifferentiableFunction{T,N}
end

Base.:*(scale::T, fun::AbstractDifferentiableFunction{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = ScaledDifferentiableFunction{CT,N}(scale, fun)
Base.:*(fun::AbstractDifferentiableFunction{CT,N}, scale::T) where {T<:Real,N,CT<:RealOrComplex{T}} = scale*fun
Base.:/(fun::AbstractDifferentiableFunction{CT,N}, scale::T) where {T<:Real,N,CT<:RealOrComplex{T}} = (1/scale)*fun

fun_eval(fun::ScaledDifferentiableFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = fun.scale*fun.fun(x)
grad_eval!(fun::ScaledDifferentiableFunction{T,N}, x::AbstractArray{T,N}, g::AbstractArray{T,N}) where {T,N} = (grad_eval!(fun.fun, x, g); g .*= fun.scale; return g)
fungrad_eval!(fun::ScaledDifferentiableFunction{T,N}, x::AbstractArray{T,N}, g::AbstractArray{T,N}) where {T,N} = (fval = fungrad_eval!(fun.fun, x, g); g .*= fun.scale; return fun.scale*fval)

struct PlusDifferentiableFunction{T,N}<:AbstractDifferentiableFunction{T,N}
    fun1::AbstractDifferentiableFunction{T,N}
    fun2::AbstractDifferentiableFunction{T,N}
end

Base.:+(fun1::AbstractDifferentiableFunction{T,N}, fun2::AbstractDifferentiableFunction{T,N}) where {T,N} = PlusDifferentiableFunction{T,N}(fun1, fun2)

fun_eval(fun::PlusDifferentiableFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = fun.fun1(x)+fun.fun2(x)
grad_eval!(fun::PlusDifferentiableFunction{T,N}, x::AbstractArray{T,N}, g::AbstractArray{T,N}) where {T,N} = (grad_eval!(fun.fun1, x, g); g .+= grad_eval(fun.fun2, x); return g)
function fungrad_eval!(fun::PlusDifferentiableFunction{T,N}, x::AbstractArray{T,N}, g::AbstractArray{T,N}) where {T,N}
    fval = fungrad_eval!(fun.fun1, x, g)
    fval2, g2 = fungrad_eval(fun.fun2, x)
    g .+= g2
    return fval+fval2
end

struct MinusDifferentiableFunction{T,N}<:AbstractDifferentiableFunction{T,N}
    fun1::AbstractDifferentiableFunction{T,N}
    fun2::AbstractDifferentiableFunction{T,N}
end

Base.:-(fun1::AbstractDifferentiableFunction{T,N}, fun2::AbstractDifferentiableFunction{T,N}) where {T,N} = MinusDifferentiableFunction{T,N}(fun1, fun2)

fun_eval(fun::MinusDifferentiableFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = fun.fun1(x)-fun.fun2(x)
grad_eval!(fun::MinusDifferentiableFunction{T,N}, x::AbstractArray{T,N}, g::AbstractArray{T,N}) where {T,N} = (grad_eval!(fun.fun1, x, g); g .-= grad_eval(fun.fun2, x); return g)
function fungrad_eval!(fun::MinusDifferentiableFunction{T,N}, x::AbstractArray{T,N}, g::AbstractArray{T,N}) where {T,N}
    fval = fungrad_eval!(fun.fun1, x, g)
    fval2, g2 = fungrad_eval(fun.fun2, x)
    g .-= g2
    return fval-fval2
end


# Proximable linear algebra

struct ScaledProximableFunction{T,N}<:AbstractProximableFunction{T,N}
    scale::Real
    fun::AbstractProximableFunction{T,N}
end

Base.:*(scale::T, fun::AbstractProximableFunction{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = ScaledProximableFunction{CT,N}(scale, fun)
Base.:*(fun::AbstractProximableFunction{CT,N}, scale::T) where {T<:Real,N,CT<:RealOrComplex{T}} = scale*fun
Base.:/(fun::AbstractProximableFunction{CT,N}, scale::T) where {T<:Real,N,CT<:RealOrComplex{T}} = (1/scale)*fun

fun_eval(fun::ScaledProximableFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = fun.scale*fun.fun(x)

proxy!(y::AbstractArray{CT,N}, λ::T, g::ScaledProximableFunction{CT,N}, options::AbstractArgMinOptions, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = proxy!(y, λ*g.scale, g.fun, options, x)
project!(y::AbstractArray{CT,N}, ε::T, g::ScaledProximableFunction{CT,N}, options::AbstractArgMinOptions, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = project!(y, g.scale/ε, g.fun, options, x)


# Conjugation of proximable functions

struct ConjugateProximableFunction{T,N}<:AbstractProximableFunction{T,N}
    fun::AbstractProximableFunction{T,N}
end

conjugate(g::AbstractProximableFunction{T,N}) where {T,N} = ConjugateProximableFunction{T,N}(g)
conjugate(g::ConjugateProximableFunction) = g.fun

function proxy!(y::AbstractArray{CT,N}, λ::T, g::ConjugateProximableFunction{CT,N}, options::AbstractArgMinOptions, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}}
    proxy!(y/λ, 1/λ, g.fun, options, x)
    return x .= y-λ*x
end


# Mixed algebra

struct DiffPlusProxFunction{T,N}<:AbstractMinimizableFunction{T,N}
    diff::AbstractDifferentiableFunction{T,N}
    prox::AbstractProximableFunction{T,N}
end

Base.:+(f::AbstractDifferentiableFunction{T,N}, g::AbstractProximableFunction{T,N}) where {T,N} = DiffPlusProxFunction{T,N}(f, g)
Base.:+(g::AbstractProximableFunction{T,N}, f::AbstractDifferentiableFunction{T,N}) where {T,N} = f+g

fun_eval(fun::DiffPlusProxFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = fun.diff(x)+fun.prox(x)

function argmin!(fun::DiffPlusProxFunction{CT,N}, initial_estimate::AbstractArray{CT,N}, options::OptionsFISTA{T}, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}}
    # - FISTA: Beck, A., and Teboulle, M., 2009, A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems

    # Initialization
    x .= initial_estimate
    x_ = similar(x)
    g  = similar(x)
    L = Lipschitz_constant(options)
    isnothing(options.reset_counter) ? (counter = nothing) : (counter = 0)
    t0 = T(1)

    # Optimization loop
    @inbounds for i = 1:options.niter

        # Compute gradient
        if verbose(options) || ~isnothing(options.fun_history)
            fval_i = fungrad_eval!(fun.diff, x, g)
        else
            grad_eval!(fun.diff, x, g)
        end
        ~isnothing(options.fun_history) && (options.fun_history[i] = fval_i)

        # Print current iteration
        verbose(options) && (@info string("Iter: ", i, ", fval: ", fval_i))

        # Update
        proxy!(x-g/L, 1/L, fun.prox, x_)

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