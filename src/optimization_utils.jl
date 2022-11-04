# Optimization utilities:
# - FISTA: Beck, A., and Teboulle, M., 2009, A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems

export OptimizerFISTA, FISTA_optimizer, reset!, Lipschitz_constant, set_Lipschitz_constant, set_Lipschitz_constant!, fun_history, verbose
export minimize, minimize!
export leastsquares_solve, leastsquares_solve!
export spectral_radius


## FISTA options

mutable struct OptimizerFISTA{T<:Real}<:AbstractDiffPlusProxOptimizer
    Lipschitz_constant::T
    Nesterov::Bool
    reset_counter::Union{Nothing,Integer}
    niter::Union{Nothing,Integer}
    verbose::Bool
    t::T
    counter::Union{Nothing,Integer}
    fun_history::Union{Nothing,AbstractVector{T}}
end

function FISTA_optimizer(L::T;
                         Nesterov::Bool=true,
                         reset_counter::Union{Nothing,Integer}=nothing,
                         niter::Union{Nothing,Integer}=nothing,
                         verbose::Bool=false,
                         fun_history::Bool=false) where {T<:Real}
    (fun_history && ~isnothing(niter)) ? (fval = Array{T,1}(undef,niter)) : (fval = nothing)
    t = T(1)
    counter = isnothing(reset_counter) ? nothing : 0
    return OptimizerFISTA{T}(L, Nesterov, reset_counter, niter, verbose, t, counter, fval)
end


## Setter/Getter

function reset!(optimizer::OptimizerFISTA{T}) where {T<:Real}
    optimizer.t = 1
    ~isnothing(optimizer.counter) && (optimizer.counter = 0)
    ~isnothing(optimizer.fun_history) && ~isnothing(optimizer.niter) && (optimizer.fun_history .= Array{T,1}(undef,optimizer.niter))
    return optimizer
end

set_Lipschitz_constant!(optimizer::OptimizerFISTA{T}, L::T) where {T<:Real} = (optimizer.Lipschitz_constant = L; return optimizer)
set_Lipschitz_constant(optimizer::OptimizerFISTA{T}, L::T) where {T<:Real} = OptimizerFISTA{T}(L, optimizer.Nesterov, optimizer.reset_counter, optimizer.niter, optimizer.verbose, optimizer.t, optimizer.counter, optimizer.fun_history)

Lipschitz_constant(optimizer::OptimizerFISTA) = optimizer.Lipschitz_constant
verbose(optimizer::OptimizerFISTA) = optimizer.verbose

fun_history(optimizer::OptimizerFISTA) = optimizer.fun_history


# FISTA solver

function minimize!(fun::DiffPlusProxFunction{CT,N}, initial_estimate::AbstractArray{CT,N}, optimizer::OptimizerFISTA{T}, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}}

    # Initialization
    x .= initial_estimate
    x_ = similar(x)
    g  = similar(x)
    reset!(optimizer)
    L = Lipschitz_constant(optimizer)

    # Optimization loop
    @inbounds for i = 1:optimizer.niter

        # Compute gradient
        if verbose(optimizer) || ~isnothing(optimizer.fun_history)
            fval_i = fungrad_eval!(fun.diff, x, g)
        else
            grad_eval!(fun.diff, x, g)
        end
        ~isnothing(optimizer.fun_history) && (optimizer.fun_history[i] = fval_i)

        # Print current iteration
        verbose(optimizer) && (@info string("Iter: ", i, ", fval: ", fval_i))

        # Update
        proxy!(x-g/L, 1/L, fun.prox, x_)

        # Nesterov acceleration
        if optimizer.Nesterov
            t = (1+sqrt(1+4*optimizer.t^2))/2
            x .= x_+(optimizer.t-1)/t*(x_-x)
            optimizer.t = t
            ~isnothing(optimizer.counter) && (optimizer.counter += 1)

            # Reset momentum
            ~isnothing(optimizer.reset_counter) && (optimizer.counter >= optimizer.reset_counter) && (optimizer.t = T(1))
        end

    end

    return x

end

minimize(fun::DiffPlusProxFunction{CT,N}, initial_estimate::AbstractArray{CT,N}, optimizer::OptimizerFISTA{T}) where {T<:Real,N,CT<:RealOrComplex{T}} = minimize!(fun, initial_estimate, optimizer, similar(initial_estimate))


## Least-squares linear problem routines

leastsquares_solve!(A::AbstractLinearOperator{CT,N1,N2}, b::AbstractArray{CT,N2}, g::AbstractProximableFunction{CT,N1}, initial_estimate::AbstractArray{CT,N1}, optimizer::OptimizerFISTA{T}, x::AbstractArray{CT,N1}) where {T<:Real,N1,N2,CT<:RealOrComplex{T}} = minimize!(leastsquares_misfit(A, b)+g, initial_estimate, optimizer, x)

leastsquares_solve(A::AbstractLinearOperator{CT,N1,N2}, b::AbstractArray{CT,N2}, g::AbstractProximableFunction{CT,N1}, initial_estimate::AbstractArray{CT,N1}, optimizer::OptimizerFISTA{T}) where {T<:Real,N1,N2,CT<:RealOrComplex{T}} = leastsquares_solve!(A, b, g, initial_estimate, optimizer, similar(initial_estimate))


# Other utils

function spectral_radius(A::AT; x::Union{Nothing,AbstractArray{T,N}}=nothing, niter::Int64=10) where {T,N,AT<:Union{AbstractMatrix{T},AbstractLinearOperator{T,N,N}}}
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