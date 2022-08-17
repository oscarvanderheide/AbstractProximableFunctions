# Optimization utilities:
# - FISTA: Beck, A., and Teboulle, M., 2009, A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems

export OptimizerFISTA, FISTA_optimizer, reset!, minimize!, linesearch_backtracking, spectral_radius, leastsquares_solve!, leastsquares_solve, verbose, fun_history, set_proxy


## FISTA options

mutable struct OptimizerFISTA{T,PT}<:AbstractOptimizer
    Lipschitz_constant::T
    prox::PT
    Nesterov::Bool
    reset_counter::Union{Nothing,Integer}
    niter::Union{Nothing,Integer}
    verbose::Bool
    t::T
    counter::Union{Nothing,Integer}
    fun_history::Union{Nothing,AbstractVector{T}}
end

function FISTA_optimizer(L::T;
                         prox::PT=nothing,
                         Nesterov::Bool=true,
                         reset_counter::Union{Nothing,Integer}=nothing,
                         niter::Union{Nothing,Integer}=nothing,
                         verbose::Bool=false,
                         fun_history::Bool=false) where {T<:Real,N,CT<:RealOrComplex{T},PT<:Union{Nothing,ProximableFunction{CT,N}}}
    (fun_history && ~isnothing(niter)) ? (fval = Array{T,1}(undef,niter)) : (fval = nothing)
    t = T(1)
    counter = isnothing(reset_counter) ? nothing : 0
    return OptimizerFISTA{T,PT}(L, prox, Nesterov, reset_counter, niter, verbose, t, counter, fval)
end

function reset!(opt::OptimizerFISTA{T,PT}) where {T<:Real,N,CT<:RealOrComplex{T},PT<:ProximableFunction{CT,N}}
    opt.t = 1
    ~isnothing(opt.counter) && (opt.counter = 0)
    ~isnothing(opt.fun_history) && ~isnothing(opt.niter) && (opt.fun_history .= Array{T,1}(undef,opt.niter))
    return opt
end

set_proxy(opt::OptimizerFISTA{T,Nothing}, prox::ProximableFunction{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = OptimizerFISTA{T,typeof(prox)}(opt.Lipschitz_constant, prox, opt.Nesterov, opt.reset_counter, opt.niter, opt.verbose, opt.t, opt.counter, opt.fun_history)

verbose(opt::OptimizerFISTA) = opt.verbose

fun_history(opt::OptimizerFISTA) = opt.fun_history

function Flux.Optimise.apply!(opt::OptimizerFISTA{T,PT}, x::AbstractArray{CT,N}, g::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T},PT<:Union{Nothing,ProximableFunction{CT,N}}}

    # Gradient + proxy update
    steplength = T(1)/opt.Lipschitz_constant
    xnew = x-steplength*g
    isnothing(opt.prox) ? (g .= xnew) : proxy!(xnew, steplength, opt.prox, g)
    g .= x-g

    # Nesterov momentum
    if opt.Nesterov
        t = (1+sqrt(1+4*opt.t^2))/2
        g .*= (t+opt.t-1)/t
        opt.t = t
    end

    # Update counter
    ~isnothing(opt.counter) && (opt.counter += 1)
    ~isnothing(opt.reset_counter) && (opt.counter > opt.reset_counter) && reset!(opt)

    return g

end


# Generic FISTA solver

"""
Solver for the regularized problem via FISTA-like gradient-projections:
```math
min_x f(x)+g(x)
```
Reference: Beck, A., and Teboulle, M., 2009, A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems
https://www.ceremade.dauphine.fr/~carlier/FISTA
"""
function minimize!(fun::DifferentiableFunction{CT,N}, initial_estimate::AbstractArray{CT,N}, opt::OptimizerFISTA{T,PT}, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T},PT<:ProximableFunction{CT,N}}

    # Initialization
    x .= initial_estimate
    g  = similar(x)
    reset!(opt)

    # Optimization loop
    for i = 1:opt.niter
        fval_i = funeval!(fun, x; gradient=g, eval=opt.verbose || ~isnothing(opt.fun_history))
        ~isnothing(opt.fun_history) && (opt.fun_history[i] = fval_i)
        Flux.Optimise.update!(opt, x, g)
        opt.verbose && (@info string("iter: ", i, ", fval: ", fval_i))
    end

    return x

end

minimize!(fun::DiffPlusProxFun{CT,N}, initial_estimate::AbstractArray{CT,N}, opt::OptimizerFISTA{T,Nothing}, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = minimize!(fun.diff, initial_estimate, set_proxy(opt, fun.prox), x)


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


function linesearch_backtracking(obj::Function, x0::AbstractArray{CT,N}, p::AbstractArray{CT,N}, lr::T; fx0::Union{Nothing,T}=nothing, niter::Integer=3, mult_fact::T=T(0.5), verbose::Bool=false) where {T<:Real,N,CT<:RealOrComplex{T}}

    isnothing(fx0) && (fx0 = obj(x0))
    for _ = 1:niter
        fx = obj(x0+lr*p)
        fx < fx0 ? break : (verbose && print("."); lr *= mult_fact)
    end
    return lr

end


## Least-squares linear problem routines

leastsquares_solve!(A::AbstractLinearOperator{CT,N1,N2}, b::AbstractArray{CT,N2}, initial_estimate::AbstractArray{CT,N1}, opt::OptimizerFISTA{T,PT}, x::AbstractArray{CT,N1}) where {T<:Real,N1,N2,CT<:RealOrComplex{T},PT<:ProximableFunction{CT,N1}} = minimize!(leastsquares_misfit(A, b), initial_estimate, opt, x)

leastsquares_solve!(A::AbstractLinearOperator{CT,N1,N2}, b::AbstractArray{CT,N2}, initial_estimate::AbstractArray{CT,N1}, opt::OptimizerFISTA{T,Nothing}, x::AbstractArray{CT,N1}; prox::ProximableFunction{CT,N1}=null_prox(CT,N1)) where {T<:Real,N1,N2,CT<:RealOrComplex{T}} = leastsquares_solve!(A, b, initial_estimate, set_proxy(opt, prox), x)

leastsquares_solve(A::AbstractLinearOperator{CT,N1,N2}, b::AbstractArray{CT,N2}, initial_estimate::AbstractArray{CT,N1}, opt::OptimizerFISTA{T,PT}) where {T<:Real,N1,N2,CT<:RealOrComplex{T},PT<:ProximableFunction{CT,N1}} = leastsquares_solve!(A, b, initial_estimate, opt, similar(initial_estimate))

leastsquares_solve(A::AbstractLinearOperator{CT,N1,N2}, b::AbstractArray{CT,N2}, initial_estimate::AbstractArray{CT,N1}, opt::OptimizerFISTA{T,Nothing}; prox::ProximableFunction{CT,N1}=null_prox(CT,N1)) where {T<:Real,N1,N2,CT<:RealOrComplex{T},PT<:ProximableFunction{CT,N1}} = leastsquares_solve!(A, b, initial_estimate, set_proxy(opt, prox), similar(initial_estimate))