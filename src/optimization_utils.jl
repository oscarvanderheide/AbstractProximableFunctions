# Optimization utilities:
# - FISTA: Beck, A., and Teboulle, M., 2009, A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems

export OptimizerFISTA, FISTA_optimizer, reset!, minimize!, linesearch_backtracking, spectral_radius


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
end

FISTA_optimizer(L::T; prox::PT=nothing, Nesterov::Bool=true, reset_counter::Union{Nothing,Integer}=nothing, niter::Union{Nothing,Integer}=nothing, verbose::Bool=false) where {T<:Real,N,CT<:RealOrComplex{T},PT<:Union{Nothing,ProximableFunction{CT,N}}} = OptimizerFISTA{T,PT}(L, prox, Nesterov, reset_counter, niter, verbose, T(1), isnothing(reset_counter) ? nothing : 0)

function reset!(opt::OptimizerFISTA)
    opt.t = 1
    ~isnothing(opt.counter) && (opt.counter = 0)
end

set_proxy(opt::OptimizerFISTA{T,Nothing}, prox::ProximableFunction{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = OptimizerFISTA{T,typeof(prox)}(opt.Lipschitz_constant, prox, opt.Nesterov, opt.reset_counter, opt.niter, opt.verbose, opt.t, opt.counter)

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
function minimize!(fun::DifferentiableFunction{CT,N}, initial_estimate::AbstractArray{CT,N}, opt::OptimizerFISTA{T,PT}, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T},PT<:ProximableFunction{T,N}}

    # Initialization
    x .= initial_estimate
    g  = similar(x)
    reset!(opt)
    opt.verbose && (fval = zeros(T,opt.niter))

    # Optimization loop
    for i = 1:opt.niter
        opt.verbose ? (fval[i] = grad!(fun, x, g; eval=true)) : grad!(fun, x, g; eval=false)
        Flux.Optimise.update!(opt, x, g)
        opt.verbose && (@info string("iter: ", i, ", fval: ", fval[i]))
    end

    opt.verbose ? (return (x, fval)) : (return x)

end

minimize!(fun::DiffPlusProxFun{CT,N}, initial_estimate::AbstractArray{CT,N}, opt::OptimizerFISTA{T,Nothing}, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = minimize!(fun.f, initial_estimate, set_proxy(opt, fun.g), x)


# Other utils

function spectral_radius(A::AT, x::AbstractArray{T,N}; niter::Int64=10) where {T,N,AT<:Union{AbstractMatrix{T},AbstractLinearOperator{T,N,N}}}
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