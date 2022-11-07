# ConvexOptimizationUtils

Set of abstractions and utilities for computing proximal and projection operators of convex functionals

Example: mix and match convex differentiable objectives, convex non-differentiable terms, and hard constraints!
```
# Setup differentiable objective
A = linear_operator(Float32, (64, 64), (64, 64), x->2*x, y->2*y)
y = randn(Float32, 64, 64)
obj = leastsquares_misfit(A, y)

# Proximal function
g = norm(Float32, 2, 1) # l1-norm for 2-D images

# Hard constraints
C = zero_set(randn(Float32, 64, 64) .> 0f0)

# Setup FISTA solver
ρ = 1.01f0*spectral_radius(A*A'; niter=10)
opt = conjugateproject_FISTA(ρ; Nesterov=true, niter=100, reset_counter=20, verbose=false, fun_history=false)

# Iterative solution
x0 = zeros(Float32, size(y))
x = argmin(obj+(g+indicator(C)), x0, opt)
```

This package is highly indebted to [ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl) and related projects.