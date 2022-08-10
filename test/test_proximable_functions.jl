using LinearAlgebra, ConvexOptimizationUtils, Flux, Test, AbstractLinearOperators, Random
Random.seed!(123)
include("./test_utils.jl")

# flag_gpu = true
flag_gpu = false

# Random data
T = Complex{Float64}
n = (32, 32)
y = randn(T, n..., 2); flag_gpu && (y = y |> gpu)
t = 1e-4
rtol = 10*t

# Norm (2-D)
for g = [mixed_norm(T,2,2,2), mixed_norm(T,2,2,1), mixed_norm(T,2,2,Inf)]

    # Proxy
    λ = 0.5*norm(y)^2/g(y)
    local x = proxy(y, λ, g)

    # Gradient test
    fun = proxy_objfun(λ, g)
    test_grad(fun, y; step=t, rtol=rtol)

    # Projection test
    ε = 0.1*g(y)
    local x = project(y, ε, g)
    @test g(x) ≈ ε rtol=rtol

    ## Gradient test
    fun = proj_objfun(ε, g)
    test_grad(fun, y; step=t, rtol=rtol)

end

# Random data
n = (32, 32, 32)
y = randn(T, n..., 3); flag_gpu && (y = y |> gpu)

# Norm (3-D)
for g = [mixed_norm(T,3,2,2), mixed_norm(T,3,2,1), mixed_norm(T,3,2,Inf)]

    # Proxy
    λ = 0.5*norm(y)^2/g(y)
    local x = proxy(y, λ, g)

    # Gradient test
    fun = proxy_objfun(λ, g)
    test_grad(fun, y; step=t, rtol=rtol)

    # Projection test
    ε = 0.1*g(y)
    local x = project(y, ε, g)
    @test g(x) ≈ ε rtol=rtol

    ## Gradient test
    fun = proj_objfun(ε, g)
    test_grad(fun, y; step=t, rtol=rtol)

end

# Weighted norms (3D)
n = (32, 32, 32)
y = randn(T, n...); flag_gpu && (y = y |> gpu)
v = randn(T, n..., 3)
A = linear_operator(T, n, (n...,3), x->v.*x, y->dropdims(sum(conj(v).*y; dims=4); dims=4))
ρ = 1.01*spectral_radius(A*A'; niter=200)
opt = FISTA_optimizer(ρ; Nesterov=true, niter=400, reset_counter=10, verbose=false)
for g = [mixed_norm(T,3,2,1)∘A, mixed_norm(T,3,2,2)∘A, mixed_norm(T,3,2,Inf)∘A]

    # Gradient test (proxy)
    λ = 0.5*norm(y)^2/g(y)
    fun = proxy_objfun(λ, g, opt)
    test_grad(fun, y; step=t, rtol=rtol)

    # Projection test
    ε = 0.1*g(y)
    x = project(y, ε, g, opt)
    @test g(x) ≈ ε rtol=rtol

    # Gradient test (projection)
    fun = proj_objfun(ε, g, opt)
    test_grad(fun, y; step=t, rtol=rtol)

end