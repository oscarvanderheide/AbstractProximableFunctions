using LinearAlgebra, FastSolversForWeightedTV, CUDA, Flux, Test
CUDA.allowscalar(false)
include("./test_utils.jl")

# flag_gpu = true
flag_gpu = false

# Random data
T = Complex{Float64}
n = (256, 256)
y = randn(T, n..., 2); flag_gpu && (y = y |> gpu)
step = real(T)(1e-5)
rtol = real(T)(1e1*step)

# Norm (2-D)
for g = [mixed_norm(T,2,2,2), mixed_norm(T,2,2,1), mixed_norm(T,2,2,Inf)]

    # Proxy
    λ = real(T)(0.5)*norm(y)^2/g(y)
    local x = proxy(y, λ, g)

    # Gradient test
    fun = proxy_objfun(λ, g)
    test_grad(fun, y; step=step, rtol=rtol)

    # Projection test
    ε = real(T)(0.1)*g(y)
    local x = project(y, ε, g)
    @test g(x) ≈ ε rtol=rtol

    ## Gradient test
    fun = proj_objfun(ε, g)
    test_grad(fun, y; step=step, rtol=rtol)

end

# Random data
n = (256, 256, 256)
y = randn(T, n..., 3); flag_gpu && (y = y |> gpu)

# Norm (3-D)
for g = [mixed_norm(T,3,2,2), mixed_norm(T,3,2,1), mixed_norm(T,3,2,Inf)]

    # Proxy
    λ = real(T)(0.5)*norm(y)^2/g(y)
    local x = proxy(y, λ, g)

    # Gradient test
    fun = proxy_objfun(λ, g)
    test_grad(fun, y; step=step, rtol=rtol)

    # Projection test
    ε = real(T)(0.1)*g(y)
    local x = project(y, ε, g)
    @test g(x) ≈ ε rtol=rtol

    ## Gradient test
    fun = proj_objfun(ε, g)
    test_grad(fun, y; step=step, rtol=rtol)

end