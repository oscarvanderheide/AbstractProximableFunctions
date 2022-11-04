using LinearAlgebra, ConvexOptimizationUtils, Test, AbstractLinearOperators, Random
Random.seed!(123)

# Random data
T = Complex{Float64}
n = 32
t = 1e-4
rtol = 10*t

for dim = 1:3

    # Random data
    sz = tuple(repeat([n]; outer=dim)...)
    y = randn(T, sz...)
    v = randn(T, sz..., dim)
    A = linear_operator(T, sz, (sz...,dim), x->v.*x, y->dropdims(sum(conj(v).*y; dims=dim+1); dims=dim+1))
    ρ = 1.01*spectral_radius(A*A'; niter=200)
    opt = FISTA_optimizer(ρ; Nesterov=true, niter=400, reset_counter=20, verbose=false, fun_history=false)

    for g = [weighted_prox(mixed_norm(T,dim,2,2), A; optimizer=opt), weighted_prox(mixed_norm(T,dim,2,1), A; optimizer=opt), weighted_prox(mixed_norm(T,dim,2,Inf), A; optimizer=opt)]

        # Proxy
        λ = 0.5*norm(y)^2/g(y)
        x = proxy(y, λ, g)

        # Gradient test (proxy)
        fun = proxy_objfun(g, λ)
        @test test_grad(fun, y; step=t, rtol=rtol)

        # Projection test
        ε = 0.8*g(y)
        x = project(y, ε, g)
        @test (g(x) ≤ ε) || (abs(g(x)-ε) ≤ rtol*ε)

        ## Gradient test (projection)
        fun = proj_objfun(g, ε)
        @test test_grad(fun, y; step=t, rtol=rtol)

    end
end