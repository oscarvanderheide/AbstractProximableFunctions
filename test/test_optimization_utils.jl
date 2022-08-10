using ConvexOptimizationUtils, LinearAlgebra, Flux, Test, AbstractLinearOperators
import Flux.Optimise: Optimiser, update!

# Setting linear system
T = Float64
Q = qr(randn(T, 100, 100)).Q
A = Q*diagm(T(1).+T(0.1)*randn(T,100))*Q'
b = randn(T, 100)
xtrue = A\b

# FISTA
x0 = randn(T, 100)
L = T(1.1)*spectral_radius(A'*A; niter=1000)
g = null_prox(T, 1)
Nesterov = true
# Nesterov = false
opt_fista = FISTA_optimizer(L; prox=g, Nesterov=Nesterov, reset_counter=10, verbose=false)
niter = 100
fval_fista = Array{T,1}(undef, niter)
x = deepcopy(x0)
for i = 1:niter
    r = A*x-b
    fval_fista[i] = 0.5*norm(r)^2
    local g = A'*r
    update!(opt_fista, x, g)
end
@test x ≈ xtrue rtol=1e-5

# Via minimize routine
Aop = linear_operator(T, size(xtrue), size(xtrue), x->A*x, y->A'*y)
f = leastsquares_misfit(Aop, b)
opt_fista = FISTA_optimizer(L; Nesterov=Nesterov, niter=niter, reset_counter=10, verbose=false)
x_ = minimize(f+g, x0, opt_fista)
@test x_ ≈ xtrue rtol=1e-5