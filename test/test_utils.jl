function test_grad(fun::DifferentiableFunction{CT,N}, x::AbstractArray{CT,N}; step::T=T(1e-4), rtol::T=eps(T)) where {T<:Real,N,CT<:Union{T,Complex{T}}}

    dx = randn(T, size(x)); dx *= norm(x)/norm(dx)
    _, Δx = funeval(fun, x; gradient=true, eval=false)
    fp1 = funeval(fun, x+T(0.5)*step*dx)
    fm1 = funeval(fun, x-T(0.5)*step*dx)
    @test (fp1-fm1)/step ≈ real(dot(dx, Δx)) rtol=rtol

end