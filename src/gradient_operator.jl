#: Gradient operator via convolution

export gradient_operator, gradient_operator_batch


# Generic convolutional operator

struct ConvolutionalOperator{T,N}<:AbstractLinearOperator{T,N,N}
    cdims::DenseConvDims
    stencil::AbstractArray{T,N}
end

AbstractLinearOperators.domain_size(A::ConvolutionalOperator) = NNlib.input_size(A.cdims)
AbstractLinearOperators.range_size(A::ConvolutionalOperator) = NNlib.output_size(A.cdims)
AbstractLinearOperators.matvecprod(A::ConvolutionalOperator{T,N}, u::AbstractArray{T,N}) where {T,N} = conv(u, A.stencil)
AbstractLinearOperators.matvecprod_adj(A::ConvolutionalOperator{T,N}, v::AbstractArray{T,N}) where {T,N} = ∇conv_data(v, A.stencil, A.cdims)

Flux.gpu(A::ConvolutionalOperator{T,N}) where {T,N} = ConvolutionalOperator{T,N}(A.cdims, gpu(A.stencil))
Flux.cpu(A::ConvolutionalOperator{T,N}) where {T,N} = ConvolutionalOperator{T,N}(A.cdims, cpu(A.stencil))


# Gradient operator

function gradient_operator(n::NTuple{dim,Int64}, h::NTuple{dim,T}; complex::Bool=true) where {dim,T<:Real}

    # Gradient operator (w/out shaping)
    stencil = gradient_stencil(h; complex=complex)
    cdims = DenseConvDims((n..., 1, 1), size(stencil))
    complex ? (CT = Complex{T}) : (CT = T)
    ∇_ = ConvolutionalOperator{CT,dim+2}(cdims, stencil)

    # Reshaping operator
    Rin  = reshape_operator(CT, n, (n...,1,1))
    Rout = reshape_operator(CT, ((n.-1)...,dim,1), ((n.-1)...,dim))

    return Rout*∇_*Rin

end

function gradient_operator_batch(n::NTuple{dim,Int64}, nc::Int64, nb::Int64, h::NTuple{dim,T}; complex::Bool=true) where {dim,T<:Real}

    # Gradient operator (w/out shaping)
    stencil = gradient_stencil(h; complex=complex)
    cdims = DenseConvDims((n..., 1, nc*nb), size(stencil))
    complex ? (CT = Complex{T}) : (CT = T)
    ∇_ = ConvolutionalOperator{CT,dim+2}(cdims, stencil)

    # Reshaping operator
    Rin  = reshape_operator(CT, (n...,nc,nb), (n...,1,nc*nb))
    Rout = reshape_operator(CT, ((n.-1)...,dim,nc*nb), ((n.-1)...,dim*nc,nb))

    return Rout*∇_*Rin

end


# Gradient stencil utils

function gradient_stencil(h::NTuple{D,T}; complex::Bool=false) where {D,T<:Real}
    k = tuple(2*ones(Int64, D)...)
    stencil = zeros(T,k...,1,D)
    for i = 1:D
        idx = Vector{UnitRange{Int64}}(undef,D); fill!(idx, 2:2)
        idx[i] = 1:2
        view(stencil, tuple(idx...)...,1,i)[1:2] .= [T(1)/h[i]; T(-1)/h[i]]
    end
    complex ? (return Base.complex(stencil)) : (return stencil)
end