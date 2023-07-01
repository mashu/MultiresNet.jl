module MultiresNet
    using Flux: conv, Conv, glorot_uniform, gelu, @functor, unsqueeze, LayerNorm, sigmoid
    using Flux
    using CUDA
    using Zygote
    using ChainRulesCore
    CUDA.allowscalar(false)

    Zygote.@adjoint CUDA.zeros(x...) = CUDA.zeros(x...), _ -> map(_ -> nothing, x)

    """
        glu_kernel(output::CuDeviceArray{Float32, 3, 1}, input::CuDeviceArray{Float32, 3, 1}, dim::Int)

    Kernel for GLU function for CUDA
    """
    function glu_kernel!(output::CuDeviceArray{Float32, 3, 1}, input::CuDeviceArray{Float32, 3, 1}, dim::Int)
        idx = threadIdx().x + blockDim().x * (blockIdx().x - 1)
        idy = threadIdx().y + blockDim().y * (blockIdx().y - 1)
        idz = threadIdx().z + blockDim().z * (blockIdx().z - 1)
        
        if (dim == 1 && idx ≤ size(input, 1) ÷ 2 && idy ≤ size(input, 2) && idz ≤ size(input, 3)) ||
        (dim == 2 && idx ≤ size(input, 1) && idy ≤ size(input, 2) ÷ 2 && idz ≤ size(input, 3)) ||
        (dim == 3 && idx ≤ size(input, 1) && idy ≤ size(input, 2) && idz ≤ size(input, 3) ÷ 2)
            if dim == 1
                @inbounds output[idx, idy, idz] = input[idx, idy, idz] * (1 / (1 + exp(-input[idx + size(input, 1) ÷ 2, idy, idz])))
            elseif dim == 2
                @inbounds output[idx, idy, idz] = input[idx, idy, idz] * (1 / (1 + exp(-input[idx, idy + size(input, 2) ÷ 2, idz])))
            else
                @inbounds output[idx, idy, idz] = input[idx, idy, idz] * (1 / (1 + exp(-input[idx, idy, idz + size(input, 3) ÷ 2])))
            end
        end
        return
    end

    """
        glu(input::CuArray{Float32,3}, dim::Integer)

    Custom GLU function for CUDA, temporarily unused because custom @rrule is required
    """
    function glu(input::CuArray{Float32,3}, dim::Integer)
        # Define the dimensions of the output array
        output_dims = ntuple(i -> i == dim ? size(input, i) ÷ 2 : size(input, i), 3)
        output = CUDA.zeros(Float32, output_dims)

        # Number of threads and blocks
        threads = (8, 8, 8)
        blocks = (ceil(Int, output_dims[1]/threads[1]), ceil(Int, output_dims[2]/threads[2]), ceil(Int, output_dims[3]/threads[3]))

        # Run the kernel
        @cuda threads=threads blocks=blocks pairwise_mul_kernel!(output, input, dim)

        return output
    end

    """
        reverse_dims(x)

    Reverse all dimensions in 3D array
    """
    function reverse_dims(x::AbstractArray{T,3}) where T
        permutedims(x, (3,2,1))
    end

    """
        flip_dims(x)

    Flip second with first dimension in 3D array
    """
    function flip_dims(x::AbstractArray{T,3}) where T
        permutedims(x, (2,1,3))
    end

    struct MultiresLayer{H<:AbstractArray{Float32,3}}
        h0::H
        h1::H
        w::H
    end
    @functor MultiresLayer

    struct MultiresBlock{T}
        conv::T
    end
    @functor MultiresBlock

    """
        MultiresBlock(d_model::Int, depth::Int, kernel_size::Int)
    
    Basic MultiresBlock building block for layer that performes multiresolution convolution
    """
    function MultiresBlock(d_model::Int, depth::Int, kernel_size::Int, drop::Float32)
        conv = SkipConnection(
                Chain(
                    MultiresLayer(d_model, depth, kernel_size),
                    Dropout(drop, dims=2),
                    MixingLayer(d_model),
                    Dropout(drop, dims=2)
                    ),
                +)
        MultiresBlock(conv)
    end

    """
        (m::MultiresBlock)(xin)
    
    Forward pass through MultiresBlock
    """
    function (m::MultiresBlock)(xin)
        m.conv(xin)
    end

    """
        init(dims::Integer...)

    Simpified initialization function for weights as in the original paper
    """
    function init(dims::Integer...)
        (rand(Float32, dims...) .- 1.0f0) .* sqrt(2.0f0 / (prod(dims) * 2.0f0))
    end
    
    """
        w_init(dims::Integer...; depth::Integer=1)

    Simpified initialization function for weights as in the original paper
    """
    function w_init(dims::Integer...; depth::Integer=1)
        (rand(Float32, dims...) .- 1.0f0) .* sqrt(2.0f0 / (2.0f0*depth + 2.0f0))
    end

    """
        MultiresLayer(channels, depth, kernel_size)

    Basic MultiresLayer building block for layer that performes multiresolution convolution without mixing
    """
    function MultiresLayer(channels::Int, depth::Int, kernel_size::Int; init=init, w_init=w_init)
        h0 = init(kernel_size, 1, channels)
        h1 = init(kernel_size, 1, channels)
        w = unsqueeze(w_init(channels, depth + 2, depth=depth), dims=1)
        MultiresLayer(h0, h1, w)
    end

    function zero_array_like(xin::AbstractArray{Float32})
        y = fill!(similar(xin), 0)
    end

    @non_differentiable zero_array_like(x)

    """
        (m::MultiresLayer)(xin; σ=gelu)

    Object-like function that takes input data through the MultiresLayer
    """
    function (m::MultiresLayer)(xin::AbstractArray{Float32}; activation=gelu)
        kernel_size, depth, groups = size(m.h0)[1], size(m.w)[3]-2, size(xin)[2]
        res_lo = xin
        y = zero_array_like(xin)
        for i in depth:-1:1
            exponent = depth-i
            padding = (2^exponent) * (kernel_size -1)
            res_hi = conv(res_lo, m.h1, dilation=2^exponent, groups=groups, flipped=true, pad=(padding,0))
            res_lo = conv(res_lo, m.h0, dilation=2^exponent, groups=groups, flipped=true, pad=(padding,0))
            y = y .+ (res_hi .* m.w[:,:,i+1])
        end
        y = y .+ (res_lo .* m.w[:,:,1])
        y = y .+ (xin .*m.w[:,:,end])
        activation.(y)
    end
    
    struct EmbeddLayer{T}
        conv::T
    end
    @functor EmbeddLayer
    function EmbeddLayer(channels_in::Int, channels_out::Int; σ=gelu)
        conv = Conv((1,), channels_in => channels_out, σ)
        EmbeddLayer(conv)
    end

    """
        (m::EmbeddLayer)(zin)

    Object-like function applying embedding with convolution
    """
    function (m::EmbeddLayer)(zin)
        m.conv(zin)
    end

    struct ChannelLayerNorm{T}
        norm::T
    end
    @functor ChannelLayerNorm

    """
        ChannelLayerNorm(channels_in::Int)

    Layer normalization along channels
    """
    function ChannelLayerNorm(channels_in::Int)
        norm = LayerNorm(channels_in)
        ChannelLayerNorm(norm)
    end

    """
        (m::ChannelLayerNorm)(zin)

    Object-like function applying layer normalization along channels
    """
    function (m::ChannelLayerNorm)(zin)
        flip_dims(m.norm(flip_dims(zin)))
    end

    struct MixingLayer{T}
        conv::T
    end
    @functor MixingLayer
    function MixingLayer(channels::Int)
        conv = Conv((1,), channels=>2*channels)
        MixingLayer(conv)
    end

    """
        (m::MixingLayer)(zin)

    Object-like function applying mixing layer to convolved input
    """
    function (m::MixingLayer)(zin)
        Flux.NNlib.glu(m.conv(zin),2)
    end

    """
        flatten_image(img)

    Function to flatten first two dimensions of 4D array into 3D array
    """
    function flatten_image(img::AbstractArray{T,4}) where T
        reshape(img, prod(size(img)[1:2]),size(img)[3:end]...)
    end
end
