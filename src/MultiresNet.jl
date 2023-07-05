module MultiresNet
    using Flux: conv, Conv, glorot_uniform, gelu, @functor, unsqueeze, LayerNorm, sigmoid
    using Flux
    using CUDA
    using Zygote
    using ChainRulesCore
    CUDA.allowscalar(false)

    Zygote.@adjoint CUDA.zeros(x...) = CUDA.zeros(x...), _ -> map(_ -> nothing, x)

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
    function (m::MultiresLayer)(xin::AbstractArray{Float32}; activation=gelu) :: AbstractArray{Float32}
        kernel_size, depth = size(m.h0)[1], size(m.w)[3]-2
        res_lo = xin
        y = zero_array_like(xin)
    
        for i in depth:-1:1
            exponent = depth-i
            res_hi = conv_pad(res_lo, m.h1, exponent, kernel_size)
            res_lo = conv_pad(res_lo, m.h0, exponent, kernel_size)
            y = y .+ weight_multiplication(res_hi, m.w, i+1)
        end
    
        y = y .+ weight_multiplication(res_lo, m.w, 1)
        y = y .+ weight_multiplication(xin, m.w, depth+2)
        return activation.(y)
    end
    
    """
        conv_pad(res::AbstractArray{Float32}, h::AbstractArray{Float32}, exponent::Int, kernel_size::Int) :: AbstractArray{Float32}

    Function that performs convolution with padding
    """
    @inline function conv_pad(res::AbstractArray{Float32}, h::AbstractArray{Float32}, exponent::Int, kernel_size::Int) :: AbstractArray{Float32}
        padding = (2^exponent) * (kernel_size -1)
        return depthwiseconv(res, h, dilation=2^exponent, flipped=true, pad=(padding,0))
    end
    
    """
        weight_multiplication(res::AbstractArray{Float32}, weights::AbstractArray{Float32}, index::Int) :: AbstractArray{Float32}
    
    Function that performs multiplication of input with weights
    """
    @inline function weight_multiplication(res::AbstractArray{Float32}, weights::AbstractArray{Float32}, index::Int) :: AbstractArray{Float32}
        return res .* weights[:,:,index]
    end

    struct EmbeddLayer{T}
        conv::T
    end
    @functor EmbeddLayer
    function EmbeddLayer(channels_in::Int, channels_out::Int)
        conv = Conv((1,), channels_in => channels_out)
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
