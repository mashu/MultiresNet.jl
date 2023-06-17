module MultiresNet
    using Flux: conv, Conv, glorot_uniform, gelu, @functor
    using Flux.NNlib: glu
    using CUDA
    using Zygote
    using Tullio

    Zygote.@adjoint CUDA.zeros(x...) = CUDA.zeros(x...), _ -> map(_ -> nothing, x)

    bmm_vec(A,B) = @tullio C[i,j,k] := A[i,j,k] * B[j]

    struct MultiresBlock
        h0::AbstractArray
        h1::AbstractArray
        w::AbstractArray
    end
    @functor MultiresBlock

    """
        MultiresBlock(channels, depth, kernel_size)

    Basic MultiresBlock building block for layer that performes multiresolution convolution without mixing
    """
    function MultiresBlock(channels::Int, depth::Int, kernel_size::Int)
        h0 = glorot_uniform(kernel_size, 1, channels)
        h1 = glorot_uniform(kernel_size, 1, channels)
        w = glorot_uniform(depth + 2, channels)
        MultiresBlock(h0, h1, w)
    end

    """
        (m::MultiresBlock)(xin; σ=gelu)

    Object-like function that takes input data through the MultiresBlock
    """
    function (m::MultiresBlock)(xin; σ=gelu)
        kernel_size=size(m.h0)[1]
        depth = size(m.w)[1]-2
        d_channels = size(xin)[2]
        res_lo = xin
        y = isa(xin, CuArray) ? CUDA.zeros(eltype(xin), size(xin)) : zeros(eltype(xin), size(xin))
        groups = size(xin)[2]
        for i in depth:-1:1
            exponent = depth-i
            padding = (2^exponent) * (kernel_size -1)
            res_hi = conv(res_lo, m.h1, dilation=2^exponent, groups=groups, flipped=true, pad=(padding,0))
            res_lo = conv(res_lo, m.h0, dilation=2^exponent, groups=groups, flipped=true, pad=(padding,0))
            y = (y .+ bmm_vec(res_hi, m.w[i+1,:]))
        end
        y = (y .+ bmm_vec(res_lo, m.w[1,:]))
        y = (y .+ bmm_vec(xin, m.w[end,:]))
        σ.(y)
    end

    struct EmbeddBlock
        conv
    end
    @functor EmbeddBlock
    function EmbeddBlock(channels_in::Int, channels_out::Int)
        conv = Conv((1,), channels_in => channels_out)
        EmbeddBlock(conv)
    end

    """
        (m::EmbeddBlock)(zin)

    Object-like function applying embedding with convolution
    """
    function (m::EmbeddBlock)(zin)
        m.conv(zin)
    end

    struct MixingBlock
        conv
    end
    @functor MixingBlock
    function MixingBlock(channels::Int)
        conv = Conv((1,), channels=>2*channels)
        MixingBlock(conv)
    end

    """
        (m::MixingBlock)(zin)

    Object-like function applying mixing layer to convolved input
    """
    function (m::MixingBlock)(zin)
        glu(m.conv(zin),2)
    end

    """
        flatten_image(img)

    Function to flatten first two dimensions of 4D array into 3D array
    """
    function flatten_image(img::Array{T,4}) where T
        reshape(img, prod(size(img)[1:2]),size(img)[3:end]...)
    end
end
