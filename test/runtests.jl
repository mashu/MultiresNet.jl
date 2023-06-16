using MultiresNet
using Test
using Flux
using CUDA

@testset "MultiresNet.jl" begin
    channels = 3
    depth = 4
    seqlen = 4
    kernel = 4
    x = Array{Float64, 3}(undef, 1, 3, 4)
    x[1,1,:] = [1.0, 2.0, 3.0, 4.0]
    x[1,2,:] = [5.0, 6.0, 7.0, 8.0]
    x[1,3,:] = [9.0, 10.0, 11.0, 12.0]
    h = Array{Float64, 3}(undef, 3, 1, 4)
    h[1,1,:] = [1.0, 2.0, 3.0, 4.0]
    h[2,1,:] = [5.0, 6.0, 7.0, 8.0]
    h[3,1,:] = [9.0, 10.0, 11.0, 12.0]
    w = Array{Float64, 2}(undef, channels, depth+2)
    w[1,:] .= [1.0, 2.0, 3.0, 5.0, 6.0, 7.0]
    w[2,:] .= [1.0, 2.0, 3.0, 6.0, 7.0, 8.0]
    w[3,:] .= [3.0, 4.0, 5.0, 7.0, 8.0, 9.0]
    seq_block = MultiresNet.MultiresBlock(h, h, w)
    @test seq_block != nothing
    # Test if forward pass matches
    output = seq_block(MultiresNet.reverse_dims(x))[:,:,1]'
    @test output ≈ [1071.0 2940.0 6121.0 10153.0; 71360.0 148037.0 290440.0 439288.0; 1.394145e6 2.826942e6 5.562655e6 8.346751e6]
    # Test gradients
    @test sum(sum(Flux.gradient(xin->sum(seq_block(xin)),x))) != 0
    # Test helper functions
    @test MultiresNet.flatten_image(Flux.unsqueeze(x, dims=1)) |> size == (1,3,4)
    @test MultiresNet.reverse_dims(x) |>size == (4,3,1)
    @test MultiresNet.flip_dims(x) |>size == (3,1,4)
    # Test adjoint for zeros on CUDA
    @test sum(sum(Flux.gradient(x->sum(x), CUDA.zeros(10)))) == 10
end
