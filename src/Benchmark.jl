using MLDatasets: CIFAR10
using Flux
using MultiresNet
using OneHotArrays
using CUDA
CUDA.math_mode!(CUDA.FAST_MATH)
CUDA.allowscalar(false)
using Zygote: pullback

# Hyperparameters
d_input = 3
d_output = 10
max_length = 1024
d_model = 64 # Orignal model 256
depth = 10
kernel_size = 2
drop = 0.25
batch_size = 64

function load_cifar_data()
    # load the CIFAR10 dataset
    trainset_x, trainset_y = CIFAR10(split=:train)[:]

    # onehot
    trainset_y = onehotbatch(trainset_y, 0:9)

    # flatten the first two dimensions
    trainset_x = CUDA.Mem.pin(reshape(trainset_x, :, size(trainset_x, 3), size(trainset_x, 4)))

    return trainset_x, trainset_y
end

# Load
trainset_x, trainset_y = load_cifar_data()
trainloader = Flux.DataLoader((trainset_x, trainset_y), batchsize=batch_size, shuffle=true, parallel=true)
batch = first(trainloader)
input, target = batch
x = input  |> gpu
y = target |> gpu

model1 = Chain(MultiresNet.EmbeddLayer(d_input, d_model)) |> gpu
model2 = Chain(MultiresNet.EmbeddLayer(d_input, d_model),
               MultiresNet.MultiresLayer(d_model, depth, kernel_size)) |> gpu
model3 = Chain(MultiresNet.EmbeddLayer(d_input, d_model),
               MultiresNet.MultiresLayer(d_model, depth, kernel_size),
               MultiresNet.MixingLayer(d_model)) |> gpu
model4 = Chain(MultiresNet.EmbeddLayer(d_input, d_model),
               MultiresNet.MultiresLayer(d_model, depth, kernel_size),
               MultiresNet.MixingLayer(d_model),
               MultiresNet.ChannelLayerNorm(d_model)) |> gpu
model5 = Chain(MultiresNet.EmbeddLayer(d_input, d_model),
               SkipConnection(Chain(
                    MultiresNet.MultiresLayer(d_model, depth, kernel_size),
                    MultiresNet.MixingLayer(d_model),
               ),+),
               MultiresNet.ChannelLayerNorm(d_model)) |> gpu

models = [model1, model2, model3, model4, model5]

for (i,model) in enumerate(models)
    @info "Model $i"
    z = model(x); # Precompile
    CUDA.@time model(x);
    CUDA.@time model(x);
    CUDA.@time model(x);
end
