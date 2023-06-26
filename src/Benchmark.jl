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
d_model = 256
depth = 10
kernel_size = 2
drop = 0.1
batch_size = 64

# Load data
trainset = CIFAR10(:train)
trainloader = Flux.DataLoader(trainset, batchsize=batch_size, shuffle=true, parallel=true)

emb = MultiresNet.EmbeddBlock(d_input, d_model)
model = MultiresNet.MultiresBlock(d_model, depth, kernel_size) |> gpu
ps = Flux.params(model)

batch = first(trainloader)
input, target = batch
x = emb(Float32.(MultiresNet.flatten_image(input))) |> gpu
y = onehotbatch(target, 0:9)    |> gpu

@info "1 layer"
z = model(x);
CUDA.@time model(x);
CUDA.@time model(x);
CUDA.@time model(x);

big_model = Chain([MultiresNet.MultiresBlock(d_model, depth, kernel_size) for _ in 1:10]...) |> gpu
z = big_model(x);
@info "Chain of 10 layers"
CUDA.@time big_model(x);
CUDA.@time big_model(x);
CUDA.@time big_model(x);

function train(x)
    let y_hat
        _, back = pullback(big_model, x)
    end
end

train(x);
@info "Pullback time 10 layers"
CUDA.@time train(x);
CUDA.@time train(x);
CUDA.@time train(x);
model(x);
@info "Forward pass 10 layers"
CUDA.@time big_model(x);
CUDA.@time big_model(x);
CUDA.@time big_model(x);
"Finished"
