using MLDatasets: CIFAR10
using Flux
using MultiresNet
using OneHotArrays
using ProgressMeter
using CUDA
CUDA.math_mode!(CUDA.PEDANTIC_MATH)

# Hyperparameters
d_input = 3
d_output = 10
max_length = 1024
d_model = 64
depth = 4
kernel_size = 4
batch_size = 64

trainset = CIFAR10(:train)
trainloader = Flux.DataLoader(trainset, batchsize=batch_size)

model = Chain(
    MultiresNet.EmbeddBlock(d_input, d_model),
    SkipConnection(
        Chain(MultiresNet.MultiresBlock(d_model, depth, kernel_size),
              MultiresNet.MixingBlock(d_model)),
    +),
    LayerNorm(max_length),
    SkipConnection(
        Chain(MultiresNet.MultiresBlock(d_model, depth, kernel_size),
              MultiresNet.MixingBlock(d_model)),
        +),
    LayerNorm(max_length),
    GlobalMeanPool(),
    Flux.flatten,
    Dense(d_model, d_output)) |> gpu

function accuracy(ŷ,y)
    correct = sum((onecold(ŷ, 0:9) |> cpu) .== y)
    total = length(y)
    "Accuracy: $correct / $total ($(correct/total))"
end

optim = Flux.Adam(1e-4)
ps = Flux.params(model)
# Training loop
losses = []
for epoch in 1:100
    for (batch_ind, batch) in enumerate(trainloader)
        input, target = batch
        x = Float32.(MultiresNet.flatten_image(input))   |> gpu # 1024 x 3 x 128 (seq x channels x batch)
        y = onehotbatch(target, 0:9)                     |> gpu # 10 x 128 (class x batch)
        loss, grads = Flux.withgradient(ps) do
            y_hat = model(x)
            Flux.Losses.logitcrossentropy(y_hat, y)
        end
        Flux.update!(optim, ps, grads)
        push!(losses, loss)
        if batch_ind % 100 == 0
            println("Loss: $(losses[end]) $(accuracy(model(x), target))")
        end
    end
end
