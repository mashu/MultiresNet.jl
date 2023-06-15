using MLDatasets: CIFAR10
using Flux
using MultiresNet
using OneHotArrays
using ProgressMeter
using CUDA

# Hyperparameters
d_input = 3
d_output = 10
max_length = 1024
d_model = 256
depth = 4
kernel_size = 4

trainset = CIFAR10(:train)
trainloader = Flux.DataLoader(trainset, batchsize=128)

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
    x->x[1,:,:],
    Dense(d_model, d_output)) |> gpu

optim = Flux.setup(Flux.Adam(0.01), model)

# Training loop
losses = []
@showprogress for epoch in 1:10
    for (batch_ind, batch) in enumerate(trainloader)
        input, target = batch
        x = MultiresNet.flatten_image(input) |> gpu # 1024 x 3 x 128 (seq x channels x batch)
        y = onehotbatch(target, 0:9)         |> gpu # 10 x 128 (class x batch)
        loss, grads = Flux.withgradient(model) do m
            y_hat = model(x)
            Flux.Losses.logitcrossentropy(y_hat, y)
        end
        Flux.update!(optim, model, grads[1])
        push!(losses, loss)
    end
end
