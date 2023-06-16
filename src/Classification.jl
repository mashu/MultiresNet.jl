using MLDatasets: CIFAR10
using Flux
using MultiresNet
using OneHotArrays
using ProgressMeter
using CUDA
using JLD2
CUDA.math_mode!(CUDA.FAST_MATH)

function correct(ŷ,y)
    sum((onecold(ŷ, 0:9) |> cpu) .== y)
end

# Hyperparameters
d_input = 3
d_output = 10
max_length = 1024
d_model = 256
depth = 10
kernel_size = 2
batch_size = 64

trainset = CIFAR10(:train)
testset = CIFAR10(:test)
trainloader = Flux.DataLoader(trainset, batchsize=batch_size, shuffle=true)
testloader = Flux.DataLoader(testset, batchsize=batch_size, shuffle=false)

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

optim = Flux.Optimiser(Flux.WeightDecay(1f-5), Flux.Adam(1f-3))
ps = Flux.params(model)

# Training loop for 40 epochs
n = 40
# p = Progress(n)
for epoch in 1:n
    train_loss = 0
    train_correct = 0
    test_loss = 0
    test_correct = 0
    test_total = length(testset)
    train_total = length(trainset)
    # ProgressMeter.next!(p; showvalues = [(:epoch, epoch)])
    for (batch_ind, batch) in enumerate(trainloader)
        input, target = batch
        x = Float32.(MultiresNet.flatten_image(input))   |> gpu # 1024 x 3 x 128 (seq x channels x batch)
        y = onehotbatch(target, 0:9)                     |> gpu # 10 x 128 (class x batch)
        let y_hat
            loss, grads = Flux.withgradient(ps) do
                y_hat = model(x)
                Flux.Losses.logitcrossentropy(y_hat, y)
            end
            Flux.update!(optim, ps, grads)
            # Update training statistics
            train_correct += correct(y_hat, target)
            train_loss += loss
        end      
    end
    # Run testset
    for (test_batch_ind, test_batch) in enumerate(testloader)
        test_input, test_target = test_batch
        test_x = Float32.(MultiresNet.flatten_image(test_input)) |> gpu
        test_y = onehotbatch(test_target, 0:9)                   |> gpu
        test_y_hat = model(test_x)
        test_loss += Flux.Losses.logitcrossentropy(test_y_hat, test_y)
        test_correct += correct(test_y_hat, test_target)
    end
    # Compute statistics
    println("Epoch $epoch, Train loss: $(train_loss/train_total) Test loss: $(test_loss/test_total) Train acc: $(train_correct/train_total) Test acc: $(test_correct/test_total)")
    # Checkpoint
    jldsave("model-checkpoint-epoch$(epoch).jld2", model_state = Flux.state(cpu(model)))
end
