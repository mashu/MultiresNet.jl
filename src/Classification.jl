using MLDatasets: CIFAR10
using Flux
using MultiresNet
using OneHotArrays
using ProgressMeter
using CUDA
using JLD2
CUDA.math_mode!(CUDA.FAST_MATH)
using ParameterSchedulers: Stateful, next!
using ParameterSchedulers

"""
    correct(ŷ,y)

Function computes the number of correct predictions.
"""
function correct(ŷ,y)
    sum(onecold(ŷ, 0:9) .== onecold(y, 0:9))
end

# Define the means and standard deviations
means = [0.4914f0, 0.4822f0, 0.4465f0]
stds = [0.2023f0, 0.1994f0, 0.2010f0]

# Define the normalization function
normalize(x, μ=means, σ=stds) = (x .- μ) ./ σ

"""
    normalize_channels(x, μ=means, σ=stds)

Function to normalize channels of sCIFAR10 dataset as in the original paper.
"""
function normalize_channels(x, μ=means, σ=stds)
    x = convert(Array{Float32}, x)
    x = permutedims(x, [2, 1, 3])
    x = normalize(x, μ, σ)
    x = permutedims(x, [2, 1, 3])
    return x
end

"""
    load_cifar_data()

Function that loads the CIFAR10 dataset and returns
the training and test sets with first two dimenions flatten.
"""
function load_cifar_data()
    # load the CIFAR10 dataset
    trainset_x, trainset_y = CIFAR10(split=:train)[:]
    testset_x, testset_y = CIFAR10(split=:test)[:]

    # onehot
    trainset_y = onehotbatch(trainset_y, 0:9)
    testset_y = onehotbatch(testset_y, 0:9)

    # flatten the first two dimensions
    trainset_x = reshape(trainset_x, :, size(trainset_x, 3), size(trainset_x, 4))
    testset_x = reshape(testset_x, :, size(testset_x, 3), size(testset_x, 4))

    # normalize the channels
    trainset_x = CUDA.Mem.pin(normalize_channels(trainset_x))
    testset_x = CUDA.Mem.pin(normalize_channels(testset_x))

    return trainset_x, trainset_y, testset_x, testset_y
end

# Hyperparameters
d_input = 3
d_output = 10
max_length = 1024
d_model = 256
depth = 10
kernel_size = 2
batch_size = 50
drop = 0.25f0

trainset_x, trainset_y, testset_x, testset_y = load_cifar_data()
trainloader = Flux.DataLoader((trainset_x, trainset_y), batchsize=batch_size, shuffle=true, parallel=true, buffer=true)
testloader = Flux.DataLoader((testset_x, testset_y), batchsize=batch_size, shuffle=false, parallel=true, buffer=true)

model = Chain(
    # Block 1
    MultiresNet.EmbeddLayer(d_input, d_model),
    MultiresNet.MultiresBlock(d_model, depth, kernel_size, drop),
    MultiresNet.ChannelLayerNorm(d_model),
    # Block 2
    MultiresNet.MultiresBlock(d_model, depth, kernel_size, drop),
    MultiresNet.ChannelLayerNorm(d_model),
    # Block 3
    MultiresNet.MultiresBlock(d_model, depth, kernel_size, drop),
    MultiresNet.ChannelLayerNorm(d_model),
    # Block 4
    MultiresNet.MultiresBlock(d_model, depth, kernel_size, drop),
    MultiresNet.ChannelLayerNorm(d_model),
    # Block 5
    MultiresNet.MultiresBlock(d_model, depth, kernel_size, drop),
    MultiresNet.ChannelLayerNorm(d_model),
    # Block 6
    MultiresNet.MultiresBlock(d_model, depth, kernel_size, drop),
    MultiresNet.ChannelLayerNorm(d_model),
    # Block 7
    MultiresNet.MultiresBlock(d_model, depth, kernel_size, drop),
    MultiresNet.ChannelLayerNorm(d_model),
    # Block 8
    MultiresNet.MultiresBlock(d_model, depth, kernel_size, drop),
    MultiresNet.ChannelLayerNorm(d_model),
    # Block 9
    MultiresNet.MultiresBlock(d_model, depth, kernel_size, drop),
    MultiresNet.ChannelLayerNorm(d_model),
    # Block 10
    MultiresNet.MultiresBlock(d_model, depth, kernel_size, drop),
    MultiresNet.ChannelLayerNorm(d_model),    
    GlobalMeanPool(),
    Flux.flatten,
    Dense(d_model, d_output)) |> gpu

schedule = Stateful(CosAnneal(λ0 = 1f-4, λ1 = 1f-2, period = 50, restart=false))
optim = Flux.Optimiser(Flux.WeightDecay(1f-3), Flux.AdamW(0.0045f0))
ps = Flux.params(model)

# Training loop
n = 500
for epoch in 1:n
    optim[2][1].eta = next!(schedule)
    train_loss = 0f0
    # train_correct = 0f0
    test_loss = 0f0
    test_correct = 0f0
    test_total = Float32(length(testset_y))
    train_total = Float32(length(trainset_y))
    # Run trainset
    for (batch_ind, batch) in enumerate(trainloader)
        input, target = batch
        x = input  |> gpu
        y = target |> gpu
        let y_hat
            loss, grads = Flux.withgradient(ps) do
                y_hat = model(x)
                Flux.Losses.logitcrossentropy(y_hat, y)
            end
            Flux.update!(optim, ps, grads)
            # Update training statistics
            # train_correct += correct(y_hat, target)
            train_loss += loss
        end      
    end
    # Run testset
    for (test_batch_ind, test_batch) in enumerate(testloader)
        test_input, test_target = test_batch
        test_x = test_input  |> gpu
        test_y = test_target |> gpu
        test_y_hat = model(test_x)
        test_loss += Flux.Losses.logitcrossentropy(test_y_hat, test_y)
        test_correct += correct(test_y_hat, test_target)
    end
    # Compute statistics
    # println("Epoch $epoch, Train loss: $(train_loss*batch_size/train_total) Test loss: $(test_loss*batch_size/test_total) Train acc: $(train_correct/train_total) Test acc: $(test_correct/test_total)")
    println("Epoch $epoch, Train loss: $(train_loss*batch_size/train_total) Test loss: $(test_loss*batch_size/test_total) Test acc: $(test_correct/test_total)")
    # Checkpoint
    jldsave("model-epoch$(epoch).jld2", model_state = Flux.state(cpu(model)))
end
