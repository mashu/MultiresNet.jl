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
    sum(onecold(ŷ, 0:9) .== y)
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

    # flatten the first two dimensions
    trainset_x = reshape(trainset_x, :, size(trainset_x, 3), size(trainset_x, 4))
    testset_x = reshape(testset_x, :, size(testset_x, 3), size(testset_x, 4))

    return trainset_x, trainset_y, testset_x, testset_y
end

# Hyperparameters
d_input = 3
d_output = 10
max_length = 1024
d_model = 256
depth = 10
kernel_size = 2
batch_size = 64
drop = 0.1

trainset_x, trainset_y, testset_x, testset_y = load_cifar_data()
trainloader = Flux.DataLoader((trainset_x, trainset_y), batchsize=batch_size, shuffle=true, parallel=true, buffer=true)
testloader = Flux.DataLoader((testset_x, testset_y), batchsize=batch_size, shuffle=false, parallel=true, buffer=true)

model = Chain(
    MultiresNet.EmbeddBlock(d_input, d_model),
    SkipConnection(
        Chain(MultiresNet.MultiresBlock(d_model, depth, kernel_size),
              Dropout(drop, dims=2),
              MultiresNet.MixingBlock(d_model),
              Dropout(drop, dims=2)),
    +),
    MultiresNet.ChannelLayerNorm(d_model),
    SkipConnection(
        Chain(MultiresNet.MultiresBlock(d_model, depth, kernel_size),
              Dropout(drop, dims=2),
              MultiresNet.MixingBlock(d_model),
              Dropout(drop, dims=2)),
    +),
    MultiresNet.ChannelLayerNorm(d_model),
    SkipConnection(
        Chain(MultiresNet.MultiresBlock(d_model, depth, kernel_size),
              Dropout(drop, dims=2),
              MultiresNet.MixingBlock(d_model),
              Dropout(drop, dims=2)),
    +),
    MultiresNet.ChannelLayerNorm(d_model),
    SkipConnection(
        Chain(MultiresNet.MultiresBlock(d_model, depth, kernel_size),
              Dropout(drop, dims=2),
              MultiresNet.MixingBlock(d_model),
              Dropout(drop, dims=2)),
    +),
    MultiresNet.ChannelLayerNorm(d_model),
    SkipConnection(
        Chain(MultiresNet.MultiresBlock(d_model, depth, kernel_size),
              Dropout(drop, dims=2),
              MultiresNet.MixingBlock(d_model),
              Dropout(drop, dims=2)),
    +),
    MultiresNet.ChannelLayerNorm(d_model),
    SkipConnection(
        Chain(MultiresNet.MultiresBlock(d_model, depth, kernel_size),
              Dropout(drop, dims=2),
              MultiresNet.MixingBlock(d_model),
              Dropout(drop, dims=2)),
    +),
    MultiresNet.ChannelLayerNorm(d_model),
    SkipConnection(
        Chain(MultiresNet.MultiresBlock(d_model, depth, kernel_size),
              Dropout(drop, dims=2),
              MultiresNet.MixingBlock(d_model),
              Dropout(drop, dims=2)),
    +),
    MultiresNet.ChannelLayerNorm(d_model),
    SkipConnection(
        Chain(MultiresNet.MultiresBlock(d_model, depth, kernel_size),
              Dropout(drop, dims=2),
              MultiresNet.MixingBlock(d_model),
              Dropout(drop, dims=2)),
    +),
    MultiresNet.ChannelLayerNorm(d_model),
    SkipConnection(
        Chain(MultiresNet.MultiresBlock(d_model, depth, kernel_size),
              Dropout(drop, dims=2),
              MultiresNet.MixingBlock(d_model),
              Dropout(drop, dims=2)),
    +),
    MultiresNet.ChannelLayerNorm(d_model),
    SkipConnection(
        Chain(MultiresNet.MultiresBlock(d_model, depth, kernel_size),
              Dropout(drop, dims=2),
              MultiresNet.MixingBlock(d_model),
              Dropout(drop, dims=2)),
    +),
    MultiresNet.ChannelLayerNorm(d_model),
    GlobalMeanPool(),
    Flux.flatten,
    Dense(d_model, d_output)) |> gpu

schedule = Stateful(CosAnneal(λ0 = 1f-4, λ1 = 1f-2, period = 10))
optim = Flux.Optimiser(Flux.WeightDecay(1f-5), Flux.AdamW(1f-3))
ps = Flux.params(model)

# Training loop for 40 epochs
n = 400
p = Progress(n)
for epoch in 1:n
    optim[2][1].eta = next!(schedule)
    train_loss = 0
    train_correct = 0
    test_loss = 0
    test_correct = 0
    test_total = length(testset_y)
    train_total = length(trainset_y)
    ProgressMeter.next!(p; showvalues = [(:epoch, epoch), (:eta, optim[2][1].eta)])
    # Run trainset
    for (batch_ind, batch) in enumerate(CUDA.CuIterator(trainloader))
        input, target = batch
        x = Float32.(input)
        y = onehotbatch(target, 0:9)
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
    for (test_batch_ind, test_batch) in enumerate(CUDA.CuIterator(testloader))
        test_input, test_target = test_batch
        test_x = Float32.(test_input)
        test_y = onehotbatch(test_target, 0:9)
        test_y_hat = model(test_x)
        test_loss += Flux.Losses.logitcrossentropy(test_y_hat, test_y)
        test_correct += correct(test_y_hat, test_target)
    end
    # Compute statistics
    println("Epoch $epoch, Train loss: $(train_loss*batch_size/train_total) Test loss: $(test_loss*batch_size/test_total) Train acc: $(train_correct/train_total) Test acc: $(test_correct/test_total)")
    # Checkpoint
    jldsave("model-checkpoint-epoch$(epoch).jld2", model_state = Flux.state(cpu(model)))
end
