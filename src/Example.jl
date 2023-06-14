using MLDatasets: CIFAR10
using Flux
using MultiresNet

# Hyperparameters
d_input = 3
d_output = 10
max_length = 1024
d_model = 256
depth = 4
kernel_size = 4

trainset = CIFAR10(:train)
trainloader = Flux.DataLoader(trainset, batchsize=128)

for (batch_ind, batch) in enumerate(trainloader)
    global input, target = batch
    break
end

img = MultiresNet.flatten_image(input) # 1024 x 3 x 128 (seq x channels x batch)

x = Float32.(img)
emb_layer = MultiresNet.EmbeddBlock(d_input, d_model)
seq_layer = MultiresNet.MultiresBlock(d_model, depth, kernel_size)
mix_layer = MultiresNet.MixingBlock(d_model)

x_emb = emb_layer(x)
block = Chain(
            SkipConnection(Chain(seq_layer, mix_layer),+),
            LayerNorm(max_length), 
            SkipConnection(Chain(seq_layer, mix_layer),+), 
            LayerNorm(max_length),
            GlobalMeanPool()
            )
z = block(x_emb)[1,:,:] # 1024 x 256 x 128 (seq x channels x batch) -> GlobalMeanPool
