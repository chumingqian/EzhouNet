import torch
import torch.nn as nn

# 1. Create random input data:
#    Let's simulate a batch of size 4, each sample has 32 features.
x = torch.randn(4, 32)

# 2. Define a linear layer that maps from 32 -> 1 neuron
linear_layer = nn.Linear(32, 1)

# 3. Define the sigmoid activation
sigmoid_activation = nn.Sigmoid()

# Forward pass:
# a) Pass through the linear layer
output_linear = linear_layer(x)

# b) Pass through the sigmoid activation
output_sigmoid = sigmoid_activation(output_linear)

print("Input shape:", x.shape)
print("Linear layer output shape:", output_linear.shape)
print("Output after sigmoid shape:", output_sigmoid.shape)
print("Output after sigmoid:\n", output_sigmoid)
