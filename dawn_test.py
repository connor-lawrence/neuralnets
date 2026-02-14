import numpy as np
from neuralnets import dawn

# XOR dataset
inputs = np.array([[0,0,1,1],
                   [0,1,0,1]])  # 2 features, 4 samples
targets = np.array([[0,1,1,0]])  # XOR outputs

# Initialize Dawn: 2 inputs, 2 hidden neurons, 1 output
d = dawn([2, 2, 1], "tanh", "tanh")
d.init_weights("xavier_normal")

# Training loop
d.teach(inputs, targets, 0.1, "backprop", 2000, 10)

# Test Dawn
for j in range(inputs.shape[1]):
    inp = inputs[:, j].reshape(-1,1)
    out = d.think(inp)
    print(f"Input: {inputs[:, j]}, Output: {out.flatten()[0]:.4f}")

