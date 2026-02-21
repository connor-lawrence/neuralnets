import numpy as np
from neuralnets import dawn

inputs = np.array([[0,0,1,1],[0,1,0,1]])
targets = np.array([[0,1,1,0]])

d = dawn([2, 2, 1], "tanh", "tanh")
d.init_weights("xavier_normal")

d.teach(inputs, targets, 0.1, "backprop", 2000, 10, 1)

for j in range(inputs.shape[1]):
    inp = inputs[:, j].reshape(-1,1)
    out = d.think(inp)
    print(f"[neuralnets/dawn_test] Input: {inputs[:, j]}, Output: {out.flatten()[0]:.4f}")