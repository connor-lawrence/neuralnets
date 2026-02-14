import numpy as np
import threading
import time
import sys

class dawn:

  # Init

  def __init__(self, net, hidden_act, output_act):

    np.random.seed(314159)
    self.net = net
    self.hidden_act = hidden_act
    self.output_act = output_act
    self.weights = []
    self.biases = []
    self.layer_inputs = []
    self.layer_outputs = []
    self.stop = False
    self.pause = False
    threading.Thread(target=self.watch, daemon=True).start()

  # Init Weights

  def init_weights(self, type):

    self.weights = []
    self.biases = []

    for i in range(len(self.net) - 1):

      in_n = self.net[i]
      out_n = self.net[i+1]

      if type == "xavier_normal":
        new_weights = np.random.randn(out_n, in_n) * np.sqrt(2 / (in_n + out_n))
      elif type == "xavier_uniform":
        limit = np.sqrt(6 / (in_n + out_n))
        new_weights = np.random.uniform(-limit, limit, (out_n, in_n))
      elif type == "he_normal":        
        new_weights = np.random.randn(out_n, in_n) * np.sqrt(2 / in_n)
      elif type == "he_uniform":
        limit = np.sqrt(6 / in_n)
        new_weights = np.random.uniform(-limit, limit, (out_n, in_n))
      else:
        limit = np.sqrt(6 / (in_n + out_n))
        new_weights = np.random.uniform(-limit, limit, (out_n, in_n))

      self.weights.append(new_weights)
      self.biases.append(np.zeros((out_n, 1)))

  # Forward Pass

  def think(self, input_data):

    self.layer_inputs = []
    self.layer_outputs = []
    current_layer = input_data
    
    for i in range(len(self.weights)):
      
      weight_set = self.weights[i]
      bias_set = self.biases[i]
      result = np.dot(weight_set, current_layer) + bias_set
      self.layer_inputs.append(result)

      if i == len(self.weights) - 1:
        current_layer = self.activate(result, "output")
      else:
        current_layer = self.activate(result, "hidden")

      self.layer_outputs.append(current_layer)

    return current_layer

  # Train Network (RIGHT NOW ONLY ONE AT A TIME! FIX!)

  def teach(self, input_data, target, learn_rate, method, epochs, checks, decay):
    
    if method == "backprop":

      print(f"[dawn.teach] LR: {learn_rate}, {method}, EP:{epochs}, DC:{decay}")
      current_rate = learn_rate

      for epoch in range(epochs):

        if self.stop:
          return

        while self.pause:
          time.sleep(0.1)

        self.think(input_data)
        deltas = [None] * len(self.weights)
        delta_output = (self.layer_outputs[-1] - target) * self.derivative(self.layer_inputs[-1], "output")
        deltas[-1] = delta_output

        for i in reversed(range(len(self.weights) - 1)):
          deltas[i] = np.dot(self.weights[i+1].T, deltas[i+1]) * self.derivative(self.layer_inputs[i], "hidden")

        for i in range(len(self.weights)):
          if i == 0:
            previous_output = np.array(input_data)
          else:
            previous_output = self.layer_outputs[i-1]

          self.weights[i] -= current_rate * np.dot(deltas[i], previous_output.T) / input_data.shape[1]
          self.biases[i] -= current_rate * np.mean(deltas[i], axis=1, keepdims=True)

        mse = np.mean((self.layer_outputs[-1] - target) ** 2)
        current_rate *= decay

        if epoch % (epochs // checks) == 0:
          print(f"[dawn.teach] {round(100*(epoch/epochs)):02}%, Ep: {epoch:0{len(str(epochs))}d}, LR: {current_rate:.3f}, MSE: {mse:.6f}")

    elif method == "reinforce":
      pass # ADD REINFORCEMENT HERE!

  # User Functions

  def watch(self):
      global stop_training, pause_training
      while True:
          key = input().lower()
          if key == 'q':
              self.stop = True
              print("[dawn.watch] Training aborted")
              sys.exit()
          elif key == 'p':
              self.pause = True
              print("[dawn.watch] Training paused")
          elif key == 'r':
              self.pause = False
              print("[dawn.watch] Training resumed")

  # Math Functions

  def activate(self, x, layer_type):
    if layer_type == "output":
      if self.output_act == "sigmoid":
        return self.sigmoid(x)
      elif self.output_act == "tanh":
        return self.tanh(x)
      elif self.output_act == "relu":
        return self.relu(x)
      else:
        return self.sigmoid(x)
    else:
      if self.hidden_act == "sigmoid":
        return self.sigmoid(x)
      elif self.hidden_act == "tanh":
        return self.tanh(x)
      elif self.hidden_act == "relu":
        return self.relu(x)
      else:
        return self.sigmoid(x)

  def derivative(self, x, layer_type):
    if layer_type == "output":
      if self.output_act == "sigmoid":
        return self.sigmoid_der(x)
      elif self.output_act == "tanh":
        return self.tanh_der(x)
      elif self.output_act == "relu":
        return self.relu_der(x)
      else:
        return self.sigmoid_der(x)
    else:
      if self.hidden_act == "sigmoid":
        return self.sigmoid_der(x)
      elif self.hidden_act == "tanh":
        return self.tanh_der(x)
      elif self.hidden_act == "relu":
        return self.relu_der(x)
      else:
        return self.sigmoid_der(x)

  def sigmoid(self, x):
    return 1 / (1 + np.exp(-1 * np.clip(x, -500, 500)))

  def sigmoid_der(self, x):
    return np.clip(x, 0, 1) * (1 - np.clip(x, 0, 1))

  def tanh(self, x):
    return np.tanh(x)

  def tanh_der(self, x):
    return 1 - np.tanh(x) ** 2

  def relu(self, x):
    return np.maximum(0, x)

  def relu_der(self, x):
    return (x > 0).astype(float)

#######################################################################################################################

class byte:

  def __init__(self, layer_sizes):
    print("")
    self.layer_sizes = layer_sizes
    self.weights = []
    np.random.seed(1)
    for i in range(len(layer_sizes) - 1):
      limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i+1]))
      w = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i+1]))
      self.weights.append(w)
    print("[nnet] Byte Setup Complete!")

  def sigmoid(self, x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

  def sigmoid_der(self, x):
    return x * (1 - x)

  def push(self, x):
    layer = x
    for w in self.weights:
      layer = self.sigmoid(np.dot(layer, w))
    return layer

  def train(self, x, y, epochs, learn_rate, checks):
    print("[nnet] Byte Training underway...")
    for epoch in range(epochs):
      layers = [x]
      for w in self.weights:
        layers.append(self.sigmoid(np.dot(layers[-1], w)))
      error = y - layers[-1]
      delta = error * self.sigmoid_der(layers[-1])
      deltas = [delta]
      for i in reversed(range(len(self.weights) - 1)):
        delta = deltas[-1].dot(self.weights[i+1].T) * self.sigmoid_der(layers[i+1])
        deltas.append(delta)
      deltas.reverse()
      for i in range(len(self.weights)):
        self.weights[i] += layers[i].T.dot(deltas[i]) * learn_rate
      if epoch % (epochs // checks) == 0:
        mse = np.mean((y - layers[-1])**2)
        print(f"[nnet] {round(100 * (epoch / epochs))}%, Epoch: {epoch} , MSE: {mse:.6f}")
    print("[nnet] Byte Training Complete!")
    print("")
  
