import math
from typing import List
import numpy as np

class SimpleNetwork:
    """A simple feedforward network where all units have sigmoid activation,
    including at final layer (output) of model, and there are no bias term
    parameters, only layer weights. Input, output, and weight matrices follow
    denominator layout format (same as UDL).
    """

    @classmethod
    def random(cls, *layer_units: int):
        """Creates a feedforward neural network with the given number of units
        for each layer (including input (first) and output (last) layers).
        """
        def uniform(n_in, n_out):
            epsilon = math.sqrt(6) / math.sqrt(n_in + n_out)
            return np.random.uniform(-epsilon, +epsilon, size=(n_out, n_in))

        pairs = zip(layer_units, layer_units[1:])
        return cls(*[uniform(i, o) for i, o in pairs])

    def __init__(self, *layer_weights: np.ndarray):
        """Creates a neural network from a list of weight matrices."""
        self.layer_weights = list(layer_weights)
        self.num_layers = len(layer_weights) + 1

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Applies the sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of the sigmoid function."""
        sig = self.sigmoid(x)
        return sig * (1 - sig)

    def predict(self, input_matrix: np.ndarray) -> np.ndarray:
        """Performs forward propagation over the neural network."""
        activation = input_matrix
        for weights in self.layer_weights:
            activation = self.sigmoid(np.dot(weights, activation))
        return activation

    def predict_zero_one(self, input_matrix: np.ndarray) -> np.ndarray:
        """Performs forward propagation and converts outputs to binary (0 or 1)."""
        predictions = self.predict(input_matrix)
        return np.where(predictions >= 0.5, 1, 0)

    def gradients(self,
                  input_matrix: np.ndarray,
                  target_output_matrix: np.ndarray) -> List[np.ndarray]:
        """Calculates the gradients for each of the weight matrices."""
        pre_activations = []
        activations = [input_matrix]

        # Forward pass: compute activations and pre-activations
        activation = input_matrix
        for weights in self.layer_weights:
            z = np.dot(weights, activation)
            pre_activations.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        # Backward pass: compute gradients
        gradients = [None] * len(self.layer_weights)

        # Compute output layer error (delta)
        delta = (activations[-1] - target_output_matrix) * self.sigmoid_derivative(pre_activations[-1])

        # Calculate gradient for the output layer
        gradients[-1] = 2 * np.dot(delta, activations[-2].T) / input_matrix.shape[1]  # Multiply by 2

        # Backpropagation through the hidden layers
        for layer in reversed(range(len(self.layer_weights) - 1)):
            delta = np.dot(self.layer_weights[layer + 1].T, delta) * self.sigmoid_derivative(pre_activations[layer])
            gradients[layer] = 2 * np.dot(delta, activations[layer].T) / input_matrix.shape[1]  # Multiply by 2

        return gradients

    def train(self,
              input_matrix: np.ndarray,
              target_output_matrix: np.ndarray,
              iterations: int = 10,
              learning_rate: float = 0.1) -> None:
        """Trains the neural network on an input matrix and an expected output matrix."""
        for _ in range(iterations):
            # Calculate gradients
            grads = self.gradients(input_matrix, target_output_matrix)

            # Update weights
            for i in range(len(self.layer_weights)):
                self.layer_weights[i] -= learning_rate * grads[i]
