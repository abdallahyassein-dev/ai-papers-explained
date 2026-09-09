"""
lenet5.py — The Complete LeNet-5 Architecture

Assembles all layers into the exact architecture described in the paper:

    Input (1, 32, 32)
      ↓
    C1: Conv(1→6, 5×5)         → (6, 28, 28)     156 params
      ↓
    S2: SubSample(6, 2×2)      → (6, 14, 14)      12 params
      ↓
    C3: Conv(6→16, 5×5, partial)→ (16, 10, 10)  1,516 params
      ↓
    S4: SubSample(16, 2×2)     → (16, 5, 5)        32 params
      ↓
    C5: Conv(16→120, 5×5)      → (120, 1, 1)   48,120 params
      ↓  (flatten to 120)
    F6: FC(120→84)             → (84,)          10,164 params
      ↓
    Output: RBF(84→10)         → (10,)               0 params
                                              ────────────────
                                              Total: 60,000 params

The C3 partial connection table is a key design feature:
- Not all S2 feature maps connect to all C3 feature maps
- This FORCES the network to learn diverse features
- It also reduces the parameter count
"""

import numpy as np
from layers import ConvLayer, SubSamplingLayer, FullyConnectedLayer, RBFOutputLayer


# ═══════════════════════════════════════════════════════════════════════
# C3 PARTIAL CONNECTION TABLE (from the paper, Table 1)
# ═══════════════════════════════════════════════════════════════════════
#
# This table defines which S2 feature maps connect to which C3 feature maps.
# Read it as: C3_CONNECTIONS[j] = list of S2 maps that feed into C3 map j.
#
#                     C3 Feature Maps
#               0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
# S2 map 0:     ✓  .  .  .  ✓  ✓  ✓  .  .  ✓  ✓  ✓  ✓  .  ✓  ✓
# S2 map 1:     ✓  ✓  .  .  .  ✓  ✓  ✓  .  .  ✓  ✓  ✓  ✓  .  ✓
# S2 map 2:     ✓  ✓  ✓  .  .  .  ✓  ✓  ✓  .  .  ✓  .  ✓  ✓  ✓
# S2 map 3:     .  ✓  ✓  ✓  .  .  ✓  ✓  ✓  ✓  .  .  ✓  .  ✓  ✓
# S2 map 4:     .  .  ✓  ✓  ✓  .  .  ✓  ✓  ✓  ✓  .  ✓  ✓  .  ✓
# S2 map 5:     .  .  .  ✓  ✓  ✓  .  .  ✓  ✓  ✓  ✓  .  ✓  ✓  ✓
#
# Feature maps 0-5:   connected to 3 input maps each
# Feature maps 6-11:  connected to 4 input maps each
# Feature maps 12-14: connected to 4 input maps each
# Feature map 15:     connected to all 6 input maps

C3_CONNECTIONS = [
    [0, 1, 2],          # C3 map 0  ← 3 inputs
    [1, 2, 3],          # C3 map 1  ← 3 inputs
    [2, 3, 4],          # C3 map 2  ← 3 inputs
    [3, 4, 5],          # C3 map 3  ← 3 inputs
    [0, 4, 5],          # C3 map 4  ← 3 inputs
    [0, 1, 5],          # C3 map 5  ← 3 inputs
    [0, 1, 2, 3],       # C3 map 6  ← 4 inputs
    [1, 2, 3, 4],       # C3 map 7  ← 4 inputs
    [2, 3, 4, 5],       # C3 map 8  ← 4 inputs
    [0, 3, 4, 5],       # C3 map 9  ← 4 inputs
    [0, 1, 4, 5],       # C3 map 10 ← 4 inputs
    [0, 1, 2, 5],       # C3 map 11 ← 4 inputs
    [0, 2, 4, 5],       # C3 map 12 ← 4 inputs (non-contiguous)
    [0, 1, 3, 5],       # C3 map 13 ← 4 inputs (non-contiguous)
    [0, 2, 3, 4],       # C3 map 14 ← 4 inputs (non-contiguous)
    [0, 1, 2, 3, 4, 5], # C3 map 15 ← ALL 6 inputs
]


class LeNet5:
    """
    The complete LeNet-5 Convolutional Neural Network.

    Usage:
        model = LeNet5()
        output = model.forward(image)     # image shape: (1, 32, 32)
        loss = model.backward(label)      # label: int 0-9
        model.update_weights(lr=0.0005)   # SGD update
    """

    def __init__(self):
        """Build all layers of LeNet-5."""
        print("Building LeNet-5 architecture...")

        # Layer C1: Convolution 1→6 feature maps, 5×5 kernel, input 32×32
        self.c1 = ConvLayer(
            in_channels=1, out_channels=6,
            kernel_size=5, input_size=32,
            connection_table=None  # fully connected (1 input map)
        )

        # Layer S2: Sub-sampling 6 maps, 28→14
        self.s2 = SubSamplingLayer(num_channels=6, input_size=28)

        # Layer C3: Convolution 6→16 feature maps, 5×5, PARTIAL connections
        self.c3 = ConvLayer(
            in_channels=6, out_channels=16,
            kernel_size=5, input_size=14,
            connection_table=C3_CONNECTIONS
        )

        # Layer S4: Sub-sampling 16 maps, 10→5
        self.s4 = SubSamplingLayer(num_channels=16, input_size=10)

        # Layer C5: Convolution 16→120, 5×5 (fully connected because input=5×5)
        self.c5 = ConvLayer(
            in_channels=16, out_channels=120,
            kernel_size=5, input_size=5,
            connection_table=None  # fully connected
        )

        # Layer F6: Fully connected 120→84
        self.f6 = FullyConnectedLayer(in_size=120, out_size=84)

        # Output: RBF layer 84→10
        self.output = RBFOutputLayer(in_size=84, num_classes=10)

        # Store all layers in order for easy iteration
        self.layers = [self.c1, self.s2, self.c3, self.s4, self.c5, self.f6, self.output]

        # Print architecture summary
        self._print_summary()

    def _print_summary(self):
        """Print a summary of the architecture matching the paper's table."""
        total_params = 0
        layer_info = [
            ("C1  Conv", "6×28×28", self.c1.param_count()),
            ("S2  Pool", "6×14×14", self.s2.param_count()),
            ("C3  Conv", "16×10×10", self.c3.param_count()),
            ("S4  Pool", "16×5×5", self.s4.param_count()),
            ("C5  Conv", "120×1×1", self.c5.param_count()),
            ("F6  FC", "84", self.f6.param_count()),
            ("Out RBF", "10", self.output.param_count()),
        ]

        print(f"{'Layer':<12} {'Output Shape':<14} {'Parameters':>10}")
        print("─" * 40)
        for name, shape, params in layer_info:
            total_params += params
            print(f"{name:<12} {shape:<14} {params:>10,}")
        print("─" * 40)
        print(f"{'TOTAL':<12} {'':14} {total_params:>10,}")
        print()

    def forward(self, x):
        """
        Forward pass through the entire network.

        Parameters
        ----------
        x : np.ndarray of shape (1, 32, 32)
            Preprocessed input image (1 channel, 32×32 pixels)

        Returns
        -------
        np.ndarray of shape (10,)
            RBF distances to each class (smaller = better match)

        Data flow:
        ──────────
        (1,32,32) →C1→ (6,28,28) →S2→ (6,14,14) →C3→ (16,10,10)
        →S4→ (16,5,5) →C5→ (120,1,1) →flatten→ (120,)
        →F6→ (84,) →RBF→ (10,)
        """
        # C1: Convolution
        out = self.c1.forward(x)                # (6, 28, 28)

        # S2: Sub-sampling
        out = self.s2.forward(out)              # (6, 14, 14)

        # C3: Convolution with partial connections
        out = self.c3.forward(out)              # (16, 10, 10)

        # S4: Sub-sampling
        out = self.s4.forward(out)              # (16, 5, 5)

        # C5: Convolution (effectively fully connected)
        out = self.c5.forward(out)              # (120, 1, 1)

        # Flatten for fully connected layers
        out = out.flatten()                     # (120,)

        # F6: Fully connected
        out = self.f6.forward(out)              # (84,)

        # Output: RBF distances
        out = self.output.forward(out)          # (10,)

        return out

    def backward(self, label):
        """
        Backward pass: Compute all gradients via backpropagation.

        This is where the magic from Rumelhart et al. (1986) happens!
        Gradients flow backward through every layer using the chain rule.

        Parameters
        ----------
        label : int
            The correct class (0-9)

        Returns
        -------
        loss : float
            The loss value for this example

        Gradient flow:
        ─────────────
        Loss ←─ RBF ←─ F6 ←─ C5 ←─ S4 ←─ C3 ←─ S2 ←─ C1

        At each layer:
          grad_input = layer.backward(grad_output)
          This also computes and stores weight gradients internally.
        """
        # RBF output backward → gives gradient w.r.t. F6 output + loss
        grad, loss = self.output.backward(label)

        # F6 backward
        grad = self.f6.backward(grad)           # shape: (120,)

        # Reshape for C5: (120,) → (120, 1, 1)
        grad = grad.reshape(120, 1, 1)

        # C5 backward
        grad = self.c5.backward(grad)           # (16, 5, 5)

        # S4 backward
        grad = self.s4.backward(grad)           # (16, 10, 10)

        # C3 backward
        grad = self.c3.backward(grad)           # (6, 14, 14)

        # S2 backward
        grad = self.s2.backward(grad)           # (6, 28, 28)

        # C1 backward
        grad = self.c1.backward(grad)           # (1, 32, 32)

        # We don't need the gradient w.r.t. input (nothing before C1)
        return loss

    def update_weights(self, learning_rate):
        """
        Update all weights using Stochastic Gradient Descent (SGD).

        For each parameter:
            w_new = w_old - learning_rate × gradient

        This is the simplest form of gradient descent from the
        backpropagation paper. No momentum, no adaptive learning rate —
        just pure gradient descent as described by Rumelhart et al.

        Parameters
        ----------
        learning_rate : float
            Step size η (eta) for the weight update
        """
        for layer in self.layers:
            for param, grad in layer.get_params():
                param -= learning_rate * grad

    def predict(self, x):
        """
        Predict the digit class for an input image.

        Parameters
        ----------
        x : np.ndarray of shape (1, 32, 32)

        Returns
        -------
        int : predicted digit (0-9)
        np.ndarray : all 10 RBF distances
        """
        distances = self.forward(x)
        # The predicted class is the one with MINIMUM distance
        predicted = np.argmin(distances)
        return predicted, distances
