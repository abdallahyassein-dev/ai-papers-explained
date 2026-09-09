"""
============================================================================
BACKPROPAGATION FROM SCRATCH
Based on: "Learning Representations by Back-propagating Errors"
          Rumelhart, Hinton & Williams (1986), Nature 323, 533-536

Every line of math maps directly to an equation from the paper.
No frameworks (no PyTorch, no TensorFlow) — just NumPy.
============================================================================
"""

import numpy as np
import time

# ──────────────────────────────────────────────────────────────────────────
# SECTION 1: THE NEURAL NETWORK CLASS
# ──────────────────────────────────────────────────────────────────────────

class NeuralNetwork:
    """
    A feedforward neural network trained with backpropagation,
    implemented exactly as described in the 1986 paper.
    """

    def __init__(self, layer_sizes, learning_rate=0.5, momentum=0.9, seed=42):
        """
        Initialize the network.

        Parameters:
            layer_sizes: list of ints, e.g. [2, 4, 1] means:
                         2 input units, 4 hidden units, 1 output unit
            learning_rate: eta (η) from the paper
            momentum: alpha (α) from the paper
            seed: random seed for reproducibility
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.eta = learning_rate      # η — learning rate
        self.alpha = momentum         # α — momentum coefficient

        np.random.seed(seed)

        # ── Initialize weights and biases ──
        # Paper: "weights are initialized to small random values"
        self.weights = []   # weights[l] = weight matrix from layer l to layer l+1
        self.biases = []    # biases[l]  = bias vector for layer l+1

        for i in range(self.num_layers - 1):
            # Xavier initialization (small random values as paper suggests)
            limit = np.sqrt(6.0 / (layer_sizes[i] + layer_sizes[i + 1]))
            w = np.random.uniform(-limit, limit,
                                  size=(layer_sizes[i + 1], layer_sizes[i]))
            b = np.random.uniform(-limit, limit,
                                  size=(layer_sizes[i + 1], 1))
            self.weights.append(w)
            self.biases.append(b)

        # ── Initialize momentum terms (previous weight changes) ──
        # Paper equation: Δw_ji(t+1) = η·δ_j·o_i + α·Δw_ji(t)
        self.prev_dw = [np.zeros_like(w) for w in self.weights]
        self.prev_db = [np.zeros_like(b) for b in self.biases]

        # ── Training history ──
        self.error_history = []

    # ──────────────────────────────────────────────────────────────────
    # PAPER EQUATION 2: The Sigmoid Activation Function
    #
    #   f(x) = σ(x) = 1 / (1 + e^(-x))
    #
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function — Equation 2 from the paper."""
        # Clip to avoid overflow in exp
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    # ──────────────────────────────────────────────────────────────────
    # PAPER EQUATION 3: Sigmoid Derivative
    #
    #   f'(x) = f(x) · (1 - f(x))
    #
    # Note: computed from the OUTPUT of sigmoid, not from x directly!
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def sigmoid_derivative(output):
        """
        Derivative of sigmoid, computed from the sigmoid output itself.
        Equation 3 from the paper: f'(x) = f(x) * (1 - f(x))
        """
        return output * (1.0 - output)

    # ──────────────────────────────────────────────────────────────────
    # PAPER EQUATION 1 + 2: The Forward Pass
    #
    #   net_j = Σ_i (w_ji · o_i) + b_j     (Equation 1: net input)
    #   o_j   = f(net_j)                     (Equation 2: activation)
    #
    # ──────────────────────────────────────────────────────────────────
    def forward(self, X):
        """
        Forward pass — propagate input through the network.
        Returns list of activations for each layer (needed for backprop).
        """
        activations = [X]  # activations[0] = input data

        current = X
        for l in range(self.num_layers - 1):
            # Equation 1: net_j = Σ_i(w_ji · o_i) + b_j
            net = self.weights[l] @ current + self.biases[l]

            # Equation 2: o_j = σ(net_j)
            output = self.sigmoid(net)

            activations.append(output)
            current = output

        return activations

    # ──────────────────────────────────────────────────────────────────
    # PAPER EQUATION 4: Error Function
    #
    #   E = (1/2) · Σ_p Σ_k (y_pk - o_pk)²
    #
    # ──────────────────────────────────────────────────────────────────
    def compute_error(self, y_target, y_actual):
        """Compute sum of squared errors — Equation 4 from the paper."""
        return 0.5 * np.sum((y_target - y_actual) ** 2)

    # ──────────────────────────────────────────────────────────────────
    # PAPER EQUATIONS 5a, 5b, 6: The Backward Pass
    #
    #   5a. δ_k = (y_k - o_k) · f'(net_k)              (output delta)
    #   5b. δ_j = f'(net_j) · Σ_k(δ_k · w_kj)          (hidden delta)
    #   6.  Δw_ji = η · δ_j · o_i + α · Δw_ji(prev)    (weight update)
    #
    # ──────────────────────────────────────────────────────────────────
    def backward(self, activations, y_target):
        """
        Backward pass — compute deltas and update weights.
        This is the core of the 1986 paper.
        """
        num_samples = y_target.shape[1]

        # ── Step 1: Compute delta for the OUTPUT layer ──
        # Equation 5a: δ_k = (y_k - o_k) · f'(net_k)
        output = activations[-1]
        error = y_target - output                          # (y_k - o_k)
        delta = error * self.sigmoid_derivative(output)    # × f'(net_k)

        deltas = [None] * (self.num_layers - 1)
        deltas[-1] = delta

        # ── Step 2: Backpropagate delta to HIDDEN layers ──
        # Equation 5b: δ_j = f'(net_j) · Σ_k(δ_k · w_kj)
        for l in range(self.num_layers - 3, -1, -1):
            # Σ_k(δ_k · w_kj) — propagate error backward through weights
            weighted_error = self.weights[l + 1].T @ deltas[l + 1]

            # × f'(net_j) — multiply by sigmoid derivative
            delta = weighted_error * self.sigmoid_derivative(activations[l + 1])
            deltas[l] = delta

        # ── Step 3: Update weights and biases ──
        # Equation 6: Δw_ji = η · δ_j · o_i + α · Δw_ji(t)
        for l in range(self.num_layers - 1):
            # η · δ_j · o_i — the gradient step
            dw = (self.eta / num_samples) * (deltas[l] @ activations[l].T)
            db = (self.eta / num_samples) * np.sum(deltas[l], axis=1, keepdims=True)

            # + α · Δw_ji(t) — momentum term
            dw += self.alpha * self.prev_dw[l]
            db += self.alpha * self.prev_db[l]

            # Apply the update: w_ji(new) = w_ji(old) + Δw_ji
            self.weights[l] += dw
            self.biases[l] += db

            # Save for next momentum calculation
            self.prev_dw[l] = dw
            self.prev_db[l] = db

    # ──────────────────────────────────────────────────────────────────
    # TRAINING LOOP
    # ──────────────────────────────────────────────────────────────────
    def train(self, X, Y, epochs=1000, print_every=100, early_stop_error=None):
        """
        Train the network using backpropagation.

        Parameters:
            X: input data, shape (num_features, num_samples)
            Y: target data, shape (num_outputs, num_samples)
            epochs: number of training iterations
            print_every: how often to print progress
            early_stop_error: stop if error falls below this value
        """
        self.error_history = []

        for epoch in range(1, epochs + 1):
            # Forward pass
            activations = self.forward(X)

            # Compute error (Equation 4)
            error = self.compute_error(Y, activations[-1])
            self.error_history.append(error)

            # Backward pass (Equations 5a, 5b, 6)
            self.backward(activations, Y)

            # Print progress
            if epoch % print_every == 0 or epoch == 1:
                print(f"  Epoch {epoch:>6d}  |  Error: {error:.6f}")

            # Early stopping
            if early_stop_error and error < early_stop_error:
                print(f"  Epoch {epoch:>6d}  |  Error: {error:.6f}  <-- Early stop!")
                break

        return self.error_history

    # ──────────────────────────────────────────────────────────────────
    # PREDICTION
    # ──────────────────────────────────────────────────────────────────
    def predict(self, X):
        """Run a forward pass and return the output."""
        activations = self.forward(X)
        return activations[-1]

    def get_hidden_representation(self, X, layer=1):
        """Get the internal representation at a specific hidden layer."""
        activations = self.forward(X)
        return activations[layer]


# ──────────────────────────────────────────────────────────────────────────
# HELPER: Print a nice ASCII error curve
# ──────────────────────────────────────────────────────────────────────────
def plot_ascii_error_curve(error_history, width=60, height=15, title="Training Error"):
    """Print an ASCII art graph of the error curve."""
    if not error_history:
        return

    # Sample points to fit the width
    n = len(error_history)
    step = max(1, n // width)
    sampled = [error_history[i] for i in range(0, n, step)]

    max_err = max(sampled)
    min_err = min(sampled)
    err_range = max_err - min_err if max_err != min_err else 1.0

    print(f"\n  {title}")
    print(f"  {'─' * (width + 8)}")

    for row in range(height):
        threshold = max_err - (row / (height - 1)) * err_range
        label = f"{threshold:.4f}" if row % 3 == 0 else "      "
        line = f"  {label:>7s} │"
        for val in sampled:
            if abs(val - threshold) < err_range / (height * 2):
                line += "●"
            elif val >= threshold:
                line += " "
            else:
                line += " "
        print(line)

    print(f"  {'':>7s} └{'─' * width}")
    print(f"  {'':>8s} Epoch 1{' ' * (width - 14)}Epoch {n}")


# ══════════════════════════════════════════════════════════════════════════
# EXAMPLE 1: XOR PROBLEM
# The classic example from the paper — impossible for a single-layer network
# ══════════════════════════════════════════════════════════════════════════
def example_xor():
    print("\n" + "=" * 70)
    print("  EXAMPLE 1: THE XOR PROBLEM")
    print("  From the paper: Figure 2")
    print("  A single-layer network CANNOT solve this.")
    print("  The hidden layer must learn to warp the input space.")
    print("=" * 70)

    # ── Data ──
    # XOR truth table
    X = np.array([[0, 0, 1, 1],    # x1
                  [0, 1, 0, 1]])    # x2

    Y = np.array([[0, 1, 1, 0]])    # XOR output

    print("\n  Input Data:")
    print("  ┌────┬────┬─────┐")
    print("  │ x1 │ x2 │ XOR │")
    print("  ├────┼────┼─────┤")
    for i in range(4):
        print(f"  │  {int(X[0,i])} │  {int(X[1,i])} │  {int(Y[0,i])}  │")
    print("  └────┴────┴─────┘")

    # ── Network: 2 inputs → 4 hidden → 1 output ──
    nn = NeuralNetwork(
        layer_sizes=[2, 4, 1],
        learning_rate=2.0,
        momentum=0.9,
        seed=42
    )

    print("\n  Network Architecture: 2 → 4 → 1")
    print("  Learning Rate (η): 2.0")
    print("  Momentum (α): 0.9")
    print("\n  Training...")

    errors = nn.train(X, Y, epochs=5000, print_every=1000, early_stop_error=0.001)

    # ── Results ──
    predictions = nn.predict(X)
    print("\n  Results after training:")
    print("  ┌────┬────┬──────────┬──────────┬───────────┐")
    print("  │ x1 │ x2 │ Expected │ Actual   │ Rounded   │")
    print("  ├────┼────┼──────────┼──────────┼───────────┤")
    for i in range(4):
        pred = predictions[0, i]
        rounded = 1 if pred > 0.5 else 0
        correct = "✓" if rounded == int(Y[0, i]) else "✗"
        print(f"  │  {int(X[0,i])} │  {int(X[1,i])} │   {int(Y[0,i]):>4d}   │ {pred:>8.4f} │   {rounded}   {correct}   │")
    print("  └────┴────┴──────────┴──────────┴───────────┘")

    # ── Show internal representations ──
    hidden = nn.get_hidden_representation(X, layer=1)
    print("\n  ★ Internal Representations (Hidden Layer Activations):")
    print("  This is what the paper calls 'learned representations'!")
    print("  ┌────┬────┬───────────┬───────────┬───────────┬───────────┐")
    print("  │ x1 │ x2 │   h1      │   h2      │   h3      │   h4      │")
    print("  ├────┼────┼───────────┼───────────┼───────────┼───────────┤")
    for i in range(4):
        h_vals = "│".join([f" {hidden[j,i]:>8.4f} " for j in range(hidden.shape[0])])
        print(f"  │  {int(X[0,i])} │  {int(X[1,i])} │{h_vals}│")
    print("  └────┴────┴───────────┴───────────┴───────────┴───────────┘")

    plot_ascii_error_curve(errors, title="XOR Training Error")

    # ── Show learned weights ──
    print("\n  Learned Weights:")
    for l in range(len(nn.weights)):
        layer_name = "Hidden" if l == 0 else "Output"
        print(f"\n  {layer_name} Layer Weights (w):")
        print(f"  {nn.weights[l].round(4)}")
        print(f"  {layer_name} Layer Biases (b):")
        print(f"  {nn.biases[l].round(4).flatten()}")


# ══════════════════════════════════════════════════════════════════════════
# EXAMPLE 2: SYMMETRY DETECTION
# From the paper: Can the network detect if a binary pattern is symmetric?
# ══════════════════════════════════════════════════════════════════════════
def example_symmetry():
    print("\n\n" + "=" * 70)
    print("  EXAMPLE 2: SYMMETRY DETECTION")
    print("  From the paper: Section on symmetry")
    print("  Input: a 6-bit binary pattern")
    print("  Output: 1 if symmetric about center, 0 otherwise")
    print("  The hidden layer must learn to compare opposite ends.")
    print("=" * 70)

    # ── Generate all 6-bit patterns and label them ──
    patterns = []
    labels = []

    for i in range(64):  # 2^6 = 64 patterns
        bits = [(i >> b) & 1 for b in range(5, -1, -1)]
        is_symmetric = 1 if bits == bits[::-1] else 0
        patterns.append(bits)
        labels.append(is_symmetric)

    X = np.array(patterns).T           # Shape: (6, 64)
    Y = np.array([labels])             # Shape: (1, 64)

    num_symmetric = sum(labels)
    num_asymmetric = len(labels) - num_symmetric

    print(f"\n  Total patterns: {len(labels)}")
    print(f"  Symmetric: {num_symmetric}  |  Asymmetric: {num_asymmetric}")

    print("\n  Sample patterns:")
    print("  ┌──────────────────┬───────────┐")
    print("  │ Pattern          │ Symmetric │")
    print("  ├──────────────────┼───────────┤")
    examples = [(0, "000000"), (9, "001001"), (21, "010101"),
                (27, "011011"), (33, "100001"), (63, "111111"),
                (1, "000001"), (15, "001111"), (42, "101010")]
    for idx, bits_str in examples:
        sym = "Yes ✓" if labels[idx] == 1 else "No  ✗"
        formatted = " ".join(bits_str)
        print(f"  │ [{formatted}]  │   {sym}   │")
    print("  └──────────────────┴───────────┘")

    # ── Network: 6 inputs → 4 hidden → 1 output ──
    nn = NeuralNetwork(
        layer_sizes=[6, 4, 1],
        learning_rate=3.0,
        momentum=0.9,
        seed=123
    )

    print("\n  Network Architecture: 6 → 4 → 1")
    print("  Learning Rate (η): 3.0")
    print("  Momentum (α): 0.9")
    print("\n  Training...")

    errors = nn.train(X, Y, epochs=5000, print_every=1000, early_stop_error=0.1)

    # ── Results ──
    predictions = nn.predict(X)
    rounded = (predictions > 0.5).astype(int)
    accuracy = np.mean(rounded == Y) * 100

    print(f"\n  Accuracy: {accuracy:.1f}%")

    # ── Show some predictions ──
    print("\n  Sample Predictions:")
    print("  ┌──────────────────┬──────────┬──────────┬─────────┐")
    print("  │ Pattern          │ Expected │ Actual   │ Correct │")
    print("  ├──────────────────┼──────────┼──────────┼─────────┤")
    test_indices = [0, 9, 21, 27, 33, 63, 1, 15, 42]
    for idx in test_indices:
        bits = "".join([str(int(X[j, idx])) for j in range(6)])
        formatted = " ".join(bits)
        pred = predictions[0, idx]
        expected = int(Y[0, idx])
        r = 1 if pred > 0.5 else 0
        correct = "✓" if r == expected else "✗"
        print(f"  │ [{formatted}]  │    {expected}     │  {pred:.4f} │    {correct}    │")
    print("  └──────────────────┴──────────┴──────────┴─────────┘")

    # ── Analyze hidden representations ──
    hidden = nn.get_hidden_representation(X, layer=1)
    print("\n  ★ Hidden Layer Analysis:")
    print("  The network should learn units that compare opposite positions:")
    print("  h1 ~ compares (x1, x6), h2 ~ compares (x2, x5), etc.")
    print("\n  Weights from input to hidden layer:")
    print("  Each ROW is a hidden unit, each COLUMN is an input (x1..x6):")
    print()
    header = "  Unit   " + "".join([f"   x{i+1}   " for i in range(6)])
    print(header)
    print("  " + "─" * (len(header) - 2))
    for h in range(nn.weights[0].shape[0]):
        vals = "".join([f"  {nn.weights[0][h, i]:>6.3f} " for i in range(6)])
        print(f"   h{h+1}   {vals}")

    print("\n  Notice: each hidden unit develops OPPOSITE weights for")
    print("  positions that should match (x1↔x6, x2↔x5, x3↔x4)!")

    plot_ascii_error_curve(errors, title="Symmetry Detection Training Error")


# ══════════════════════════════════════════════════════════════════════════
# EXAMPLE 3: IRIS FLOWER CLASSIFICATION
# A real-world dataset — classifying 3 species of iris flowers
# using 4 measurements (sepal/petal length and width)
# ══════════════════════════════════════════════════════════════════════════
def example_iris():
    print("\n\n" + "=" * 70)
    print("  EXAMPLE 3: IRIS FLOWER CLASSIFICATION")
    print("  A real-world dataset with 150 samples, 4 features, 3 classes")
    print("  Features: sepal length, sepal width, petal length, petal width")
    print("  Classes:  Setosa, Versicolor, Virginica")
    print("=" * 70)

    # ── Iris dataset (embedded — no external dependencies) ──
    # 50 samples per class, 4 features each
    # Source: Fisher, R.A. (1936)
    iris_data = [
        # Setosa (class 0) — 50 samples
        [5.1,3.5,1.4,0.2,0], [4.9,3.0,1.4,0.2,0], [4.7,3.2,1.3,0.2,0],
        [4.6,3.1,1.5,0.2,0], [5.0,3.6,1.4,0.2,0], [5.4,3.9,1.7,0.4,0],
        [4.6,3.4,1.4,0.3,0], [5.0,3.4,1.5,0.2,0], [4.4,2.9,1.4,0.2,0],
        [4.9,3.1,1.5,0.1,0], [5.4,3.7,1.5,0.2,0], [4.8,3.4,1.6,0.2,0],
        [4.8,3.0,1.4,0.1,0], [4.3,3.0,1.1,0.1,0], [5.8,4.0,1.2,0.2,0],
        [5.7,4.4,1.5,0.4,0], [5.4,3.9,1.3,0.4,0], [5.1,3.5,1.4,0.3,0],
        [5.7,3.8,1.7,0.3,0], [5.1,3.8,1.5,0.3,0], [5.4,3.4,1.7,0.2,0],
        [5.1,3.7,1.5,0.4,0], [4.6,3.6,1.0,0.2,0], [5.1,3.3,1.7,0.5,0],
        [4.8,3.4,1.9,0.2,0], [5.0,3.0,1.6,0.2,0], [5.0,3.4,1.6,0.4,0],
        [5.2,3.5,1.5,0.2,0], [5.2,3.4,1.4,0.2,0], [4.7,3.2,1.6,0.2,0],
        [4.8,3.1,1.6,0.2,0], [5.4,3.4,1.5,0.4,0], [5.2,4.1,1.5,0.1,0],
        [5.5,4.2,1.4,0.2,0], [4.9,3.1,1.5,0.2,0], [5.0,3.2,1.2,0.2,0],
        [5.5,3.5,1.3,0.2,0], [4.9,3.6,1.4,0.1,0], [4.4,3.0,1.3,0.2,0],
        [5.1,3.4,1.5,0.2,0], [5.0,3.5,1.3,0.3,0], [4.5,2.3,1.3,0.3,0],
        [4.4,3.2,1.3,0.2,0], [5.0,3.5,1.6,0.6,0], [5.1,3.8,1.9,0.4,0],
        [4.8,3.0,1.4,0.3,0], [5.1,3.8,1.6,0.2,0], [4.6,3.2,1.4,0.2,0],
        [5.3,3.7,1.5,0.2,0], [5.0,3.3,1.4,0.2,0],
        # Versicolor (class 1) — 50 samples
        [7.0,3.2,4.7,1.4,1], [6.4,3.2,4.5,1.5,1], [6.9,3.1,4.9,1.5,1],
        [5.5,2.3,4.0,1.3,1], [6.5,2.8,4.6,1.5,1], [5.7,2.8,4.5,1.3,1],
        [6.3,3.3,4.7,1.6,1], [4.9,2.4,3.3,1.0,1], [6.6,2.9,4.6,1.3,1],
        [5.2,2.7,3.9,1.4,1], [5.0,2.0,3.5,1.0,1], [5.9,3.0,4.2,1.5,1],
        [6.0,2.2,4.0,1.0,1], [6.1,2.9,4.7,1.4,1], [5.6,2.9,3.6,1.3,1],
        [6.7,3.1,4.4,1.4,1], [5.6,3.0,4.5,1.5,1], [5.8,2.7,4.1,1.0,1],
        [6.2,2.2,4.5,1.5,1], [5.6,2.5,3.9,1.1,1], [5.9,3.2,4.8,1.8,1],
        [6.1,2.8,4.0,1.3,1], [6.3,2.5,4.9,1.5,1], [6.1,2.8,4.7,1.2,1],
        [6.4,2.9,4.3,1.3,1], [6.6,3.0,4.4,1.4,1], [6.8,2.8,4.8,1.4,1],
        [6.7,3.0,5.0,1.7,1], [6.0,2.9,4.5,1.5,1], [5.7,2.6,3.5,1.0,1],
        [5.5,2.4,3.8,1.1,1], [5.5,2.4,3.7,1.0,1], [5.8,2.7,3.9,1.2,1],
        [6.0,2.7,5.1,1.6,1], [5.4,3.0,4.5,1.5,1], [6.0,3.4,4.5,1.6,1],
        [6.7,3.1,4.7,1.5,1], [6.3,2.3,4.4,1.3,1], [5.6,3.0,4.1,1.3,1],
        [5.5,2.5,4.0,1.3,1], [5.5,2.6,4.4,1.2,1], [6.1,3.0,4.6,1.4,1],
        [5.8,2.6,4.0,1.2,1], [5.0,2.3,3.3,1.0,1], [5.6,2.7,4.2,1.3,1],
        [5.7,3.0,4.2,1.2,1], [5.7,2.9,4.2,1.3,1], [6.2,2.9,4.3,1.3,1],
        [5.1,2.5,3.0,1.1,1], [5.7,2.8,4.1,1.3,1],
        # Virginica (class 2) — 50 samples
        [6.3,3.3,6.0,2.5,2], [5.8,2.7,5.1,1.9,2], [7.1,3.0,5.9,2.1,2],
        [6.3,2.9,5.6,1.8,2], [6.5,3.0,5.8,2.2,2], [7.6,3.0,6.6,2.1,2],
        [4.9,2.5,4.5,1.7,2], [7.3,2.9,6.3,1.8,2], [6.7,2.5,5.8,1.8,2],
        [7.2,3.6,6.1,2.5,2], [6.5,3.2,5.1,2.0,2], [6.4,2.7,5.3,1.9,2],
        [6.8,3.0,5.5,2.1,2], [5.7,2.5,5.0,2.0,2], [5.8,2.8,5.1,2.4,2],
        [6.4,3.2,5.3,2.3,2], [6.5,3.0,5.5,1.8,2], [7.7,3.8,6.7,2.2,2],
        [7.7,2.6,6.9,2.3,2], [6.0,2.2,5.0,1.5,2], [6.9,3.2,5.7,2.3,2],
        [5.6,2.8,4.9,2.0,2], [7.7,2.8,6.7,2.0,2], [6.3,2.7,4.9,1.8,2],
        [6.7,3.3,5.7,2.1,2], [7.2,3.2,6.0,1.8,2], [6.2,2.8,4.8,1.8,2],
        [6.1,3.0,4.9,1.8,2], [6.4,2.8,5.6,2.1,2], [7.2,3.0,5.8,1.6,2],
        [7.4,2.8,6.1,1.9,2], [7.9,3.8,6.4,2.0,2], [6.4,2.8,5.6,2.2,2],
        [6.3,2.8,5.1,1.5,2], [6.1,2.6,5.6,1.4,2], [7.7,3.0,6.1,2.3,2],
        [6.3,3.4,5.6,2.4,2], [6.4,3.1,5.5,1.8,2], [6.0,3.0,4.8,1.8,2],
        [6.9,3.1,5.4,2.1,2], [6.7,3.1,5.6,2.4,2], [6.9,3.1,5.1,2.3,2],
        [5.8,2.7,5.1,1.9,2], [6.8,3.2,5.9,2.3,2], [6.7,3.3,5.7,2.5,2],
        [6.7,3.0,5.2,2.3,2], [6.3,2.5,5.0,1.9,2], [6.5,3.0,5.2,2.0,2],
        [6.2,3.4,5.4,2.3,2], [5.9,3.0,5.1,1.8,2],
    ]

    data = np.array(iris_data)
    features = data[:, :4]
    labels = data[:, 4].astype(int)

    # ── Normalize features to [0, 1] ──
    # Important: sigmoid outputs are in (0,1), so inputs should be scaled
    feat_min = features.min(axis=0)
    feat_max = features.max(axis=0)
    features_norm = (features - feat_min) / (feat_max - feat_min)

    # ── One-hot encode targets ──
    # 3 classes → 3 output units
    # Setosa:     [1, 0, 0]
    # Versicolor: [0, 1, 0]
    # Virginica:  [0, 0, 1]
    targets = np.zeros((150, 3))
    for i, label in enumerate(labels):
        targets[i, label] = 1.0

    # ── Shuffle and split: 120 train, 30 test ──
    np.random.seed(42)
    indices = np.random.permutation(150)
    train_idx = indices[:120]
    test_idx = indices[120:]

    X_train = features_norm[train_idx].T  # Shape: (4, 120)
    Y_train = targets[train_idx].T        # Shape: (3, 120)
    X_test = features_norm[test_idx].T    # Shape: (4, 30)
    Y_test = targets[test_idx].T          # Shape: (3, 30)
    test_labels = labels[test_idx]

    print(f"\n  Dataset: 150 samples, 4 features, 3 classes")
    print(f"  Training set: {X_train.shape[1]} samples")
    print(f"  Test set:     {X_test.shape[1]} samples")

    print("\n  Sample data (first 5):")
    print("  ┌───────┬───────┬───────┬───────┬────────────┐")
    print("  │ Sepal │ Sepal │ Petal │ Petal │ Species    │")
    print("  │  Len  │  Wid  │  Len  │  Wid  │            │")
    print("  ├───────┼───────┼───────┼───────┼────────────┤")
    species = ["Setosa", "Versicolor", "Virginica"]
    for i in range(5):
        s = species[int(data[i, 4])]
        print(f"  │ {data[i,0]:>5.1f} │ {data[i,1]:>5.1f} │ {data[i,2]:>5.1f} │ {data[i,3]:>5.1f} │ {s:<10s} │")
    print("  └───────┴───────┴───────┴───────┴────────────┘")

    # ── Network: 4 inputs → 8 hidden → 3 outputs ──
    nn = NeuralNetwork(
        layer_sizes=[4, 8, 3],
        learning_rate=1.5,
        momentum=0.9,
        seed=42
    )

    print("\n  Network Architecture: 4 → 8 → 3")
    print("  Learning Rate (η): 1.5")
    print("  Momentum (α): 0.9")
    print("\n  Training...")

    errors = nn.train(X_train, Y_train, epochs=3000, print_every=500)

    # ── Evaluate on test set ──
    predictions = nn.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=0)
    correct = np.sum(predicted_classes == test_labels)
    accuracy = correct / len(test_labels) * 100

    print(f"\n  ── Test Set Results ──")
    print(f"  Accuracy: {correct}/{len(test_labels)} = {accuracy:.1f}%")

    # ── Confusion matrix ──
    print("\n  Confusion Matrix:")
    print("  ┌────────────┬──────────┬────────────┬───────────┐")
    print("  │ Actual ↓   │ Pred.    │ Pred.      │ Pred.     │")
    print("  │ Pred. →    │ Setosa   │ Versicolor │ Virginica │")
    print("  ├────────────┼──────────┼────────────┼───────────┤")
    for actual_class in range(3):
        row = []
        for pred_class in range(3):
            count = np.sum((test_labels == actual_class) &
                          (predicted_classes == pred_class))
            row.append(count)
        print(f"  │ {species[actual_class]:<10s} │    {row[0]:>2d}    │     {row[1]:>2d}     │    {row[2]:>2d}     │")
    print("  └────────────┴──────────┴────────────┴───────────┘")

    # ── Show some test predictions ──
    print("\n  Sample Test Predictions:")
    print("  ┌───┬────────────┬────────────┬───────────────────────┬─────┐")
    print("  │ # │ Actual     │ Predicted  │ Output Probabilities  │  OK │")
    print("  ├───┼────────────┼────────────┼───────────────────────┼─────┤")
    for i in range(min(15, len(test_labels))):
        actual = species[test_labels[i]]
        predicted = species[predicted_classes[i]]
        probs = " ".join([f"{predictions[j,i]:.2f}" for j in range(3)])
        ok = "✓" if predicted_classes[i] == test_labels[i] else "✗"
        print(f"  │{i+1:>2d} │ {actual:<10s} │ {predicted:<10s} │ [{probs}] │  {ok}  │")
    print("  └───┴────────────┴────────────┴───────────────────────┴─────┘")

    # ── Show hidden representations ──
    print("\n  ★ Internal Representations:")
    print("  Hidden layer outputs for one sample of each class:")
    for cls in range(3):
        idx = np.where(test_labels == cls)[0][0]
        hidden = nn.get_hidden_representation(X_test[:, idx:idx+1], layer=1)
        vals = " ".join([f"{hidden[j,0]:.3f}" for j in range(hidden.shape[0])])
        print(f"    {species[cls]:<12s}: [{vals}]")

    plot_ascii_error_curve(errors, title="Iris Classification Training Error")


# ══════════════════════════════════════════════════════════════════════════
# MAIN — Run all 3 examples
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█  BACKPROPAGATION FROM SCRATCH" + " " * 38 + "█")
    print("█  Based on Rumelhart, Hinton & Williams (1986)" + " " * 22 + "█")
    print("█  'Learning Representations by Back-propagating Errors'" + " " * 14 + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    start = time.time()

    example_xor()
    example_symmetry()
    example_iris()

    elapsed = time.time() - start

    print("\n\n" + "=" * 70)
    print(f"  All 3 examples completed in {elapsed:.2f} seconds")
    print("=" * 70)
    print("""
  Summary of Paper Equations → Code Mapping:
  ┌──────────────┬─────────────────────────────────────┬──────────────────┐
  │ Equation     │ Formula                             │ Method           │
  ├──────────────┼─────────────────────────────────────┼──────────────────┤
  │ Eq 1: Net    │ net_j = SUM(w_ji * o_i) + b_j       │ forward()        │
  │ Eq 2: Activ. │ o_j = sigma(net_j)                  │ sigmoid()        │
  │ Eq 3: Deriv. │ f'(x) = f(x) * (1 - f(x))          │ sigmoid_deriv()  │
  │ Eq 4: Error  │ E = 0.5 * SUM(y - o)^2              │ compute_error()  │
  │ Eq 5a: d_out │ d_k = (y_k - o_k) * f'(net_k)      │ backward()       │
  │ Eq 5b: d_hid │ d_j = f'(net_j) * SUM(d_k * w_kj)  │ backward()       │
  │ Eq 6: Update │ Dw = eta*d_j*o_i + alpha*Dw(prev)   │ backward()       │
  └──────────────┴─────────────────────────────────────┴──────────────────┘
    """)
