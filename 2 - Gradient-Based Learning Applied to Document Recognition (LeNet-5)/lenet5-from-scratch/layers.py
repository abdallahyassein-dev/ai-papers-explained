"""
layers.py — All Layer Implementations for LeNet-5 (From Scratch)

This file implements every layer type used in LeNet-5:

  1. ConvLayer       — Convolutional layer with optional partial connections
  2. SubSamplingLayer — The paper's trainable 2×2 average pooling
  3. FullyConnectedLayer — Standard dense layer
  4. RBFOutputLayer  — Euclidean distance to fixed target bitmaps

Each layer follows this interface:
  - forward(input)  → output     (stores input/output for backprop)
  - backward(grad)  → grad_input (computes parameter gradients)
  - get_params()    → list of (weight, grad) tuples
  - param_count()   → number of trainable parameters

CONNECTION TO BACKPROPAGATION PAPER:
------------------------------------
Remember from Rumelhart et al. (1986):
  ∂E/∂w = ∂E/∂y × ∂y/∂w      (chain rule)

In a CNN, this is the same principle, but:
  - For convolution: ∂y/∂w involves summing over all spatial positions
    where the kernel was applied (because of weight sharing)
  - For sub-sampling: ∂y/∂w involves the sum of the 2×2 block
  - For fully connected: it's exactly what you learned in the backprop paper
"""

import numpy as np
from activations import scaled_tanh, scaled_tanh_derivative


class ConvLayer:
    """
    Convolutional Layer — The core building block of CNNs.

    How convolution works (review from the paper explanation):
    ─────────────────────────────────────────────────────────
    A small kernel (e.g., 5×5) slides across the input image.
    At each position, we compute:
        output[i,j] = Σ_m Σ_n kernel[m,n] × input[i+m, j+n] + bias

    Then apply activation:
        output[i,j] = f(output[i,j])

    Key features of this implementation:
    - Supports MULTIPLE input feature maps (e.g., C3 takes 6 maps from S2)
    - Supports PARTIAL CONNECTIONS (C3 doesn't connect to all S2 maps)
    - Weight sharing: same kernel applied at every spatial position

    Parameters
    ----------
    in_channels : int
        Number of input feature maps
    out_channels : int
        Number of output feature maps (kernels)
    kernel_size : int
        Size of each square kernel (e.g., 5 for 5×5)
    input_size : int
        Spatial size of input (e.g., 32 for 32×32)
    connection_table : list of list of int, optional
        connection_table[j] = list of input channel indices connected
        to output channel j. If None, fully connected.
    """

    def __init__(self, in_channels, out_channels, kernel_size, input_size,
                 connection_table=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.output_size = input_size - kernel_size + 1

        # Connection table: which input maps connect to which output maps
        if connection_table is None:
            # Fully connected: each output map sees ALL input maps
            self.connection_table = [list(range(in_channels))] * out_channels
        else:
            self.connection_table = connection_table

        # ──────────────────────────────────────────────────────────────
        # Weight Initialization (from the paper, Section II):
        #   W ~ Uniform(-2.4/fan_in, +2.4/fan_in)
        #   fan_in = number of inputs to each neuron
        #          = num_connected_inputs × kernel_size²
        #
        # Why 2.4/fan_in?
        #   This keeps initial activations in the linear region of tanh,
        #   where gradients are large → fast initial learning.
        #   If weights were too large, tanh saturates and gradients vanish.
        # ──────────────────────────────────────────────────────────────

        # Initialize kernels: one kernel per (output_channel, input_channel) pair
        # Shape: kernels[out_ch] is a dict mapping in_ch → kernel matrix
        self.kernels = {}
        self.kernel_grads = {}
        for j in range(out_channels):
            connected = self.connection_table[j]
            fan_in = len(connected) * kernel_size * kernel_size
            limit = 2.4 / fan_in
            self.kernels[j] = {}
            self.kernel_grads[j] = {}
            for i in connected:
                self.kernels[j][i] = np.random.uniform(
                    -limit, limit, (kernel_size, kernel_size)
                )
                self.kernel_grads[j][i] = np.zeros((kernel_size, kernel_size))

        # One bias per output feature map
        self.biases = np.zeros(out_channels)
        self.bias_grads = np.zeros(out_channels)

        # Storage for forward/backward pass
        self._input = None
        self._pre_activation = None
        self._output = None

    def forward(self, x):
        """
        Forward pass: Apply convolution + activation.

        Parameters
        ----------
        x : np.ndarray of shape (in_channels, H, W)

        Returns
        -------
        np.ndarray of shape (out_channels, H_out, W_out)
            where H_out = H - kernel_size + 1

        Step-by-step for one output feature map j:
        ───────────────────────────────────────────
        1. For each connected input map i:
           - Convolve input[i] with kernel[j][i]
        2. Sum all convolution results
        3. Add bias[j]
        4. Apply scaled_tanh activation
        """
        self._input = x
        out_h = self.output_size
        out_w = self.output_size
        k = self.kernel_size

        # Pre-activation (before applying tanh)
        self._pre_activation = np.zeros((self.out_channels, out_h, out_w))

        for j in range(self.out_channels):
            for i in self.connection_table[j]:
                # Convolve input channel i with kernel[j][i]
                # This is the "valid" convolution (no padding)
                self._pre_activation[j] += self._convolve2d(x[i], self.kernels[j][i])
            # Add bias
            self._pre_activation[j] += self.biases[j]

        # Apply activation
        self._output = scaled_tanh(self._pre_activation)
        return self._output

    def backward(self, grad_output):
        """
        Backward pass: Compute gradients for kernels, biases, and input.

        Parameters
        ----------
        grad_output : np.ndarray of shape (out_channels, H_out, W_out)
            Gradient of loss w.r.t. this layer's OUTPUT.

        Returns
        -------
        grad_input : np.ndarray of shape (in_channels, H, W)
            Gradient of loss w.r.t. this layer's INPUT (to pass backward).

        The math (connecting to your backpropagation knowledge):
        ────────────────────────────────────────────────────────
        Let's call:
          y = f(net)       where net = Σ conv(input, kernel) + bias
          δ = ∂E/∂net = ∂E/∂y × f'(net)    ← the "error signal"

        Then:
          ∂E/∂kernel[m,n] = Σ_i,j  δ[i,j] × input[i+m, j+n]
                            ↑ this is also a convolution!

          ∂E/∂bias = Σ_i,j  δ[i,j]
                     ↑ just sum all error signals

          ∂E/∂input[i,j] = Σ_m,n  δ[i-m, j-n] × kernel[m,n]
                            ↑ this is a "full" convolution with flipped kernel
        """
        k = self.kernel_size

        # Step 1: Compute δ (error signal) = grad_output × f'(pre_activation)
        # We use the efficient form: f'(a) computed from output f(a)
        delta = grad_output * scaled_tanh_derivative(self._output)

        # Step 2: Compute gradient for biases
        # ∂E/∂bias[j] = sum of all δ[j] values across spatial dimensions
        for j in range(self.out_channels):
            self.bias_grads[j] = np.sum(delta[j])

        # Step 3: Compute gradient for kernels
        # ∂E/∂kernel[j][i][m,n] = Σ_{r,c} δ[j][r,c] × input[i][r+m, c+n]
        # This is: correlate(input[i], δ[j])
        for j in range(self.out_channels):
            for i in self.connection_table[j]:
                self.kernel_grads[j][i] = self._convolve2d(self._input[i], delta[j])

        # Step 4: Compute gradient for input (to pass to previous layer)
        # ∂E/∂input[i][r,c] = Σ_j Σ_{m,n} δ[j][r-m, c-n] × kernel[j][i][m,n]
        # This is: full_convolve(δ[j], flip(kernel[j][i]))
        grad_input = np.zeros_like(self._input)
        for j in range(self.out_channels):
            for i in self.connection_table[j]:
                # Flip kernel 180 degrees (required for backprop through convolution)
                flipped_kernel = self.kernels[j][i][::-1, ::-1]
                # Full convolution: pad δ and convolve
                grad_input[i] += self._full_convolve2d(delta[j], flipped_kernel)

        return grad_input

    def _convolve2d(self, image, kernel):
        """
        2D valid cross-correlation (what deep learning calls "convolution").

        Slides kernel across image, computing dot product at each position.

        Technical note: Mathematically, "convolution" flips the kernel,
        but in deep learning we skip the flip (cross-correlation).
        Since kernels are learned, flipping doesn't matter — the network
        just learns a flipped version of what true convolution would learn.
        """
        k = kernel.shape[0]
        out_h = image.shape[0] - k + 1
        out_w = image.shape[1] - k + 1
        output = np.zeros((out_h, out_w))

        for i in range(out_h):
            for j in range(out_w):
                # Extract the patch and compute dot product with kernel
                patch = image[i:i + k, j:j + k]
                output[i, j] = np.sum(patch * kernel)

        return output

    def _full_convolve2d(self, image, kernel):
        """
        2D full convolution (with zero-padding).

        Used in backpropagation to compute gradient w.r.t. input.
        The output is LARGER than the input:
          output_size = image_size + kernel_size - 1

        This "spreads" the error signal back to all input positions
        that contributed to each output position.
        """
        k = kernel.shape[0]
        # Pad the image with zeros on all sides
        padded = np.pad(image, k - 1, mode="constant", constant_values=0)
        return self._convolve2d(padded, kernel)

    def get_params(self):
        """Return list of (param, grad) tuples for SGD update."""
        params = []
        for j in range(self.out_channels):
            for i in self.connection_table[j]:
                params.append((self.kernels[j][i], self.kernel_grads[j][i]))
        params.append((self.biases, self.bias_grads))
        return params

    def param_count(self):
        """Count total trainable parameters."""
        count = 0
        for j in range(self.out_channels):
            count += len(self.connection_table[j]) * self.kernel_size ** 2
        count += self.out_channels  # biases
        return count


class SubSamplingLayer:
    """
    Sub-Sampling (Pooling) Layer — Reduces spatial resolution by 2×.

    THIS IS NOT SIMPLE AVERAGE POOLING!

    From the paper (Section II-B):
    Each sub-sampling unit computes:
        output = f(w × Σ(2×2 block) + b)

    Where:
    - w is a TRAINABLE scalar weight (one per feature map)
    - b is a TRAINABLE scalar bias (one per feature map)
    - f is the scaled tanh activation
    - Σ(2×2 block) is the sum of 4 pixels in a non-overlapping 2×2 region

    The weight w controls HOW MUCH the network "cares about" the average
    activation in each region. The bias b shifts the operating point.

    Parameters
    ----------
    num_channels : int
        Number of feature maps (each gets its own w and b)
    input_size : int
        Spatial size of input (e.g., 28 → output will be 14)
    """

    def __init__(self, num_channels, input_size):
        self.num_channels = num_channels
        self.input_size = input_size
        self.output_size = input_size // 2

        # Trainable parameters: one weight and one bias per feature map
        # Initialize weights to small positive values
        fan_in = 4  # each output sees a 2×2 = 4 input values
        limit = 2.4 / fan_in
        self.weights = np.random.uniform(-limit, limit, num_channels)
        self.biases = np.zeros(num_channels)

        # Gradients
        self.weight_grads = np.zeros(num_channels)
        self.bias_grads = np.zeros(num_channels)

        # Storage
        self._input = None
        self._sums = None  # sum of each 2×2 block (before weight/bias/activation)
        self._pre_activation = None
        self._output = None

    def forward(self, x):
        """
        Forward pass: 2×2 sub-sampling with trainable parameters.

        Parameters
        ----------
        x : np.ndarray of shape (num_channels, H, W)

        Returns
        -------
        np.ndarray of shape (num_channels, H//2, W//2)

        Step by step for channel c, position (i,j):
        ──────────────────────────────────────────
        1. Take the 2×2 block: x[c, 2i:2i+2, 2j:2j+2]
        2. Sum the 4 values: s = x[c,2i,2j] + x[c,2i,2j+1] + x[c,2i+1,2j] + x[c,2i+1,2j+1]
        3. Multiply by weight: net = w[c] × s + b[c]
        4. Apply activation: out = f(net)
        """
        self._input = x
        out_h = self.output_size
        out_w = self.output_size

        # Compute 2×2 block sums using reshaping (fast!)
        # Reshape (C, H, W) → (C, H/2, 2, W/2, 2) → sum over axes 2,4
        self._sums = x.reshape(
            self.num_channels, out_h, 2, out_w, 2
        ).sum(axis=(2, 4))

        # Apply trainable weight and bias
        self._pre_activation = np.zeros_like(self._sums)
        for c in range(self.num_channels):
            self._pre_activation[c] = self.weights[c] * self._sums[c] + self.biases[c]

        # Apply activation
        self._output = scaled_tanh(self._pre_activation)
        return self._output

    def backward(self, grad_output):
        """
        Backward pass: Compute gradients for weights, biases, and input.

        Parameters
        ----------
        grad_output : np.ndarray of shape (num_channels, H/2, W/2)

        Returns
        -------
        grad_input : np.ndarray of shape (num_channels, H, W)

        The math:
        ─────────
        For channel c at pooled position (i,j):
          net = w[c] × sum[c,i,j] + b[c]
          out = f(net)

        δ = grad_output × f'(net)

        ∂E/∂w[c]      = Σ_{i,j} δ[c,i,j] × sum[c,i,j]
        ∂E/∂b[c]      = Σ_{i,j} δ[c,i,j]
        ∂E/∂input[c,r,s] = δ[c,r//2,s//2] × w[c]
                           (each of the 4 inputs in a block gets the same gradient,
                            scaled by w[c], because forward pass summed them equally)
        """
        # δ = grad_output × f'(output)
        delta = grad_output * scaled_tanh_derivative(self._output)

        # Gradient for weights and biases
        for c in range(self.num_channels):
            self.weight_grads[c] = np.sum(delta[c] * self._sums[c])
            self.bias_grads[c] = np.sum(delta[c])

        # Gradient for input: "upsample" δ back to input resolution
        grad_input = np.zeros_like(self._input)
        for c in range(self.num_channels):
            # Each δ[c,i,j] is distributed to the 4 corresponding input pixels
            # and multiplied by w[c] (because forward was: w[c] × sum)
            upsampled = delta[c].repeat(2, axis=0).repeat(2, axis=1)
            grad_input[c] = upsampled * self.weights[c]

        return grad_input

    def get_params(self):
        """Return (param, grad) tuples for SGD."""
        return [
            (self.weights, self.weight_grads),
            (self.biases, self.bias_grads),
        ]

    def param_count(self):
        return 2 * self.num_channels  # one weight + one bias per channel


class FullyConnectedLayer:
    """
    Fully Connected (Dense) Layer — Standard neural network layer.

    THIS is exactly what you learned in Rumelhart et al. (1986)!

    Every input neuron is connected to every output neuron:
        output = f(W × input + b)

    Where:
    - W is a (out_size × in_size) weight matrix
    - b is a (out_size,) bias vector
    - f is the scaled tanh activation

    Parameters
    ----------
    in_size : int
        Number of input neurons
    out_size : int
        Number of output neurons
    """

    def __init__(self, in_size, out_size):
        self.in_size = in_size
        self.out_size = out_size

        # Weight initialization: Uniform(-2.4/fan_in, +2.4/fan_in)
        fan_in = in_size
        limit = 2.4 / fan_in
        self.weights = np.random.uniform(-limit, limit, (out_size, in_size))
        self.biases = np.zeros(out_size)

        # Gradients
        self.weight_grads = np.zeros_like(self.weights)
        self.bias_grads = np.zeros_like(self.biases)

        # Storage
        self._input = None
        self._pre_activation = None
        self._output = None

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : np.ndarray of shape (in_size,) — 1D vector

        Returns
        -------
        np.ndarray of shape (out_size,) — 1D vector
        """
        self._input = x

        # net = W × x + b
        self._pre_activation = self.weights @ x + self.biases

        # output = f(net)
        self._output = scaled_tanh(self._pre_activation)
        return self._output

    def backward(self, grad_output):
        """
        Backward pass — This is exactly from Rumelhart et al. (1986)!

        Parameters
        ----------
        grad_output : np.ndarray of shape (out_size,)

        Returns
        -------
        grad_input : np.ndarray of shape (in_size,)

        The math (you know this!):
        ──────────────────────────
        δ = grad_output × f'(net)

        ∂E/∂W[i,j] = δ[i] × input[j]     → "outer product"
        ∂E/∂b[i]   = δ[i]                  → direct copy
        ∂E/∂input[j] = Σ_i W[i,j] × δ[i]  → "transpose multiply"
        """
        # δ = grad_output × f'(output)
        delta = grad_output * scaled_tanh_derivative(self._output)

        # Weight gradients: outer product of δ and input
        self.weight_grads = np.outer(delta, self._input)

        # Bias gradients: just δ
        self.bias_grads = delta.copy()

        # Input gradient: W^T × δ (pass gradient to previous layer)
        grad_input = self.weights.T @ delta

        return grad_input

    def get_params(self):
        return [
            (self.weights, self.weight_grads),
            (self.biases, self.bias_grads),
        ]

    def param_count(self):
        return self.out_size * self.in_size + self.out_size


class RBFOutputLayer:
    """
    Radial Basis Function Output Layer — The unique output layer of LeNet-5.

    Instead of softmax (modern approach), LeNet-5 computes the EUCLIDEAN
    DISTANCE between the F6 output and fixed target patterns.

    For each class i (digit 0-9):
        y_i = Σ_{j=0}^{83} (x_j - w_ij)²

    Where:
    - x is the 84-dimensional output from F6
    - w_i is the 84-dimensional TARGET PATTERN for class i
    - y_i is the DISTANCE (smaller = better match)

    The target patterns are 7×12 BITMAPS of each digit, flattened to 84 values.
    Values are +1 (foreground) and -1 (background).

    These patterns are FIXED (not learned) — they encode what each digit
    "should look like" in a stylized form.

    The LOSS is simply: E = y_correct_class
    (minimize the distance to the correct target)

    Parameters
    ----------
    in_size : int
        Size of input vector (84 for LeNet-5)
    num_classes : int
        Number of output classes (10 for digits 0-9)
    """

    def __init__(self, in_size, num_classes):
        self.in_size = in_size
        self.num_classes = num_classes

        # Fixed target bitmaps (7×12 = 84 values each, ±1)
        # These are the stylized digit patterns from the paper
        self.targets = self._create_target_bitmaps()

        # Storage
        self._input = None
        self._output = None

    def _create_target_bitmaps(self):
        """
        Create the 7×12 target bitmaps for digits 0-9.

        Each digit is represented as a stylized 7-column × 12-row bitmap.
        +1 = foreground (part of the digit shape)
        -1 = background

        These are approximations of the bitmaps shown in the original paper
        (the paper shows them visually but doesn't list exact values).
        """
        # fmt: off
        bitmaps = {
            0: [
                [-1, +1, +1, +1, +1, +1, -1],
                [+1, +1, -1, -1, -1, +1, +1],
                [+1, -1, -1, -1, -1, -1, +1],
                [+1, -1, -1, -1, -1, -1, +1],
                [+1, -1, -1, -1, -1, -1, +1],
                [+1, -1, -1, -1, -1, -1, +1],
                [+1, -1, -1, -1, -1, -1, +1],
                [+1, -1, -1, -1, -1, -1, +1],
                [+1, +1, -1, -1, -1, +1, +1],
                [-1, +1, +1, +1, +1, +1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
            ],
            1: [
                [-1, -1, -1, +1, -1, -1, -1],
                [-1, -1, +1, +1, -1, -1, -1],
                [-1, -1, -1, +1, -1, -1, -1],
                [-1, -1, -1, +1, -1, -1, -1],
                [-1, -1, -1, +1, -1, -1, -1],
                [-1, -1, -1, +1, -1, -1, -1],
                [-1, -1, -1, +1, -1, -1, -1],
                [-1, -1, -1, +1, -1, -1, -1],
                [-1, -1, -1, +1, -1, -1, -1],
                [-1, +1, +1, +1, +1, +1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
            ],
            2: [
                [-1, +1, +1, +1, +1, +1, -1],
                [+1, +1, -1, -1, -1, +1, +1],
                [-1, -1, -1, -1, -1, -1, +1],
                [-1, -1, -1, -1, -1, +1, +1],
                [-1, -1, -1, -1, +1, +1, -1],
                [-1, -1, -1, +1, +1, -1, -1],
                [-1, -1, +1, +1, -1, -1, -1],
                [-1, +1, +1, -1, -1, -1, -1],
                [+1, +1, -1, -1, -1, -1, -1],
                [+1, +1, +1, +1, +1, +1, +1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
            ],
            3: [
                [-1, +1, +1, +1, +1, +1, -1],
                [+1, +1, -1, -1, -1, +1, +1],
                [-1, -1, -1, -1, -1, -1, +1],
                [-1, -1, -1, -1, -1, +1, +1],
                [-1, -1, +1, +1, +1, +1, -1],
                [-1, -1, -1, -1, -1, +1, +1],
                [-1, -1, -1, -1, -1, -1, +1],
                [-1, -1, -1, -1, -1, -1, +1],
                [+1, +1, -1, -1, -1, +1, +1],
                [-1, +1, +1, +1, +1, +1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
            ],
            4: [
                [-1, -1, -1, -1, +1, +1, -1],
                [-1, -1, -1, +1, +1, +1, -1],
                [-1, -1, +1, +1, +1, +1, -1],
                [-1, +1, +1, -1, +1, +1, -1],
                [+1, +1, -1, -1, +1, +1, -1],
                [+1, +1, +1, +1, +1, +1, +1],
                [-1, -1, -1, -1, +1, +1, -1],
                [-1, -1, -1, -1, +1, +1, -1],
                [-1, -1, -1, -1, +1, +1, -1],
                [-1, -1, -1, -1, +1, +1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
            ],
            5: [
                [+1, +1, +1, +1, +1, +1, +1],
                [+1, +1, -1, -1, -1, -1, -1],
                [+1, +1, -1, -1, -1, -1, -1],
                [+1, +1, +1, +1, +1, +1, -1],
                [-1, -1, -1, -1, -1, +1, +1],
                [-1, -1, -1, -1, -1, -1, +1],
                [-1, -1, -1, -1, -1, -1, +1],
                [-1, -1, -1, -1, -1, +1, +1],
                [+1, +1, -1, -1, -1, +1, +1],
                [-1, +1, +1, +1, +1, +1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
            ],
            6: [
                [-1, -1, +1, +1, +1, +1, -1],
                [-1, +1, +1, -1, -1, -1, -1],
                [+1, +1, -1, -1, -1, -1, -1],
                [+1, +1, +1, +1, +1, +1, -1],
                [+1, +1, -1, -1, -1, +1, +1],
                [+1, -1, -1, -1, -1, -1, +1],
                [+1, -1, -1, -1, -1, -1, +1],
                [+1, +1, -1, -1, -1, +1, +1],
                [-1, +1, +1, -1, -1, +1, +1],
                [-1, -1, +1, +1, +1, +1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
            ],
            7: [
                [+1, +1, +1, +1, +1, +1, +1],
                [-1, -1, -1, -1, -1, +1, +1],
                [-1, -1, -1, -1, +1, +1, -1],
                [-1, -1, -1, -1, +1, +1, -1],
                [-1, -1, -1, +1, +1, -1, -1],
                [-1, -1, -1, +1, +1, -1, -1],
                [-1, -1, +1, +1, -1, -1, -1],
                [-1, -1, +1, +1, -1, -1, -1],
                [-1, -1, +1, +1, -1, -1, -1],
                [-1, -1, +1, +1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
            ],
            8: [
                [-1, +1, +1, +1, +1, +1, -1],
                [+1, +1, -1, -1, -1, +1, +1],
                [+1, -1, -1, -1, -1, -1, +1],
                [+1, +1, -1, -1, -1, +1, +1],
                [-1, +1, +1, +1, +1, +1, -1],
                [+1, +1, -1, -1, -1, +1, +1],
                [+1, -1, -1, -1, -1, -1, +1],
                [+1, -1, -1, -1, -1, -1, +1],
                [+1, +1, -1, -1, -1, +1, +1],
                [-1, +1, +1, +1, +1, +1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
            ],
            9: [
                [-1, +1, +1, +1, +1, +1, -1],
                [+1, +1, -1, -1, -1, +1, +1],
                [+1, -1, -1, -1, -1, -1, +1],
                [+1, +1, -1, -1, -1, +1, +1],
                [-1, +1, +1, +1, +1, +1, +1],
                [-1, -1, -1, -1, -1, -1, +1],
                [-1, -1, -1, -1, -1, -1, +1],
                [-1, -1, -1, -1, -1, +1, +1],
                [-1, -1, -1, -1, +1, +1, -1],
                [-1, +1, +1, +1, +1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
            ],
        }
        # fmt: on

        # Flatten each 12×7 bitmap to an 84-element vector
        targets = np.zeros((10, 84))
        for digit in range(10):
            targets[digit] = np.array(bitmaps[digit]).flatten()
        return targets

    def forward(self, x):
        """
        Forward pass: Compute Euclidean distance to each target.

        Parameters
        ----------
        x : np.ndarray of shape (84,)
            Output from F6 layer

        Returns
        -------
        np.ndarray of shape (10,)
            Distance to each digit's target pattern.
            SMALLER distance = better match.
        """
        self._input = x

        # y_i = Σ_j (x_j - target_i_j)²
        self._output = np.sum((x[np.newaxis, :] - self.targets) ** 2, axis=1)
        return self._output

    def backward(self, label):
        """
        Backward pass: Compute gradient of loss w.r.t. input.

        The loss is: E = y_label (distance to correct class)
        So: ∂E/∂x_j = 2 × (x_j - target[label][j])

        Parameters
        ----------
        label : int
            The correct class (0-9)

        Returns
        -------
        grad_input : np.ndarray of shape (84,)
        loss : float
            The loss value (distance to correct target)
        """
        loss = self._output[label]

        # Gradient of sum((x - target)²) w.r.t. x = 2 * (x - target)
        grad_input = 2.0 * (self._input - self.targets[label])

        return grad_input, loss

    def get_params(self):
        """No trainable parameters — targets are fixed."""
        return []

    def param_count(self):
        return 0  # Targets are fixed, not learned
