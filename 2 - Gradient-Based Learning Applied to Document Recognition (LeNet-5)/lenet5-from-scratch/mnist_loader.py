"""
mnist_loader.py — Download and Preprocess MNIST for LeNet-5

The MNIST dataset consists of:
  - 60,000 training images of handwritten digits (0-9)
  - 10,000 test images
  - Each image is 28×28 pixels, grayscale (0-255)

For LeNet-5, we need to:
  1. Pad 28×28 → 32×32 (so edge features can be centered in 5×5 kernels)
  2. Normalize pixel values:
     - Background (white, originally 0) → -0.1
     - Foreground (black, originally 255) → 1.175
     - This makes mean ≈ 0 and variance ≈ 1, which helps gradient descent

The IDX file format used by MNIST:
  - Magic number (4 bytes): identifies the data type
  - Dimensions (4 bytes each): number of items, rows, cols
  - Data (1 byte per pixel): unsigned integers 0-255
"""

import gzip
import os
import struct
import urllib.request

import numpy as np

# MNIST download URLs (from Yann LeCun's website mirror)
MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}

# LeNet-5 specific constants for normalization
BACKGROUND_VALUE = -0.1    # What white pixels become
FOREGROUND_VALUE = 1.175   # What black pixels become
PADDED_SIZE = 32           # LeNet-5 expects 32×32 input


def _download_file(url, filepath):
    """Download a file from a URL if it doesn't already exist."""
    if os.path.exists(filepath):
        return
    print(f"  Downloading {os.path.basename(filepath)}...")
    urllib.request.urlretrieve(url, filepath)


def _parse_idx_images(filepath):
    """
    Parse an IDX file containing images.

    IDX format for images:
      Byte 0-3:   Magic number (2051 for images)
      Byte 4-7:   Number of images
      Byte 8-11:  Number of rows (28)
      Byte 12-15: Number of columns (28)
      Byte 16+:   Pixel data (unsigned bytes, row-major)

    Returns
    -------
    np.ndarray of shape (N, 28, 28), dtype float64, values in [0, 255]
    """
    with gzip.open(filepath, "rb") as f:
        # Read header
        magic = struct.unpack(">I", f.read(4))[0]
        assert magic == 2051, f"Invalid magic number: {magic} (expected 2051 for images)"

        num_images = struct.unpack(">I", f.read(4))[0]
        num_rows = struct.unpack(">I", f.read(4))[0]
        num_cols = struct.unpack(">I", f.read(4))[0]

        # Read pixel data
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num_images, num_rows, num_cols).astype(np.float64)

    return data


def _parse_idx_labels(filepath):
    """
    Parse an IDX file containing labels.

    IDX format for labels:
      Byte 0-3:   Magic number (2049 for labels)
      Byte 4-7:   Number of labels
      Byte 8+:    Label data (unsigned bytes, each 0-9)

    Returns
    -------
    np.ndarray of shape (N,), dtype int, values in [0, 9]
    """
    with gzip.open(filepath, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        assert magic == 2049, f"Invalid magic number: {magic} (expected 2049 for labels)"

        num_labels = struct.unpack(">I", f.read(4))[0]
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels.astype(np.int64)


def _preprocess_images(images):
    """
    Preprocess MNIST images for LeNet-5.

    Steps:
      1. Normalize from [0, 255] to [BACKGROUND_VALUE, FOREGROUND_VALUE]
         - 0 (white/background) → -0.1
         - 255 (black/foreground) → 1.175
         Formula: pixel_new = pixel_old / 255.0 * (1.175 - (-0.1)) + (-0.1)
                            = pixel_old / 255.0 * 1.275 - 0.1

      2. Pad from 28×28 to 32×32 (2 pixels on each side)
         - Padding uses BACKGROUND_VALUE (-0.1)

    Parameters
    ----------
    images : np.ndarray of shape (N, 28, 28)

    Returns
    -------
    np.ndarray of shape (N, 1, 32, 32) — (batch, channels, height, width)
    The channel dimension (1) is added because LeNet-5 layers expect
    (num_feature_maps, height, width) format.
    """
    n = images.shape[0]

    # Step 1: Normalize pixel values
    #   Original: 0 = white (background), 255 = black (ink)
    #   Target:  -0.1 = background, 1.175 = ink
    normalized = images / 255.0 * (FOREGROUND_VALUE - BACKGROUND_VALUE) + BACKGROUND_VALUE

    # Step 2: Pad 28×28 → 32×32 with background value
    padded = np.full((n, PADDED_SIZE, PADDED_SIZE), BACKGROUND_VALUE, dtype=np.float64)
    # Center the 28×28 image in the 32×32 frame
    offset = (PADDED_SIZE - 28) // 2  # = 2
    padded[:, offset:offset + 28, offset:offset + 28] = normalized

    # Step 3: Add channel dimension → (N, 1, 32, 32)
    padded = padded[:, np.newaxis, :, :]

    return padded


def load_mnist(data_dir="./data"):
    """
    Download (if needed) and load the MNIST dataset, preprocessed for LeNet-5.

    Parameters
    ----------
    data_dir : str
        Directory to store downloaded MNIST files.

    Returns
    -------
    train_images : np.ndarray of shape (60000, 1, 32, 32)
    train_labels : np.ndarray of shape (60000,)
    test_images  : np.ndarray of shape (10000, 1, 32, 32)
    test_labels  : np.ndarray of shape (10000,)

    Example
    -------
    >>> train_images, train_labels, test_images, test_labels = load_mnist()
    >>> print(train_images.shape)  # (60000, 1, 32, 32)
    >>> print(train_labels[:5])    # e.g., [5, 0, 4, 1, 9]
    """
    os.makedirs(data_dir, exist_ok=True)

    # Download all files
    print("Loading MNIST dataset...")
    filenames = {}
    for key, url in MNIST_URLS.items():
        filename = os.path.join(data_dir, os.path.basename(url))
        _download_file(url, filename)
        filenames[key] = filename

    # Parse files
    train_images = _parse_idx_images(filenames["train_images"])
    train_labels = _parse_idx_labels(filenames["train_labels"])
    test_images = _parse_idx_images(filenames["test_images"])
    test_labels = _parse_idx_labels(filenames["test_labels"])

    print(f"  Raw shapes: train={train_images.shape}, test={test_images.shape}")

    # Preprocess
    train_images = _preprocess_images(train_images)
    test_images = _preprocess_images(test_images)

    print(f"  Preprocessed shapes: train={train_images.shape}, test={test_images.shape}")
    print(f"  Pixel value range: [{train_images.min():.3f}, {train_images.max():.3f}]")
    print(f"  Labels: {np.unique(train_labels)}")
    print("  Done!\n")

    return train_images, train_labels, test_images, test_labels
