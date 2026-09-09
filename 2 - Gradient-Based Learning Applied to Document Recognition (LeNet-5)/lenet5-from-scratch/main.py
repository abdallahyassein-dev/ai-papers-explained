#!/usr/bin/env python3
"""
main.py — LeNet-5 From Scratch: Entry Point

This script:
  1. Downloads and preprocesses MNIST
  2. Builds the LeNet-5 architecture
  3. Trains the network using SGD + backpropagation
  4. Evaluates on the test set
  5. Shows sample predictions

Run with:
    python main.py

Or for a quick test with fewer epochs:
    python main.py --epochs 2 --eval-samples 500

═══════════════════════════════════════════════════════════════════════
WHAT YOU'RE ABOUT TO SEE:
═══════════════════════════════════════════════════════════════════════

This code implements EVERYTHING from the paper:
  - Convolutional layers with weight sharing
  - Sub-sampling layers with trainable parameters
  - The C3 partial connection table
  - The RBF output layer with 7×12 bitmap targets
  - Backpropagation through all layer types
  - SGD with the paper's learning rate schedule

No PyTorch. No TensorFlow. Just NumPy and math.
The same math from Rumelhart et al. (1986) that you already know.

═══════════════════════════════════════════════════════════════════════
"""

import argparse
import sys
import os
import time

import numpy as np

# Add the project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mnist_loader import load_mnist
from lenet5 import LeNet5
from train import train, evaluate
from utils import show_predictions, print_digit, compute_confusion_matrix


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LeNet-5 from scratch — Train on MNIST"
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs (default: 10, paper uses 20)"
    )
    parser.add_argument(
        "--eval-samples", type=int, default=1000,
        help="Number of test samples per epoch evaluation (default: 1000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--data-dir", type=str, default="./data",
        help="Directory to store MNIST data"
    )
    parser.add_argument(
        "--show-samples", type=int, default=20,
        help="Number of sample predictions to show after training"
    )
    parser.add_argument(
        "--confusion", action="store_true",
        help="Show confusion matrix after training (slow: evaluates full test set)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.seed)

    print("╔═══════════════════════════════════════════════════════════╗")
    print("║         LeNet-5 From Scratch (LeCun et al., 1998)        ║")
    print("║                                                          ║")
    print("║  Implementing 'Gradient-Based Learning Applied to        ║")
    print("║  Document Recognition' using only Python + NumPy         ║")
    print("║                                                          ║")
    print("║  Every convolution, every gradient, every weight update  ║")
    print("║  is computed by hand — nothing hidden behind libraries.  ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print()

    # ═══════════════════════════════════════════════════════════════
    # Step 1: Load MNIST
    # ═══════════════════════════════════════════════════════════════
    print("Step 1: Loading MNIST dataset")
    print("─" * 40)
    train_images, train_labels, test_images, test_labels = load_mnist(args.data_dir)

    # Show a sample digit
    print("Sample training digit (label = {}):".format(train_labels[0]))
    print_digit(train_images[0])
    print()

    # ═══════════════════════════════════════════════════════════════
    # Step 2: Build LeNet-5
    # ═══════════════════════════════════════════════════════════════
    print("Step 2: Building LeNet-5 architecture")
    print("─" * 40)
    model = LeNet5()

    # Quick sanity check: forward pass one image
    print("Sanity check: Forward pass on one image...")
    test_output = model.forward(train_images[0])
    print(f"  Output shape: {test_output.shape} (should be (10,))")
    print(f"  Output values: {test_output}")
    print(f"  Predicted class: {np.argmin(test_output)} (random at this point)")
    print(f"  True class: {train_labels[0]}")
    print()

    # ═══════════════════════════════════════════════════════════════
    # Step 3: Train
    # ═══════════════════════════════════════════════════════════════
    print("Step 3: Training with SGD + Backpropagation")
    print("─" * 40)

    start_time = time.time()
    logger = train(
        model,
        train_images, train_labels,
        test_images, test_labels,
        num_epochs=args.epochs,
        eval_samples=args.eval_samples,
    )
    total_time = time.time() - start_time

    # ═══════════════════════════════════════════════════════════════
    # Step 4: Show results
    # ═══════════════════════════════════════════════════════════════
    print("\n\nStep 4: Results")
    print("─" * 40)

    # Show sample predictions
    show_predictions(model, test_images, test_labels, n=args.show_samples)

    # Optional: confusion matrix
    if args.confusion:
        print("\nComputing confusion matrix on full test set...")
        compute_confusion_matrix(model, test_images, test_labels)

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 60}")
    print(f"  Total training time: {total_time:.0f}s ({total_time / 60:.1f} minutes)")
    print(f"  Paper's reported error rate: 0.95%")
    print(f"  Our error rate: {(1 - logger.epoch_test_acc[-1]):.2%}")
    print(f"{'═' * 60}")
    print()
    print("  Note: Pure NumPy is ~100x slower than PyTorch/GPU.")
    print("  The paper's results were achieved with 20 epochs of training.")
    print("  With more epochs and the full dataset, accuracy improves.")
    print()
    print("  Congratulations! You've just implemented a pioneering")
    print("  deep learning paper entirely from scratch! 🎉")


if __name__ == "__main__":
    main()
