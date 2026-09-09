"""
train.py — Training Loop for LeNet-5

Implements the training procedure from the paper:

  1. Stochastic Gradient Descent (SGD) — update weights after EACH example
  2. Learning rate schedule — decrease η over epochs
  3. Random shuffling — shuffle training data each epoch
  4. Periodic evaluation — check test accuracy after each epoch

The training procedure:
─────────────────────
  For each epoch:
    Shuffle training data
    For each training image:
      1. Forward pass: compute output
      2. Compute loss (RBF distance to correct class)
      3. Backward pass: compute gradients (backpropagation!)
      4. Update weights: w -= η × gradient
    Evaluate on test set

This is pure SGD — the simplest form of gradient descent.
No momentum, no Adam, no batch normalization — just what
was available in 1998 and what the paper describes.
"""

import numpy as np
from utils import TrainingLogger, print_progress_bar


# ═══════════════════════════════════════════════════════════════════════
# LEARNING RATE SCHEDULE (from the paper)
# ═══════════════════════════════════════════════════════════════════════
#
# The paper uses a decreasing learning rate:
#   - Start high for fast initial learning
#   - Decrease gradually for fine-tuning
#
# Think of it like parking a car:
#   - Drive fast toward the spot (high LR)
#   - Slow down as you approach (medium LR)
#   - Inch forward to align (low LR)

LEARNING_RATE_SCHEDULE = {
    0: 0.0005,    # Epochs 1-2:  large steps
    2: 0.0002,    # Epochs 3-5:  medium steps
    5: 0.0001,    # Epochs 6-8:  smaller steps
    8: 0.00005,   # Epochs 9+:   fine-tuning
}


def get_learning_rate(epoch):
    """
    Get the learning rate for the given epoch.

    Uses the schedule from the paper — learning rate decreases
    as training progresses to allow finer adjustments.
    """
    lr = 0.0005  # default
    for start_epoch, rate in sorted(LEARNING_RATE_SCHEDULE.items()):
        if epoch >= start_epoch:
            lr = rate
    return lr


def evaluate(model, images, labels, max_samples=None):
    """
    Evaluate model accuracy on a dataset.

    Parameters
    ----------
    model : LeNet5
    images : np.ndarray of shape (N, 1, 32, 32)
    labels : np.ndarray of shape (N,)
    max_samples : int or None
        If set, only evaluate on this many samples (for speed)

    Returns
    -------
    accuracy : float
        Fraction of correctly classified images
    avg_loss : float
        Average loss over the dataset
    """
    n = len(labels)
    if max_samples is not None:
        n = min(n, max_samples)

    correct = 0
    total_loss = 0.0

    for i in range(n):
        pred, distances = model.predict(images[i])
        if pred == labels[i]:
            correct += 1
        total_loss += distances[labels[i]]

    accuracy = correct / n
    avg_loss = total_loss / n
    return accuracy, avg_loss


def train(model, train_images, train_labels, test_images, test_labels,
          num_epochs=20, eval_samples=1000):
    """
    Train LeNet-5 using SGD with the paper's learning rate schedule.

    Parameters
    ----------
    model : LeNet5
        The model to train
    train_images : np.ndarray of shape (60000, 1, 32, 32)
    train_labels : np.ndarray of shape (60000,)
    test_images : np.ndarray of shape (10000, 1, 32, 32)
    test_labels : np.ndarray of shape (10000,)
    num_epochs : int
        Number of training epochs
    eval_samples : int
        Number of test samples to use for evaluation each epoch
        (full test set evaluation is slow; use subset for progress tracking)

    The Training Loop Explained:
    ────────────────────────────
    This implements the exact procedure from Rumelhart et al. (1986)
    applied to the LeNet-5 architecture:

    1. FORWARD PASS: Feed image through all layers
       Input → C1 → S2 → C3 → S4 → C5 → F6 → RBF → distances

    2. LOSS: The distance to the correct class's target pattern
       Loss = distance[correct_class]

    3. BACKWARD PASS: Backpropagate the error through all layers
       Compute ∂Loss/∂w for every weight in every layer
       (This is the chain rule applied repeatedly — the backprop paper!)

    4. UPDATE: Adjust each weight to reduce the loss
       w = w - η × ∂Loss/∂w
    """
    logger = TrainingLogger()
    n_train = len(train_labels)

    print(f"\nStarting training: {num_epochs} epochs, {n_train} training images")
    print(f"Evaluating on {eval_samples} test images per epoch")

    for epoch in range(num_epochs):
        lr = get_learning_rate(epoch)
        logger.start_epoch(epoch, num_epochs, lr)

        # ── Step 0: Shuffle training data ──────────────────────────
        # Shuffling ensures the network sees examples in random order
        # each epoch. This prevents the network from learning patterns
        # in the ordering of the training set.
        indices = np.random.permutation(n_train)

        # ── Step 1-4: Train on each example ────────────────────────
        epoch_loss = 0.0
        epoch_correct = 0

        for step, idx in enumerate(indices):
            image = train_images[idx]   # shape: (1, 32, 32)
            label = train_labels[idx]   # int: 0-9

            # FORWARD: Compute output (RBF distances)
            output = model.forward(image)

            # CHECK: Is the prediction correct?
            predicted = np.argmin(output)
            if predicted == label:
                epoch_correct += 1

            # BACKWARD: Compute gradients (backpropagation!)
            loss = model.backward(label)
            epoch_loss += loss

            # UPDATE: Adjust weights (SGD)
            model.update_weights(lr)

            # Print progress every 1000 steps
            if (step + 1) % 1000 == 0:
                running_acc = epoch_correct / (step + 1)
                running_loss = epoch_loss / (step + 1)
                print_progress_bar(
                    step + 1, n_train,
                    prefix="  Training",
                    suffix=f"loss={running_loss:.3f} acc={running_acc:.2%}"
                )

        # ── Step 5: Evaluate on test set ───────────────────────────
        avg_loss = epoch_loss / n_train
        train_acc = epoch_correct / n_train
        test_acc, _ = evaluate(model, test_images, test_labels, max_samples=eval_samples)

        logger.end_epoch(epoch, avg_loss, train_acc, test_acc)

    # Final full evaluation
    print("\n\nRunning final evaluation on FULL test set (10,000 images)...")
    final_acc, final_loss = evaluate(model, test_images, test_labels)
    print(f"  Final Test Accuracy: {final_acc:.2%}")
    print(f"  Final Test Error:    {(1 - final_acc):.2%}")

    logger.print_final_summary()

    return logger
