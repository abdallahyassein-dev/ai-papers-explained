"""
utils.py — Helper Utilities for LeNet-5

Provides:
  - Training progress display
  - Visualization of learned kernels
  - Confusion matrix
  - ASCII art display of digits
"""

import numpy as np
import time


def print_digit(image, threshold=0.5):
    """
    Display a 32×32 digit image as ASCII art in the terminal.

    Parameters
    ----------
    image : np.ndarray of shape (1, 32, 32) or (32, 32)
        The preprocessed digit image
    threshold : float
        Values above this are shown as filled, below as empty
    """
    if image.ndim == 3:
        image = image[0]  # Remove channel dimension

    for row in image:
        line = ""
        for pixel in row:
            if pixel > threshold:
                line += "██"
            elif pixel > 0.0:
                line += "▒▒"
            else:
                line += "  "
        print(line)


def print_progress_bar(current, total, prefix="", suffix="", length=40):
    """Display a progress bar in the terminal."""
    percent = current / total
    filled = int(length * percent)
    bar = "█" * filled + "░" * (length - filled)
    print(f"\r{prefix} |{bar}| {percent:.1%} {suffix}", end="", flush=True)
    if current == total:
        print()  # New line when done


class TrainingLogger:
    """
    Logs training metrics and displays progress.

    Tracks:
    - Loss per epoch
    - Training accuracy per epoch
    - Test accuracy per epoch
    - Time per epoch
    """

    def __init__(self):
        self.epoch_losses = []
        self.epoch_train_acc = []
        self.epoch_test_acc = []
        self.epoch_times = []
        self._epoch_start = None

    def start_epoch(self, epoch, total_epochs, learning_rate):
        """Mark the beginning of an epoch."""
        self._epoch_start = time.time()
        print(f"\n{'═' * 60}")
        print(f"  Epoch {epoch + 1}/{total_epochs}  |  Learning Rate: {learning_rate}")
        print(f"{'═' * 60}")

    def end_epoch(self, epoch, avg_loss, train_acc, test_acc):
        """Mark the end of an epoch and log metrics."""
        elapsed = time.time() - self._epoch_start
        self.epoch_losses.append(avg_loss)
        self.epoch_train_acc.append(train_acc)
        self.epoch_test_acc.append(test_acc)
        self.epoch_times.append(elapsed)

        print(f"\n  Results:")
        print(f"    Loss:           {avg_loss:.4f}")
        print(f"    Train Accuracy: {train_acc:.2%}")
        print(f"    Test Accuracy:  {test_acc:.2%}  ({(1 - test_acc):.2%} error)")
        print(f"    Time:           {elapsed:.1f}s")

    def print_final_summary(self):
        """Print summary after all training is complete."""
        print(f"\n{'═' * 60}")
        print(f"  TRAINING COMPLETE")
        print(f"{'═' * 60}")

        best_test_idx = np.argmax(self.epoch_test_acc)
        total_time = sum(self.epoch_times)

        print(f"\n  Best Test Accuracy: {self.epoch_test_acc[best_test_idx]:.2%} "
              f"(epoch {best_test_idx + 1})")
        print(f"  Final Test Error:   {(1 - self.epoch_test_acc[-1]):.2%}")
        print(f"  Total Time:         {total_time:.0f}s ({total_time / 60:.1f} minutes)")

        # Print learning curve as ASCII
        print(f"\n  Loss Curve:")
        max_loss = max(self.epoch_losses) if self.epoch_losses else 1
        for i, loss in enumerate(self.epoch_losses):
            bar_len = int(40 * loss / max_loss)
            bar = "█" * bar_len
            print(f"    Epoch {i + 1:2d}: {bar} {loss:.4f}")

        print(f"\n  Accuracy Curve:")
        for i, acc in enumerate(self.epoch_test_acc):
            bar_len = int(40 * acc)
            bar = "█" * bar_len
            print(f"    Epoch {i + 1:2d}: {bar} {acc:.2%}")


def compute_confusion_matrix(model, images, labels, num_classes=10):
    """
    Compute and display a confusion matrix.

    Parameters
    ----------
    model : LeNet5
    images : np.ndarray of shape (N, 1, 32, 32)
    labels : np.ndarray of shape (N,)
    num_classes : int

    Returns
    -------
    np.ndarray of shape (num_classes, num_classes)
        confusion[true][predicted] = count
    """
    confusion = np.zeros((num_classes, num_classes), dtype=int)

    for i in range(len(labels)):
        pred, _ = model.predict(images[i])
        confusion[labels[i]][pred] += 1

    # Display
    print("\nConfusion Matrix:")
    print("       Predicted")
    print("       ", end="")
    for j in range(num_classes):
        print(f"{j:>5}", end="")
    print()
    print("      +" + "─" * 50)

    for i in range(num_classes):
        correct = confusion[i][i]
        total = confusion[i].sum()
        print(f"  {i}   |", end="")
        for j in range(num_classes):
            if i == j:
                print(f" [{confusion[i][j]:3d}]", end="")  # Highlight diagonal
            else:
                print(f"  {confusion[i][j]:3d} ", end="")
        print(f"  | {correct}/{total} ({correct / max(total, 1):.0%})")

    total_correct = np.trace(confusion)
    total = confusion.sum()
    print(f"\n  Overall: {total_correct}/{total} ({total_correct / total:.2%})")

    return confusion


def show_predictions(model, images, labels, n=10):
    """
    Show predictions for a few sample images.

    Parameters
    ----------
    model : LeNet5
    images : np.ndarray of shape (N, 1, 32, 32)
    labels : np.ndarray of shape (N,)
    n : int
        Number of examples to show
    """
    print(f"\nSample Predictions (showing {n} examples):")
    print("─" * 50)

    correct = 0
    for i in range(min(n, len(labels))):
        pred, distances = model.predict(images[i])
        is_correct = pred == labels[i]
        correct += is_correct

        status = "✓" if is_correct else "✗"
        confidence = distances[labels[i]]  # Distance to correct class

        print(f"  [{status}] True: {labels[i]}, Predicted: {pred}, "
              f"Distance: {confidence:.2f}")

        if not is_correct:
            # Show top 2 predictions for wrong cases
            sorted_idx = np.argsort(distances)
            print(f"      Top 2: class {sorted_idx[0]} (dist={distances[sorted_idx[0]]:.2f}), "
                  f"class {sorted_idx[1]} (dist={distances[sorted_idx[1]]:.2f})")

    print(f"\n  {correct}/{min(n, len(labels))} correct")
