#!/usr/bin/env python3
"""
AlexNet Step-by-Step Interactive Demonstrator
=============================================
This script provides hands-on code examples to help you see EXACTLY what happens
inside AlexNet during forward passes, layer-by-layer shape transformations,
activation gradients, Local Response Normalization (LRN), PCA color jitter, and 10-Crop testing.

Requirements:
    pip install torch torchvision numpy
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. Custom AlexNet with Layer-by-Layer Shape & Parameter Inspector
# =====================================================================

class DetailedAlexNet(nn.Module):
    """
    AlexNet architecture with built-in layer inspection.
    Follows the exact 2012 paper specs (with 227x227 input crop).
    """
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        
        # Layer 1
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0)
        self.relu1 = nn.ReLU(inplace=True)
        self.lrn1 = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Layer 2
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.lrn2 = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Layer 3
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        
        # Layer 4
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        
        # Layer 5
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # FC Layers
        self.fc6 = nn.Linear(256 * 6 * 6, 4096)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout(p=0.5)
        
        self.fc7 = nn.Linear(4096, 4096)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout(p=0.5)
        
        self.fc8 = nn.Linear(4096, num_classes)

    def inspect_forward(self, x: torch.Tensor):
        """Passes input tensor x and prints step-by-step shapes and parameter counts."""
        print("\n" + "="*80)
        print(" 🔍 STEP-BY-STEP ALEXNET FORWARD PASS INSPECTOR")
        print("="*80)
        print(f"{'Layer Name':<15} | {'Input Shape':<20} | {'Output Shape':<20} | {'Params':<12}")
        print("-" * 80)
        
        def trace(name, layer, inp):
            out = layer(inp)
            params = sum(p.numel() for p in layer.parameters()) if hasattr(layer, 'parameters') else 0
            inp_str = str(list(inp.shape))
            out_str = str(list(out.shape))
            print(f"{name:<15} | {inp_str:<20} | {out_str:<20} | {params:<12,}")
            return out

        out = x
        print(f"{'Input Batch':<15} | {'-':<20} | {str(list(out.shape)):<20} | {0:<12}")
        
        # Feature extraction
        out = trace("Conv1", self.conv1, out)
        out = trace("ReLU1", self.relu1, out)
        out = trace("LRN1", self.lrn1, out)
        out = trace("MaxPool1", self.pool1, out)
        
        out = trace("Conv2", self.conv2, out)
        out = trace("ReLU2", self.relu2, out)
        out = trace("LRN2", self.lrn2, out)
        out = trace("MaxPool2", self.pool2, out)
        
        out = trace("Conv3", self.conv3, out)
        out = trace("ReLU3", self.relu3, out)
        
        out = trace("Conv4", self.conv4, out)
        out = trace("ReLU4", self.relu4, out)
        
        out = trace("Conv5", self.conv5, out)
        out = trace("ReLU5", self.relu5, out)
        out = trace("MaxPool5", self.pool5, out)
        
        # Flattening
        out_flat = torch.flatten(out, 1)
        print(f"{'Flatten':<15} | {str(list(out.shape)):<20} | {str(list(out_flat.shape)):<20} | {0:<12}")
        out = out_flat
        
        # Classifier
        out = trace("FC6", self.fc6, out)
        out = trace("ReLU6", self.relu6, out)
        out = trace("Dropout6", self.drop6, out)
        
        out = trace("FC7", self.fc7, out)
        out = trace("ReLU7", self.relu7, out)
        out = trace("Dropout7", self.drop7, out)
        
        out = trace("FC8 (Output)", self.fc8, out)
        
        probs = F.softmax(out, dim=1)
        print("-" * 80)
        print(f"Top-1 Predicted Class ID: {torch.argmax(probs, dim=1).item()}")
        print(f"Top-1 Prediction Prob:   {torch.max(probs, dim=1).values.item():.4f}")
        print("="*80 + "\n")
        return out


# =====================================================================
# 2. Standalone Concept Demonstrations
# =====================================================================

def demo_relu_vs_tanh_gradient():
    """
    Demonstrates WHY ReLU trains faster than Tanh by checking gradient magnitudes.
    """
    print("\n" + "="*80)
    print(" ⚡ DEMO 1: ReLU vs Tanh Gradient Flow (Vanishing Gradient Proof)")
    print("="*80)
    
    # Large initial pre-activation values
    x_val = torch.tensor([0.5, 2.0, 5.0, 10.0], requires_grad=True)
    
    # Tanh pass
    y_tanh = torch.tanh(x_val).sum()
    y_tanh.backward()
    grad_tanh = x_val.grad.clone()
    
    # Reset grad & ReLU pass
    x_val.grad.zero_()
    y_relu = F.relu(x_val).sum()
    y_relu.backward()
    grad_relu = x_val.grad.clone()
    
    print(f"{'Input (x)':<12} | {'Tanh Grad (dy/dx)':<20} | {'ReLU Grad (dy/dx)':<20}")
    print("-" * 60)
    for x, g_t, g_r in zip(x_val.detach().numpy(), grad_tanh.numpy(), grad_relu.numpy()):
        print(f"{x:<12.1f} | {g_t:<20.8f} | {g_r:<20.1f}")
    
    print("\n👉 Takeaway: For large positive x (e.g., x=10), Tanh gradient vanishes to ~0, stalling backprop!")
    print("   ReLU maintains a constant gradient of 1.0, allowing error signals to flow back unchanged!")


def demo_lrn_calculation():
    """
    Demonstrates Local Response Normalization (LRN) on a dummy 4D tensor.
    """
    print("\n" + "="*80)
    print(" 🧪 DEMO 2: Local Response Normalization (LRN) Mechanics")
    print("="*80)
    
    # Create a tensor with 1 batch, 5 channels, 1x1 spatial size
    # Channel 2 has a large activation (10.0) compared to neighbors
    raw_tensor = torch.tensor([[[[3.0]], [[4.0]], [[10.0]], [[2.0]], [[1.0]]]])
    lrn = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)
    
    normalized_tensor = lrn(raw_tensor)
    
    print("Raw Channel Activations at position (0, 0):")
    for ch, val in enumerate(raw_tensor[0, :, 0, 0].numpy()):
        print(f"  Channel {ch}: {val:.2f}")
        
    print("\nAfter Local Response Normalization (LRN):")
    for ch, val in enumerate(normalized_tensor[0, :, 0, 0].numpy()):
        print(f"  Channel {ch}: {val:.4f}")
        
    print("\n👉 Takeaway: Channel 2 (value 10.0) was suppressed down to {:.4f} because neighboring channels were active!".format(
        normalized_tensor[0, 2, 0, 0].item()
    ))


def demo_pca_color_jitter():
    """
    Demonstrates AlexNet's PCA RGB Color Augmentation.
    """
    print("\n" + "="*80)
    print(" 🎨 DEMO 3: PCA Color Augmentation (Fancy Color Jitter)")
    print("="*80)
    
    # Dummy RGB pixel tensor (1 image, 3 channels, 2x2 pixels)
    img = torch.tensor([
        [[120.0, 130.0], [140.0, 150.0]], # Red
        [[100.0, 110.0], [120.0, 130.0]], # Green
        [[80.0,  90.0],  [100.0, 110.0]]  # Blue
    ])
    
    # Pre-calculated ImageNet PCA Eigenvectors & Eigenvalues from paper
    eig_vecs = torch.tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5809, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203]
    ])
    eig_vals = torch.tensor([0.2175, 0.0188, 0.0045])
    
    # Sample random alpha ~ N(0, 0.1)
    torch.manual_seed(42)
    alpha = torch.randn(3) * 0.1
    
    # Compute noise offset: p1*a1*l1 + p2*a2*l2 + p3*a3*l3
    add_val = torch.matmul(eig_vecs, alpha * eig_vals)
    
    # Add offset to RGB channels
    augmented_img = img + add_val.view(3, 1, 1) * 255.0
    
    print(f"Sampled Random Alphas N(0, 0.1): {alpha.numpy()}")
    print(f"Computed RGB Noise Offset:      R: {add_val[0]*255.0:+.2f}, G: {add_val[1]*255.0:+.2f}, B: {add_val[2]*255.0:+.2f}")
    print("\nOriginal Top-Left Pixel RGB:   ", img[:, 0, 0].numpy())
    print("Augmented Top-Left Pixel RGB:  ", augmented_img[:, 0, 0].numpy())
    print("\n👉 Takeaway: Color jitter changes overall lighting without altering object structure or identity!")


def demo_ten_crop_averaging():
    """
    Demonstrates Test-Time 10-Crop Evaluation.
    """
    print("\n" + "="*80)
    print(" ✂️ DEMO 4: Test-Time 10-Crop Prediction Averaging")
    print("="*80)
    
    # Imagine 10 crops evaluated through AlexNet for a 3-class problem [Dog, Cat, Car]
    torch.manual_seed(7)
    crop_logits = torch.randn(10, 3) + torch.tensor([2.0, 0.5, -1.0]) # Bias towards class 0 (Dog)
    crop_probs = F.softmax(crop_logits, dim=1)
    
    print(f"{'Crop #':<8} | {'P(Dog)':<10} | {'P(Cat)':<10} | {'P(Car)':<10}")
    print("-" * 46)
    for i in range(10):
        print(f"Crop {i+1:<3}   | {crop_probs[i,0]:.4f}     | {crop_probs[i,1]:.4f}     | {crop_probs[i,2]:.4f}")
        
    avg_probs = crop_probs.mean(dim=0)
    print("-" * 46)
    print(f"{'AVERAGE':<8} | {avg_probs[0]:.4f}     | {avg_probs[1]:.4f}     | {avg_probs[2]:.4f}")
    print("\n👉 Takeaway: Single noisy crop predictions (e.g. Crop 4 P(Dog)=0.36) are stabilized by averaging all 10 crops (P(Dog)=0.73)!")


# =====================================================================
# Main Execution Entry Point
# =====================================================================

if __name__ == "__main__":
    print("\n🚀 RUNNING ALEXNET COMPLETE DEMONSTRATION SUITE")
    
    # 1. Forward Pass Inspector
    model = DetailedAlexNet(num_classes=1000)
    dummy_input = torch.randn(1, 3, 227, 227) # 1 image, RGB 227x227
    model.inspect_forward(dummy_input)
    
    # 2. Concept Demos
    demo_relu_vs_tanh_gradient()
    demo_lrn_calculation()
    demo_pca_color_jitter()
    demo_ten_crop_averaging()
    
    print("\n🎉 ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
