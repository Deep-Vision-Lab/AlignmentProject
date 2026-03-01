#!/usr/bin/env python3
"""
Test script for Soft-DTW Loss implementation.
"""

import torch
import torch.nn.functional as F
from LossFunctionWithHelpers import *

def test_soft_dtw_basic():
    """Test basic Soft-DTW functionality."""
    print("Testing basic Soft-DTW loss...")
    
    # Create dummy sequences
    batch_size, seq_len1, seq_len2, dim = 2, 10, 12, 64
    pred = torch.randn(batch_size, seq_len1, dim)
    target = torch.randn(batch_size, seq_len2, dim)
    
    # Test functional interface
    loss = soft_dtw_loss(pred, target, gamma=1.0)
    print(f"Functional loss: {loss.item():.4f}")
    
    # Test class interface
    criterion = SoftDTWLoss(gamma=1.0, distance_type='euclidean')
    loss_class = criterion(pred, target)
    print(f"Class loss: {loss_class.item():.4f}")
    
    # Test alignment matrix
    alignment = criterion.get_alignment_matrix(pred, target)
    print(f"Alignment matrix shape: {alignment.shape}")
    print(f"Alignment matrix sum: {alignment.sum().item():.4f}")
    
    print("✓ Basic Soft-DTW test passed\n")


def test_soft_dtw_distance_types():
    """Test different distance types."""
    print("Testing different distance types...")
    
    batch_size, seq_len, dim = 2, 8, 32
    pred = torch.randn(batch_size, seq_len, dim)
    target = torch.randn(batch_size, seq_len, dim)
    
    for distance_type in ['euclidean', 'cosine', 'dot']:
        criterion = SoftDTWLoss(gamma=1.0, distance_type=distance_type)
        loss = criterion(pred, target)
        print(f"Distance type '{distance_type}': {loss.item():.4f}")
    
    print("✓ Distance types test passed\n")


def test_soft_dtw_alignment_loss():
    """Test combined Soft-DTW alignment loss."""
    print("Testing Soft-DTW alignment loss...")
    
    batch_size, height, width = 2, 15, 20
    final_pred = torch.randn(batch_size, height, width)
    target = torch.randn(batch_size, height, width)
    
    # Text-image similarity matrices
    text_len1, img_len1 = 12, 18
    text_len2, img_len2 = 14, 16
    img_txt_sim1 = torch.randn(batch_size, text_len1, img_len1)
    img_txt_sim2 = torch.randn(batch_size, text_len2, img_len2)
    
    # Test combined loss
    total_loss, loss_dict = soft_dtw_alignment_loss(
        final_pred=final_pred,
        target=target,
        img_txt_sim1=img_txt_sim1,
        img_txt_sim2=img_txt_sim2,
        gamma=1.0,
        mse_weight=1.0,
        dtw_weight=0.5
    )
    
    print(f"Total loss: {total_loss.item():.4f}")
    print("Loss components:", loss_dict)
    print("✓ Soft-DTW alignment loss test passed\n")


def test_similarity_matrix_input():
    """Test Soft-DTW with similarity matrix input."""
    print("Testing Soft-DTW with similarity matrix input...")
    
    batch_size, height, width = 2, 10, 15
    similarity_matrix = torch.randn(batch_size, height, width)
    
    # When input is already a similarity matrix, target is ignored
    criterion = SoftDTWLoss(gamma=1.0, distance_type='dot')
    loss = criterion(similarity_matrix, None)
    print(f"Similarity matrix loss: {loss.item():.4f}")
    
    print("✓ Similarity matrix input test passed\n")


def test_gradient_flow():
    """Test that gradients flow properly through Soft-DTW."""
    print("Testing gradient flow...")
    
    batch_size, seq_len, dim = 1, 8, 16
    pred = torch.randn(batch_size, seq_len, dim, requires_grad=True)
    target = torch.randn(batch_size, seq_len, dim)
    
    criterion = SoftDTWLoss(gamma=1.0)
    loss = criterion(pred, target)
    loss.backward()
    
    assert pred.grad is not None, "Gradient should flow to input"
    grad_norm = pred.grad.norm().item()
    print(f"Gradient norm: {grad_norm:.6f}")
    assert grad_norm > 0, "Gradient should be non-zero"
    
    print("✓ Gradient flow test passed\n")


if __name__ == "__main__":
    print("Running Soft-DTW Loss Tests")
    print("=" * 40)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    try:
        test_soft_dtw_basic()
        test_soft_dtw_distance_types()
        test_soft_dtw_alignment_loss()
        test_similarity_matrix_input()
        test_gradient_flow()
        
        print("🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()