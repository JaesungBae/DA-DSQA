"""
Loss functions for speech quality assessment models.

This module provides various loss functions used in training regression probes
for speech quality prediction, including robust regression losses and regularization terms.
"""

import torch
import numpy as np


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 0.5) -> torch.Tensor:
    """
    Compute Huber loss for robust regression.
    
    Huber loss is less sensitive to outliers than MSE:
    - For |residual| <= delta: L2 loss (0.5 * r^2)
    - For |residual| > delta: L1 loss (delta * |r| - 0.5 * delta^2)
    
    This makes it more robust to outliers while maintaining smooth gradients
    for small errors.
    
    Args:
        pred: Predicted values of shape [B] or [B, 1]
        target: Target values of shape [B] or [B, 1]
        delta: Threshold parameter controlling the transition from L2 to L1 (default: 0.5)
    
    Returns:
        loss: Scalar tensor containing the mean Huber loss
    
    Example:
        >>> pred = torch.tensor([1.0, 2.0, 3.0])
        >>> target = torch.tensor([1.1, 2.2, 2.8])
        >>> loss = huber_loss(pred, target, delta=0.5)
    """
    residual = pred - target
    abs_residual = residual.abs()
    quadratic = torch.clamp(abs_residual, max=delta)
    return (0.5 * quadratic**2 + delta * (abs_residual - quadratic)).mean()


def compute_speaker_level_variance_loss(
    preds: torch.Tensor, speaker_ids: list
) -> torch.Tensor:
    """
    Compute speaker-level variance loss for regularization.
    
    This loss encourages predictions from the same speaker to be consistent
    by penalizing the variance of predictions within each speaker group.
    It is used as a regularization term to improve speaker-level consistency
    while maintaining utterance-level accuracy.
    
    Args:
        preds: Tensor of predictions of shape [B]
        speaker_ids: List of speaker IDs (length B). Each element should be
                    a hashable identifier (e.g., string or int)
    
    Returns:
        variance_loss: Scalar tensor containing the mean variance across speakers.
                      Returns 0.0 if no speakers have multiple utterances in the batch.
    
    Example:
        >>> preds = torch.tensor([1.0, 1.1, 2.0, 2.2])
        >>> speaker_ids = ["spk1", "spk1", "spk2", "spk2"]
        >>> loss = compute_speaker_level_variance_loss(preds, speaker_ids)
    """
    if len(set(speaker_ids)) == len(speaker_ids):
        # If all speakers are unique in this batch, return zero loss
        return torch.tensor(0.0, device=preds.device)
    
    # Group predictions by speaker
    speaker_to_preds = {}
    for i, sid in enumerate(speaker_ids):
        if sid not in speaker_to_preds:
            speaker_to_preds[sid] = []
        speaker_to_preds[sid].append(preds[i])
    
    # Compute variance for each speaker with multiple utterances
    variances = []
    for sid, pred_list in speaker_to_preds.items():
        if len(pred_list) > 1:
            pred_tensor = torch.stack(pred_list)
            var = pred_tensor.var()
            variances.append(var)
    
    if len(variances) == 0:
        return torch.tensor(0.0, device=preds.device)
    
    # Return mean variance across all speakers (we want to minimize this)
    return torch.stack(variances).mean()


def compute_sample_weights(labels: np.ndarray, num_classes: int) -> tuple:
    """
    Compute class weights and sample weights for balanced sampling.
    
    This function computes inverse frequency weights to balance class distribution
    in the dataset. It is commonly used with WeightedRandomSampler to ensure
    balanced training.
    
    Args:
        labels: Array of integer labels (0 to num_classes-1)
        num_classes: Number of classes in the dataset
    
    Returns:
        class_weights: Array of weights for each class (normalized)
        sample_weights: Array of weights for each sample
    
    Example:
        >>> labels = np.array([0, 0, 1, 1, 1, 2])
        >>> class_weights, sample_weights = compute_sample_weights(labels, num_classes=3)
    """
    counts = np.bincount(labels, minlength=num_classes)
    class_weights = 1.0 / np.maximum(counts, 1)
    class_weights = class_weights / class_weights.mean()
    sample_weights = [float(class_weights[y]) for y in labels]
    return class_weights, sample_weights


