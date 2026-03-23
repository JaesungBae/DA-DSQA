"""
Contrastive Pre-training for pre_net Layer

Pre-trains the shared backbone (LayerNorm + pre_net Linear(1280, 256)) using
a contrastive learning objective. The learned representations capture severity
as distance from a "typical speech" centroid, without requiring severity labels.

Loss = w_contrast * L_simclr + w_var * L_vicreg_var + w_anchor * L_anchor_dist

After pre-training, saves norm + pre_net weights for loading into MultiTaskModel.

Both atypical and typical speech use pre-extracted .npy features.

Usage:
    python pretrain_contrastive.py \
        --atypical_feature_dir /path/to/whisper_features \
        --atypical_data_dir ./dataset_labeled \
        --typical_data_dir ./dataset_typical \
        --epochs 100 --batch_size 256 --lr 1e-3
"""

import os
import json
import argparse
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ============================================================================
# DDP Utilities
# ============================================================================

def is_ddp():
    """Check if running under torchrun / distributed launch."""
    # torchrun automatically sets RANK and WORLD_SIZE env vars;
    # their presence signals that we are in a multi-GPU DDP run.
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ

def setup_ddp():
    """Initialize DDP process group using torchrun env vars."""
    # NCCL backend is the standard choice for GPU-to-GPU communication on CUDA.
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()           # global rank across all nodes
    local_rank = int(os.environ["LOCAL_RANK"])  # rank within this node (maps to GPU index)
    world_size = dist.get_world_size()           # total number of processes
    torch.cuda.set_device(local_rank)  # bind this process to its assigned GPU
    return rank, local_rank, world_size

def cleanup_ddp():
    """Destroy DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process(rank):
    """Check if this is the main (rank 0) process."""
    # Only rank 0 should print logs, save checkpoints, and run visualization
    # to avoid duplicate work and file-write conflicts.
    return rank == 0


# ============================================================================
# Feature-Space Augmentations
# ============================================================================

def random_time_mask(features, max_mask_ratio=0.2):
    """Zero out a random contiguous block of time frames."""
    T, D = features.shape
    # Choose a random block length up to max_mask_ratio of the sequence
    mask_len = random.randint(1, max(1, int(T * max_mask_ratio)))
    start = random.randint(0, max(0, T - mask_len))
    features = features.clone()  # avoid mutating the original tensor in-place
    features[start:start + mask_len] = 0.0
    return features


def gaussian_noise(features, std=0.01):
    """Add small Gaussian noise to feature dimensions."""
    # Simulates microphone/recording noise; std is kept small so the
    # semantic content of the features is preserved.
    return features + torch.randn_like(features) * std


def random_crop(features, min_ratio=0.7):
    """Randomly crop a subsequence and pad back to original length."""
    T, D = features.shape
    # Crop at least min_ratio of the original length to retain most content
    crop_len = random.randint(max(1, int(T * min_ratio)), T)
    start = random.randint(0, T - crop_len)
    cropped = features[start:start + crop_len]
    # Pad back to original length with zeros so that the collate function
    # sees a consistent shape before the final batch padding step.
    if crop_len < T:
        pad = torch.zeros(T - crop_len, D)
        cropped = torch.cat([cropped, pad], dim=0)
    return cropped


def augment(features, noise_std=0.01, max_mask_ratio=0.2, crop_min_ratio=0.7):
    """Apply a random combination of augmentations to create one view.

    The two views produced per sample (view1, view2) must be "different enough"
    for the contrastive objective to be non-trivial, yet similar enough for the
    model to learn meaningful representations rather than trivial invariances.
    Gaussian noise is always applied; time masking and cropping are applied
    with 50% probability each, giving four possible augmentation combinations.
    """
    # Gaussian noise is always applied as the base perturbation
    out = gaussian_noise(features, std=noise_std)
    # Randomly apply time masking (simulates partial occlusion / VAD gaps)
    if random.random() < 0.5:
        out = random_time_mask(out, max_mask_ratio=max_mask_ratio)
    # Randomly apply cropping (simulates utterance truncation)
    if random.random() < 0.5:
        out = random_crop(out, min_ratio=crop_min_ratio)
    return out


# ============================================================================
# Dataset
# ============================================================================

class ContrastiveDataset(Dataset):
    """
    Dataset for contrastive pre-training.

    Loads pre-extracted .npy features from separate typical and atypical sources.
    Returns two augmented views per sample plus an is_typical flag.
    """

    def __init__(self, feature_dir, metadata_path, is_typical, noise_std=0.01,
                 max_mask_ratio=0.2, crop_min_ratio=0.7,
                 label12_as_typical=False, label12_target="Intelligibility",
                 label_typical_threshold=2.0,
                 label_supcon_target="Average", typical_supcon_group=-1):
        """
        Args:
            feature_dir: Directory containing .npy feature files
            metadata_path: Path to JSON metadata
            is_typical: Whether this dataset contains typical speech (True) or atypical (False)
            noise_std: Gaussian noise std for augmentation
            max_mask_ratio: Max ratio of time frames to mask
            crop_min_ratio: Minimum crop ratio for random cropping
            label12_as_typical: If True, treat atypical samples with rating <= threshold as typical
            label12_target: Rating key to check for label12_as_typical (e.g. "Intelligibility", "Naturalness")
            label_typical_threshold: Samples with rating <= this value are treated as typical
            label_supcon_target: Rating key to use for fine-grained label supcon loss
            typical_supcon_group: Group index assigned to typical speech for label supcon (-1 = exclude)
        """
        self.noise_std = noise_std
        self.max_mask_ratio = max_mask_ratio
        self.crop_min_ratio = crop_min_ratio
        self.is_typical = is_typical

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        self.samples = []  # list of (feat_path, is_typical_flag, rating_value)
        num_relabeled = 0

        for filename, info in metadata.items():
            # Keys starting with "_" are metadata fields, not filenames
            if filename.startswith("_"):
                continue

            # Feature files are .npy with the same stem as the audio filename
            feat_name = os.path.splitext(filename)[0] + ".npy"
            feat_path = os.path.join(feature_dir, feat_name)

            # Default: inherit the dataset-level typical flag
            sample_typical = is_typical
            # NaN rating means "no label available" → excluded from label_supcon loss
            rating_value = float('nan')

            if isinstance(info, dict):
                ratings = info.get("ratings", {})
                # Read the rating used for the fine-grained label supcon loss
                rv = ratings.get(label_supcon_target, None)
                if rv is not None:
                    rating_value = float(rv)

                # label12_as_typical: re-label very mild atypical samples as typical
                # so the contrastive loss treats them as "near-normal" anchors.
                if not is_typical and label12_as_typical:
                    rv12 = ratings.get(label12_target, None)
                    if rv12 is not None and float(rv12) <= label_typical_threshold:
                        sample_typical = True
                        num_relabeled += 1

            if is_typical:
                # For true typical speech, assign a representative rating value that maps
                # back to the intended group index via rating_to_group.
                # With default boundaries (1.5, 2.5, ..., 6.5), adding 1.0 to the group
                # index produces a rating at the centre of each integer bin:
                #   group 0 → rating 1.0 → rating_to_group = 0  (SAP scale minimum, most normal)
                #   group 1 → rating 2.0 → rating_to_group = 1
                #   group k → rating k+1 → rating_to_group = k
                # Using float(group) alone was wrong: group 1 → 1.0 → rating_to_group = 0.
                # typical_supcon_group=-1 excludes typical speech from label_supcon entirely.
                rating_value = float(typical_supcon_group) + 1.0 if typical_supcon_group >= 0 else float('nan')

            self.samples.append((feat_path, sample_typical, rating_value))

        label = "typical" if is_typical else "atypical"
        print(f"Loaded {len(self.samples)} {label} samples from {metadata_path}")
        if num_relabeled > 0:
            print(f"  -> {num_relabeled} samples with {label12_target}<={label_typical_threshold} relabeled as typical")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat_path, sample_typical, rating_value = self.samples[idx]
        features = torch.from_numpy(np.load(feat_path)).float()

        # Create two augmented views
        view1 = augment(features, self.noise_std, self.max_mask_ratio, self.crop_min_ratio)
        view2 = augment(features, self.noise_std, self.max_mask_ratio, self.crop_min_ratio)

        return view1, view2, sample_typical, rating_value




def contrastive_collate_fn(batch):
    """Collate variable-length sequences with padding for two views.

    Each sample produces two augmented views with potentially different lengths
    (because random_crop can shorten a view). We pad all views to the max length
    across *both* view sets so the model receives a uniform tensor, then pass
    actual lengths for masked mean pooling inside the model.
    """
    views1, views2, is_typical, ratings = zip(*batch)

    # Determine the global max length across both view sets to ensure consistent padding
    all_views = list(views1) + list(views2)
    max_len = max(v.shape[0] for v in all_views)
    hidden_dim = all_views[0].shape[1]
    batch_size = len(views1)

    # Zero-initialize padded tensors; padding frames stay 0 and are masked out during pooling
    padded1 = torch.zeros(batch_size, max_len, hidden_dim)
    padded2 = torch.zeros(batch_size, max_len, hidden_dim)
    lengths1 = torch.zeros(batch_size, dtype=torch.long)
    lengths2 = torch.zeros(batch_size, dtype=torch.long)

    for i in range(batch_size):
        l1 = views1[i].shape[0]
        padded1[i, :l1] = views1[i]
        lengths1[i] = l1

        l2 = views2[i].shape[0]
        padded2[i, :l2] = views2[i]
        lengths2[i] = l2

    is_typical = torch.tensor(is_typical, dtype=torch.bool)
    ratings = torch.tensor(ratings, dtype=torch.float)  # NaN entries excluded in label_supcon

    return padded1, lengths1, padded2, lengths2, is_typical, ratings


# ============================================================================
# Model
# ============================================================================

class ContrastiveModel(nn.Module):
    """
    Contrastive pre-training model.

    LayerNorm → pre_net (Linear) → ReLU → Mean pooling → Projection head
    """

    def __init__(self, input_dim=1280, hidden_dim=256, proj_dim=128):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.pre_net = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()

        # Projection head (discarded after pre-training)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch, time, input_dim)
            lengths: (batch,)
        Returns:
            representations: (batch, hidden_dim) — from pre_net
            projections: (batch, proj_dim) — from projection head
        """
        x = self.norm(x)               # normalize across the feature dimension per frame
        x = self.relu(self.pre_net(x)) # project 1280 → hidden_dim, apply non-linearity

        # Masked mean pooling: ignore zero-padded frames beyond each sample's true length
        if lengths is not None:
            batch_size, max_len, _ = x.shape
            # mask[b, t] = True if frame t is real (not padding) for sample b
            mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            x = x * mask.unsqueeze(-1).float()
            # Divide by the true length (not max_len) to get an unbiased mean
            representations = x.sum(dim=1) / lengths.unsqueeze(1).float().clamp(min=1)
        else:
            representations = x.mean(dim=1)

        # Projection head maps representations to a lower-dim space where the
        # contrastive loss is applied. The projection head is discarded after
        # pre-training; only norm + pre_net weights are transferred.
        projections = self.projection(representations)
        return representations, projections


class ContrastiveModelV2(nn.Module):
    """
    Contrastive pre-training model v2.

    LayerNorm → pre_net1 (Linear) → ReLU → pre_net2 (Linear) → ReLU
    → Statistics pooling (mean+std) → Projection head
    """

    def __init__(self, input_dim=1280, hidden_dim=256, proj_dim=128, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.pre_net = nn.Linear(input_dim, hidden_dim)
        self.pre_net2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Projection head input is 2*hidden_dim due to mean+std concat
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch, time, input_dim)
            lengths: (batch,)
        Returns:
            representations: (batch, hidden_dim*2) — statistics pooling output
            projections: (batch, proj_dim) — from projection head
        """
        x = self.norm(x)
        x = self.dropout(self.relu(self.pre_net(x)))    # 1280 → hidden_dim
        x = self.dropout(self.relu(self.pre_net2(x)))   # hidden_dim → hidden_dim (deeper encoding)

        # Statistics pooling: concatenate per-dimension mean and std across time.
        # The std component captures speaking-rate and prosodic variability, which
        # are informative signals for severity beyond what the mean alone captures.
        if lengths is not None:
            batch_size, max_len, _ = x.shape
            mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            mask_f = mask.unsqueeze(-1).float()
            x_masked = x * mask_f
            lengths_f = lengths.unsqueeze(1).float().clamp(min=1)
            mean = x_masked.sum(dim=1) / lengths_f
            # Biased variance computed from masked frames only:
            # Var = E[x²] - (E[x])²
            var = (x_masked ** 2).sum(dim=1) / lengths_f - mean ** 2
            std = var.clamp(min=1e-8).sqrt()  # clamp avoids sqrt(0) NaN
        else:
            mean = x.mean(dim=1)
            std = x.std(dim=1)

        # Concatenate mean and std → (batch, hidden_dim*2)
        representations = torch.cat([mean, std], dim=1)

        projections = self.projection(representations)
        return representations, projections


# ============================================================================
# Loss Functions
# ============================================================================

def simclr_nt_xent_loss(z1, z2, tau=0.1):
    """
    SimCLR NT-Xent (InfoNCE) loss.

    Pulls together two views of the same sample, pushes apart different samples.

    Args:
        z1, z2: (batch, proj_dim) L2-normalized projections
        tau: Temperature parameter
    Returns:
        Scalar loss
    """
    batch_size = z1.shape[0]
    # L2 normalization ensures that the dot product equals cosine similarity,
    # bounding the similarity values to [-1/tau, 1/tau].
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Stack both views: rows 0..N-1 are view-1, rows N..2N-1 are view-2
    z = torch.cat([z1, z2], dim=0)  # (2N, proj_dim)
    sim = torch.mm(z, z.t()) / tau  # (2N, 2N) cosine similarity matrix scaled by temperature

    # Remove self-similarity (diagonal) so a sample is never its own positive
    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)

    # Positive pairs:
    #   - for row i   (view-1 of sample i): positive is row i+N (view-2 of sample i)
    #   - for row i+N (view-2 of sample i): positive is row i   (view-1 of sample i)
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=z.device),
        torch.arange(0, batch_size, device=z.device),
    ])

    # Cross-entropy treats the positive pair as the "correct class" among all 2N-1 negatives
    loss = F.cross_entropy(sim, labels)
    return loss


def supervised_contrastive_loss(z1, z2, is_typical, tau=0.1):
    """
    Supervised Contrastive Loss (SupCon).

    All typical samples are positives of each other, all atypical samples are
    positives of each other, and cross-type pairs are negatives.

    Args:
        z1, z2: (batch, proj_dim) projections from two augmented views
        is_typical: (batch,) boolean mask
        tau: Temperature parameter
    Returns:
        Scalar loss
    """
    batch_size = z1.shape[0]
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Concatenate both views so each sample appears twice (once per view)
    z = torch.cat([z1, z2], dim=0)  # (2N, proj_dim)
    n = 2 * batch_size

    # Duplicate labels for both views: typical=1, atypical=0
    labels = is_typical.long()              # (N,)
    labels = torch.cat([labels, labels], dim=0)  # (2N,)

    sim = torch.mm(z, z.t()) / tau  # (2N, 2N) cosine similarity

    # Exclude self-similarity from the denominator
    self_mask = torch.eye(n, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(self_mask, -1e9)

    # Positive pairs: all samples of the same class (typical–typical or atypical–atypical),
    # including cross-view pairs of the same sample. Self-pairs are excluded.
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask  # (2N, 2N)

    # SupCon loss formula (Khosla et al., 2020):
    # L = -1/|P(i)| * sum_{p in P(i)} log[ exp(sim(i,p)/tau) / sum_{a != i} exp(sim(i,a)/tau) ]
    exp_sim = torch.exp(sim)       # (2N, 2N); self entries are ~0 due to -1e9 fill
    log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True))  # log of partition function
    log_prob = sim - log_denom     # log-softmax over all non-self pairs

    num_positives = pos_mask.float().sum(dim=1).clamp(min=1)  # |P(i)| per anchor
    mean_log_prob = (pos_mask.float() * log_prob).sum(dim=1) / num_positives

    loss = -mean_log_prob.mean()
    return loss


def vicreg_variance_loss(z, gamma=1.0):
    """
    VICReg variance regularization.

    Encourages each dimension to have std >= gamma, preventing collapse.

    Args:
        z: (batch, dim) representations
        gamma: Target standard deviation
    Returns:
        Scalar loss
    """
    # Compute per-dimension std across the batch
    std = z.std(dim=0)  # (dim,)
    # Penalize any dimension whose std falls below gamma (hinge loss).
    # This prevents the model from collapsing to a constant embedding,
    # which would make all contrastive losses trivially zero.
    loss = F.relu(gamma - std).mean()
    return loss


def anchor_distance_consistency_loss(repr1, repr2, is_typical):
    """
    Anchor-distance consistency loss.

    Computes typical-speech centroid from batch, then enforces that both views
    of the same sample have the same distance to this centroid.

    Args:
        repr1, repr2: (batch, hidden_dim) representations from pre_net
        is_typical: (batch,) boolean mask for typical speech samples
    Returns:
        Scalar loss (0.0 if no typical samples in batch)
    """
    if is_typical.sum() == 0:
        return torch.tensor(0.0, device=repr1.device)

    # Estimate the typical-speech centroid from both views (more robust than using one view)
    typical_repr = torch.cat([repr1[is_typical], repr2[is_typical]], dim=0)
    centroid = typical_repr.mean(dim=0, keepdim=True)  # (1, hidden_dim)

    # L2 distance of each view from the typical centroid
    dist1 = torch.norm(repr1 - centroid, dim=1)  # (batch,)
    dist2 = torch.norm(repr2 - centroid, dim=1)  # (batch,)

    # Both views of the same sample should be equally far from the centroid
    # (i.e., augmentation should not change the severity "distance").
    # MSE penalizes inconsistent distance estimates across views.
    loss = F.mse_loss(dist1, dist2)
    return loss


def margin_separation_loss(repr1, repr2, is_typical, margin=10.0):
    """
    Margin-based loss to enforce separation between typical and atypical centroids.

    Penalizes when the distance between the typical centroid and atypical centroid
    is smaller than the specified margin.

    Args:
        repr1, repr2: (batch, hidden_dim) representations from pre_net
        is_typical: (batch,) boolean mask for typical speech samples
        margin: Minimum desired distance between centroids
    Returns:
        Scalar loss (0.0 if either group is missing from the batch)
    """
    atypical_mask = ~is_typical
    # Both groups must be present in the batch for the loss to be defined
    if is_typical.sum() == 0 or atypical_mask.sum() == 0:
        return torch.tensor(0.0, device=repr1.device)

    # Average centroids over both views for robustness
    typical_center = torch.cat([repr1[is_typical], repr2[is_typical]], dim=0).mean(dim=0)
    atypical_center = torch.cat([repr1[atypical_mask], repr2[atypical_mask]], dim=0).mean(dim=0)

    dist = torch.norm(typical_center - atypical_center)
    # Hinge loss: penalize only when the inter-centroid distance is smaller than margin.
    # Once the two groups are separated by at least `margin`, the gradient goes to zero.
    loss = F.relu(margin - dist)
    return loss


def rating_to_group(rating: float, boundaries=(1.5, 2.5, 3.5, 4.5, 5.5, 6.5)) -> int:
    """
    Map a continuous rating to a discrete group index.

    Default boundaries split [1, 7] into 7 groups:
        [1.0, 1.5) → 0
        [1.5, 2.5) → 1
        [2.5, 3.5) → 2
        [3.5, 4.5) → 3
        [4.5, 5.5) → 4
        [5.5, 6.5) → 5
        [6.5, 7.0] → 6
    """
    group = 0
    for b in boundaries:
        if rating >= b:
            group += 1
        else:
            break
    return group


def label_supervised_contrastive_loss(z1, z2, ratings, tau=0.1,
                                      boundaries=(1.5, 2.5, 3.5, 4.5, 5.5, 6.5)):
    """
    Fine-grained supervised contrastive loss using label group assignments.

    Samples in the same rating group are treated as positives; samples in
    different groups are negatives.  Samples with NaN ratings are excluded.

    Args:
        z1, z2: (batch, proj_dim) projections from two augmented views
        ratings: (batch,) float tensor of raw rating values (NaN = exclude)
        tau: temperature
        boundaries: bin boundaries used by rating_to_group
    Returns:
        Scalar loss (0.0 if fewer than 2 valid samples)
    """
    # Map continuous ratings to discrete group indices; NaN → -1 (excluded from this loss)
    groups = torch.full((z1.shape[0],), -1, dtype=torch.long, device=z1.device)
    for i, r in enumerate(ratings):
        if not torch.isnan(r):
            groups[i] = rating_to_group(r.item(), boundaries)

    valid_mask = groups >= 0
    if valid_mask.sum() < 2:
        # Need at least 2 labeled samples to form any positive/negative pair
        return torch.tensor(0.0, device=z1.device)

    # Restrict to samples with known severity labels
    z1_v = F.normalize(z1[valid_mask], dim=1)
    z2_v = F.normalize(z2[valid_mask], dim=1)
    groups_v = groups[valid_mask]

    # Same SupCon structure as supervised_contrastive_loss, but positives are
    # now defined by fine-grained severity group rather than binary typical/atypical.
    # This encourages embeddings of similar severity to cluster together while
    # embeddings of different severity are pushed apart.
    n = 2 * z1_v.shape[0]
    z = torch.cat([z1_v, z2_v], dim=0)               # (2N, proj_dim)
    labels = torch.cat([groups_v, groups_v], dim=0)   # (2N,) same group for both views

    sim = torch.mm(z, z.t()) / tau
    self_mask = torch.eye(n, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(self_mask, -1e9)

    # Positive mask: same severity group, excluding self
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask

    exp_sim = torch.exp(sim)
    log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True))
    log_prob = sim - log_denom

    num_positives = pos_mask.float().sum(dim=1).clamp(min=1)
    mean_log_prob = (pos_mask.float() * log_prob).sum(dim=1) / num_positives

    loss = -mean_log_prob.mean()
    return loss


def continuous_label_supervised_contrastive_loss(z1, z2, ratings, tau=0.1, radius=1.0):
    """
    Continuous-label supervised contrastive loss.

    Instead of snapping ratings to discrete bins (as label_supervised_contrastive_loss
    does via rating_to_group), this loss treats a pair as positive whenever their
    rating difference falls within a continuous neighbourhood:

        positive  if  |rᵢ − rⱼ| < radius
        negative  if  |rᵢ − rⱼ| ≥ radius

    This avoids the quantisation artefacts of hard bin boundaries: two samples
    with ratings 1.4 and 1.6 land in different bins under the default boundaries
    but are correctly treated as positives here (radius=1 → |1.4−1.6|=0.2 < 1).

    Anchors with no positive neighbour in the batch are excluded from the loss
    (rather than forcing a vacuous gradient).

    Args:
        z1, z2: (batch, proj_dim) projections from two augmented views
        ratings: (batch,) float tensor of raw rating values (NaN = exclude)
        tau: temperature
        radius: half-width of the positive neighbourhood in rating space
                (e.g. radius=1.0 → pairs within 1 rating point are positive)
    Returns:
        Scalar loss (0.0 if fewer than 2 valid samples or no positive pairs)
    """
    valid_mask = ~torch.isnan(ratings)
    if valid_mask.sum() < 2:
        return torch.tensor(0.0, device=z1.device)

    z1_v = F.normalize(z1[valid_mask], dim=1)
    z2_v = F.normalize(z2[valid_mask], dim=1)
    r_v = ratings[valid_mask]

    n_v = z1_v.shape[0]
    n = 2 * n_v
    z = torch.cat([z1_v, z2_v], dim=0)   # (2N, proj_dim)
    r = torch.cat([r_v, r_v], dim=0)      # (2N,) — same rating for both views of a sample

    sim = torch.mm(z, z.t()) / tau
    self_mask = torch.eye(n, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(self_mask, -1e9)

    # Positive mask: |rᵢ − rⱼ| < radius, excluding self-pairs
    rating_diff = (r.unsqueeze(0) - r.unsqueeze(1)).abs()  # (2N, 2N)
    pos_mask = (rating_diff < radius) & ~self_mask

    # Skip anchors that have no positive neighbour in this batch
    has_positive = pos_mask.any(dim=1)
    if not has_positive.any():
        return torch.tensor(0.0, device=z1.device)

    exp_sim = torch.exp(sim)
    log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True))
    log_prob = sim - log_denom

    num_positives = pos_mask.float().sum(dim=1).clamp(min=1)
    mean_log_prob = (pos_mask.float() * log_prob).sum(dim=1) / num_positives

    # Average only over anchors that have at least one positive neighbour
    loss = -mean_log_prob[has_positive].mean()
    return loss


def rank_n_contrast_loss(z1, z2, ratings, tau=2.0, feature_sim='l2'):
    """
    Rank-N-Contrast (RNC) loss for continuous-label regression.

    Reference: Zha et al., "Rank-N-Contrast: Learning Continuous Representations
    for Regression", NeurIPS 2023 (Spotlight).
    https://arxiv.org/abs/2210.01189

    For each anchor i and every candidate j (j != i), the samples k with larger
    label distance to i than j has become negatives:

        N(i, j) = { k != i : |y_i - y_k| > |y_i - y_j| }

    Loss per valid pair (i, j):

        L(i,j) = -log( exp(s(z_i, z_j) / tau)
                       / (exp(s(z_i, z_j) / tau) + sum_{k in N(i,j)} exp(s(z_i, z_k) / tau)) )

    Pairs where N(i, j) is empty (j is already the farthest sample from i) are
    skipped, as there is no contrastive signal.  The total loss is the mean over
    all valid (i, j) pairs.

    Args:
        z1, z2: (batch, proj_dim) projections from two augmented views
        ratings: (batch,) float tensor of continuous labels (NaN = exclude)
        tau: temperature (paper default 2.0; larger than SimCLR because
             negative-L2 similarity spans a wider numerical range)
        feature_sim: 'l2'     → negative L2 distance (paper default)
                     'cosine' → cosine similarity
    Returns:
        Scalar loss (0.0 if fewer than 2 valid samples or no valid pairs)
    """
    valid_mask = ~torch.isnan(ratings)
    if valid_mask.sum() < 2:
        return torch.tensor(0.0, device=z1.device)

    z1_v = z1[valid_mask]
    z2_v = z2[valid_mask]
    r_v = ratings[valid_mask]

    # Concatenate both views so each sample appears twice
    z = torch.cat([z1_v, z2_v], dim=0)   # (2N, dim)
    r = torch.cat([r_v, r_v], dim=0)     # (2N,) — same label for both views
    total_n = z.shape[0]

    # --- Feature similarity matrix ---
    if feature_sim == 'l2':
        # Negative L2 distance: s(a, b) = -||a - b||_2
        # Computed via ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b  (numerically stable)
        # eps inside sqrt prevents inf gradient when two vectors are identical:
        # d/dx[sqrt(x)]|_{x=0} = inf, which propagates NaN through backprop.
        sq_norm = (z ** 2).sum(dim=1, keepdim=True)          # (2N, 1)
        sq_dist = (sq_norm + sq_norm.t() - 2.0 * torch.mm(z, z.t())).clamp(min=0.0)
        sim_mat = -(sq_dist + 1e-8).sqrt()
        sim_mat = sim_mat / tau
    else:  # cosine
        z_norm = F.normalize(z, dim=1)
        sim_mat = torch.mm(z_norm, z_norm.t()) / tau          # (2N, 2N)

    # --- Label difference matrix (L1) ---
    r_diff = (r.unsqueeze(1) - r.unsqueeze(0)).abs()         # (2N, 2N)

    # Self-mask: anchor i must not compare with itself
    self_mask = torch.eye(total_n, device=z.device, dtype=torch.bool)

    # exp(sim) with diagonal zeroed out (self-pairs contribute nothing)
    exp_sim = torch.exp(sim_mat).masked_fill(self_mask, 0.0)  # (2N, 2N)

    total_loss = torch.tensor(0.0, device=z.device)
    count = 0

    # Iterate over candidate j; inner ops are vectorised over anchor i.
    # Memory is O(N^2) per step, which is acceptable for typical batch sizes.
    for j in range(total_n):
        # neg_mask[i, k]: True if d(y_i, y_k) > d(y_i, y_j) and k != i
        # r_diff[:, j] gives d(y_i, y_j) for every anchor i  → shape (2N,)
        neg_mask = (r_diff > r_diff[:, j].unsqueeze(1)) & ~self_mask  # (2N, 2N)

        # Sum of exp(sim(i, k)) over all negatives k for each anchor i
        neg_sum = (exp_sim * neg_mask.float()).sum(dim=1)    # (2N,)

        # Valid anchor–candidate pairs: i != j AND at least one negative exists
        valid = (~self_mask[:, j]) & (neg_sum > 0)
        if not valid.any():
            continue

        # Numerator: exp(sim(i, j)) for each valid anchor i
        num = exp_sim[:, j]                                  # (2N,)

        # -log(num / (num + neg_sum))
        loss_terms = -torch.log(num[valid] / (num[valid] + neg_sum[valid]))
        total_loss = total_loss + loss_terms.sum()
        count += valid.sum().item()

    if count > 0:
        total_loss = total_loss / count
    return total_loss


def mixup_distance_smoothness_loss(z1, z2, atypical_mask, typical_centroid):
    """Force smooth interpolation in distance-to-centroid."""
    if not atypical_mask.any():
        return torch.tensor(0.0, device=z1.device)

    atypical_z = z1[atypical_mask]
    n = len(atypical_z)

    if n < 2:
        return torch.tensor(0.0, device=z1.device)

    # Sample pairs
    idx1 = torch.randint(0, n, (n // 2,))
    idx2 = torch.randint(0, n, (n // 2,))

    # Mix them
    lambda_ = torch.rand(len(idx1), 1, device=z1.device)
    z_mixed = lambda_ * atypical_z[idx1] + (1 - lambda_) * atypical_z[idx2]

    # Distances
    d1 = torch.norm(atypical_z[idx1] - typical_centroid, dim=1)
    d2 = torch.norm(atypical_z[idx2] - typical_centroid, dim=1)
    d_mixed = torch.norm(z_mixed - typical_centroid, dim=1)

    # Expected: linear interpolation
    d_expected = lambda_.squeeze() * d1 + (1 - lambda_.squeeze()) * d2

    return F.mse_loss(d_mixed, d_expected)


# ============================================================================
# Evaluation Dataset & Visualization
# ============================================================================

class EvalDataset(Dataset):
    """
    Dataset for evaluation/visualization of learned embeddings.

    Loads pre-extracted .npy features + labels (no augmentation).
    Combines atypical samples (from dataset_labeled with severity ratings)
    and typical samples (pre-extracted .npy, subsampled for speed).
    """

    def __init__(self, atypical_feature_dir, atypical_metadata_path,
                 typical_feature_dirs=None, typical_metadata_paths=None,
                 typical_max_samples=500):
        """
        Args:
            atypical_feature_dir: Directory containing atypical .npy features
            atypical_metadata_path: Path to labeled JSON metadata
            typical_feature_dirs: List of directories containing typical .npy features
            typical_metadata_paths: List of paths to typical JSON metadata files
            typical_max_samples: Max total typical samples (subsampled across all splits)
        """
        self.samples = []  # list of (feat_path, intell, natur, avg, is_typical)

        # Load atypical samples with labels
        with open(atypical_metadata_path, 'r') as f:
            metadata = json.load(f)

        for filename, info in metadata.items():
            if filename.startswith("_"):
                continue
            ratings = info.get("ratings", {})
            intell = ratings.get("Intelligibility", None)
            natur = ratings.get("Naturalness", None)
            if intell is None or natur is None:
                continue

            avg = ratings.get("Average", None)
            if avg is None:
                avg = (intell + natur) / 2.0

            feat_name = os.path.splitext(filename)[0] + ".npy"
            feat_path = os.path.join(atypical_feature_dir, feat_name)

            self.samples.append((feat_path, int(intell), int(natur), float(avg), False))

        print(f"EvalDataset: {len(self.samples)} atypical samples with labels")

        # Load typical samples from multiple splits (pre-extracted .npy, subsampled)
        if typical_feature_dirs and typical_metadata_paths:
            typical_feats = []
            total_available = 0
            for feat_dir, meta_path in zip(typical_feature_dirs, typical_metadata_paths):
                if not (meta_path and os.path.exists(meta_path) and os.path.exists(feat_dir)):
                    continue
                with open(meta_path, 'r') as f:
                    typical_meta = json.load(f)
                for filename in typical_meta:
                    if filename.startswith("_"):
                        continue
                    feat_name = os.path.splitext(filename)[0] + ".npy"
                    feat_path = os.path.join(feat_dir, feat_name)
                    typical_feats.append(feat_path)
                total_available += len(typical_meta)

            # Subsample for speed
            if len(typical_feats) > typical_max_samples:
                random.shuffle(typical_feats)
                typical_feats = typical_feats[:typical_max_samples]

            for feat_path in typical_feats:
                self.samples.append((feat_path, 0, 0, 0.0, True))

            print(f"EvalDataset: {len(typical_feats)} typical samples (subsampled from {total_available})")

        print(f"EvalDataset: {len(self.samples)} total eval samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, intell, natur, avg, is_typical = self.samples[idx]
        features = torch.from_numpy(np.load(path)).float()
        return features, intell, natur, avg, is_typical


def eval_collate_fn(batch):
    """Collate variable-length sequences with padding for eval."""
    features_list, intell_list, natur_list, avg_list, typical_list = zip(*batch)

    max_len = max(f.shape[0] for f in features_list)
    hidden_dim = features_list[0].shape[1]
    batch_size = len(features_list)

    padded = torch.zeros(batch_size, max_len, hidden_dim)
    lengths = torch.zeros(batch_size, dtype=torch.long)

    for i in range(batch_size):
        l = features_list[i].shape[0]
        padded[i, :l] = features_list[i]
        lengths[i] = l

    intell = torch.tensor(intell_list, dtype=torch.long)
    natur = torch.tensor(natur_list, dtype=torch.long)
    avg = torch.tensor(avg_list, dtype=torch.float)
    is_typical = torch.tensor(typical_list, dtype=torch.bool)

    return padded, lengths, intell, natur, avg, is_typical


@torch.no_grad()
def evaluate_and_visualize(model, eval_loader, device, epoch, output_dir, use_wandb=False):
    """
    Extract embeddings from eval set, run t-SNE, and save plots + embeddings to disk.
    Optionally logs to wandb.
    """
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()

    all_repr = []
    all_proj = []
    all_intell = []
    all_natur = []
    all_avg = []
    all_typical = []

    for batch in tqdm(eval_loader, desc="Eval embedding extraction"):
        padded, lengths, intell, natur, avg, is_typical = batch
        padded = padded.to(device)
        lengths = lengths.to(device)

        representations, projections = model(padded, lengths)
        all_repr.append(representations.cpu())
        all_proj.append(projections.cpu())
        all_intell.append(intell)
        all_natur.append(natur)
        all_avg.append(avg)
        all_typical.append(is_typical)

    all_repr = torch.cat(all_repr, dim=0).numpy()       # (N, hidden_dim)
    all_proj = torch.cat(all_proj, dim=0).numpy()       # (N, proj_dim)
    all_intell = torch.cat(all_intell, dim=0).numpy()    # (N,)
    all_natur = torch.cat(all_natur, dim=0).numpy()      # (N,)
    all_avg = torch.cat(all_avg, dim=0).numpy()          # (N,)
    all_typical = torch.cat(all_typical, dim=0).numpy()   # (N,)

    # Compute distance to typical centroid
    typical_mask = all_typical.astype(bool)
    if typical_mask.sum() > 0:
        centroid = all_repr[typical_mask].mean(axis=0, keepdims=True)
    else:
        centroid = all_repr.mean(axis=0, keepdims=True)
    distances = np.linalg.norm(all_repr - centroid, axis=1)

    # Save embeddings + metadata to disk
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    npz_path = os.path.join(eval_dir, f"embeddings_epoch_{epoch:03d}.npz")
    np.savez(npz_path,
             representations=all_repr,
             projections=all_proj,
             intelligibility=all_intell,
             naturalness=all_natur,
             average=all_avg,
             is_typical=all_typical,
             distances=distances,
             centroid=centroid.squeeze())
    print(f"Saved embeddings to {npz_path}")

    # Compute and save average distance per label
    atypical_mask = ~all_typical.astype(bool)
    avg_dist_stats = {"epoch": epoch}
    # Round average to nearest 0.5 for binning in distance stats
    all_avg_binned = np.round(all_avg * 2) / 2
    for label_name, labels in [("naturalness", all_natur), ("intelligibility", all_intell), ("average", all_avg_binned)]:
        per_label = {}
        for level in sorted(set(labels[atypical_mask])):
            mask = atypical_mask & (labels == level)
            key = float(level) if label_name == "average" else int(level)
            per_label[key] = {"mean": float(distances[mask].mean()),
                              "std": float(distances[mask].std()),
                              "count": int(mask.sum())}
        avg_dist_stats[label_name] = per_label
    if all_typical.astype(bool).sum() > 0:
        avg_dist_stats["typical"] = {"mean": float(distances[all_typical.astype(bool)].mean()),
                                     "std": float(distances[all_typical.astype(bool)].std()),
                                     "count": int(all_typical.astype(bool).sum())}
    dist_stats_path = os.path.join(eval_dir, f"distance_stats_epoch_{epoch:03d}.json")
    with open(dist_stats_path, 'w') as f:
        json.dump(avg_dist_stats, f, indent=2)
    print(f"Saved distance stats to {dist_stats_path}")

    # t-SNE on representations
    print("Running t-SNE on representations...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_repr) - 1))
    embeddings_2d = tsne.fit_transform(all_repr)

    # t-SNE on projections
    print("Running t-SNE on projections...")
    tsne_proj = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_proj) - 1))
    proj_2d = tsne_proj.fit_transform(all_proj)

    # --- wandb Tables for interactive plots ---
    if use_wandb:
        table = wandb.Table(columns=["tsne_x", "tsne_y", "speech_type", "intelligibility",
                                      "naturalness", "average", "distance_to_centroid"])
        table_proj = wandb.Table(columns=["tsne_x", "tsne_y", "speech_type", "intelligibility",
                                           "naturalness", "average", "distance_to_centroid"])
        for i in range(len(all_repr)):
            speech_type = "Typical" if all_typical[i] else "Atypical"
            table.add_data(
                float(embeddings_2d[i, 0]), float(embeddings_2d[i, 1]),
                speech_type, int(all_intell[i]), int(all_natur[i]),
                float(all_avg[i]), float(distances[i])
            )
            table_proj.add_data(
                float(proj_2d[i, 0]), float(proj_2d[i, 1]),
                speech_type, int(all_intell[i]), int(all_natur[i]),
                float(all_avg[i]), float(distances[i])
            )

    # --- Matplotlib scatter plots ---
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    # 1) Speech Type scatter
    ax = axes[0]
    typ_mask = all_typical.astype(bool)
    ax.scatter(embeddings_2d[typ_mask, 0], embeddings_2d[typ_mask, 1],
               c='blue', alpha=0.5, s=10, label='Typical')
    ax.scatter(embeddings_2d[~typ_mask, 0], embeddings_2d[~typ_mask, 1],
               c='red', alpha=0.5, s=10, label='Atypical')
    ax.set_title("Speech Type")
    ax.legend()

    # 2) Naturalness scatter (atypical only, colored by rating)
    ax = axes[1]
    atypical_idx = ~typ_mask
    sc = ax.scatter(embeddings_2d[atypical_idx, 0], embeddings_2d[atypical_idx, 1],
                    c=all_natur[atypical_idx], cmap='RdYlGn', alpha=0.6, s=10, vmin=1, vmax=7)
    if typ_mask.sum() > 0:
        ax.scatter(embeddings_2d[typ_mask, 0], embeddings_2d[typ_mask, 1],
                   c='blue', alpha=0.3, s=10, marker='x', label='Typical')
        ax.legend()
    plt.colorbar(sc, ax=ax, label='Naturalness (1-7)')
    ax.set_title("Naturalness")

    # 3) Intelligibility scatter (atypical only, colored by rating)
    ax = axes[2]
    sc = ax.scatter(embeddings_2d[atypical_idx, 0], embeddings_2d[atypical_idx, 1],
                    c=all_intell[atypical_idx], cmap='RdYlGn', alpha=0.6, s=10, vmin=1, vmax=7)
    if typ_mask.sum() > 0:
        ax.scatter(embeddings_2d[typ_mask, 0], embeddings_2d[typ_mask, 1],
                   c='blue', alpha=0.3, s=10, marker='x', label='Typical')
        ax.legend()
    plt.colorbar(sc, ax=ax, label='Intelligibility (1-7)')
    ax.set_title("Intelligibility")

    # 4) Average scatter (atypical only, colored by average rating)
    ax = axes[3]
    sc = ax.scatter(embeddings_2d[atypical_idx, 0], embeddings_2d[atypical_idx, 1],
                    c=all_avg[atypical_idx], cmap='RdYlGn', alpha=0.6, s=10, vmin=1, vmax=7)
    if typ_mask.sum() > 0:
        ax.scatter(embeddings_2d[typ_mask, 0], embeddings_2d[typ_mask, 1],
                   c='blue', alpha=0.3, s=10, marker='x', label='Typical')
        ax.legend()
    plt.colorbar(sc, ax=ax, label='Average (1-7)')
    ax.set_title("Average")

    plt.suptitle(f"Representations (Epoch {epoch})")
    plt.tight_layout()

    # --- Projection t-SNE scatter plots ---
    fig_proj, axes_proj = plt.subplots(1, 4, figsize=(24, 5))

    ax = axes_proj[0]
    ax.scatter(proj_2d[typ_mask, 0], proj_2d[typ_mask, 1],
               c='blue', alpha=0.5, s=10, label='Typical')
    ax.scatter(proj_2d[~typ_mask, 0], proj_2d[~typ_mask, 1],
               c='red', alpha=0.5, s=10, label='Atypical')
    ax.set_title("Speech Type")
    ax.legend()

    ax = axes_proj[1]
    sc = ax.scatter(proj_2d[atypical_idx, 0], proj_2d[atypical_idx, 1],
                    c=all_natur[atypical_idx], cmap='RdYlGn', alpha=0.6, s=10, vmin=1, vmax=7)
    if typ_mask.sum() > 0:
        ax.scatter(proj_2d[typ_mask, 0], proj_2d[typ_mask, 1],
                   c='blue', alpha=0.3, s=10, marker='x', label='Typical')
        ax.legend()
    plt.colorbar(sc, ax=ax, label='Naturalness (1-7)')
    ax.set_title("Naturalness")

    ax = axes_proj[2]
    sc = ax.scatter(proj_2d[atypical_idx, 0], proj_2d[atypical_idx, 1],
                    c=all_intell[atypical_idx], cmap='RdYlGn', alpha=0.6, s=10, vmin=1, vmax=7)
    if typ_mask.sum() > 0:
        ax.scatter(proj_2d[typ_mask, 0], proj_2d[typ_mask, 1],
                   c='blue', alpha=0.3, s=10, marker='x', label='Typical')
        ax.legend()
    plt.colorbar(sc, ax=ax, label='Intelligibility (1-7)')
    ax.set_title("Intelligibility")

    ax = axes_proj[3]
    sc = ax.scatter(proj_2d[atypical_idx, 0], proj_2d[atypical_idx, 1],
                    c=all_avg[atypical_idx], cmap='RdYlGn', alpha=0.6, s=10, vmin=1, vmax=7)
    if typ_mask.sum() > 0:
        ax.scatter(proj_2d[typ_mask, 0], proj_2d[typ_mask, 1],
                   c='blue', alpha=0.3, s=10, marker='x', label='Typical')
        ax.legend()
    plt.colorbar(sc, ax=ax, label='Average (1-7)')
    ax.set_title("Average")

    plt.suptitle(f"Projections (Epoch {epoch})")
    plt.tight_layout()

    # --- Distance histogram ---
    fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
    if typ_mask.sum() > 0:
        ax_hist.hist(distances[typ_mask], bins=30, alpha=0.6, label='Typical', color='blue')
    for severity in sorted(set(all_natur[~typ_mask])):
        mask_s = (~typ_mask) & (all_natur == severity)
        if mask_s.sum() > 0:
            ax_hist.hist(distances[mask_s], bins=30, alpha=0.4, label=f'Naturalness={severity}')
    ax_hist.set_xlabel("Distance to Typical Centroid")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title(f"Distance Distribution (Epoch {epoch})")
    ax_hist.legend(fontsize=8)
    plt.tight_layout()

    # Save plots to disk
    scatter_path = os.path.join(eval_dir, f"tsne_scatter_epoch_{epoch:03d}.pdf")
    proj_scatter_path = os.path.join(eval_dir, f"tsne_proj_scatter_epoch_{epoch:03d}.pdf")
    hist_path = os.path.join(eval_dir, f"distance_hist_epoch_{epoch:03d}.pdf")
    fig.savefig(scatter_path)
    fig_proj.savefig(proj_scatter_path)
    fig_hist.savefig(hist_path)
    print(f"Saved plots to {scatter_path}, {proj_scatter_path}, and {hist_path}")

    # Log to wandb
    if use_wandb:
        wandb.log({
            "eval/embedding_table": table,
            "eval/projection_table": table_proj,
            "eval/tsne_scatter": wandb.Image(fig),
            "eval/tsne_proj_scatter": wandb.Image(fig_proj),
            "eval/distance_histogram": wandb.Image(fig_hist),
            "epoch": epoch,
        })

    plt.close(fig)
    plt.close(fig_proj)
    plt.close(fig_hist)
    print(f"Eval visualization complete (epoch {epoch})")


# ============================================================================
# Training
# ============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args):
    # Validate input directories exist
    assert not (args.w_supcon > 0 and args.w_label_supcon > 0), \
        "w_supcon and w_label_supcon cannot both be > 0: they conflict " \
        "(supcon pulls all atypical together; label_supcon pushes them apart by severity)"
    assert not (args.w_supcon > 0 and args.w_continuous_supcon > 0), \
        "w_supcon and w_continuous_supcon cannot both be > 0: same conflict as above"
    assert not (args.w_supcon > 0 and args.w_rnc > 0), \
        "w_supcon and w_rnc cannot both be > 0: supcon pulls all atypical together " \
        "while rnc pushes them apart by severity rank"

    assert os.path.isdir(args.atypical_feature_dir), \
        f"Atypical feature dir not found: {args.atypical_feature_dir}"
    assert os.path.isdir(args.atypical_data_dir), \
        f"Atypical data dir not found: {args.atypical_data_dir}"
    if args.typical_feature_dir and not os.path.isdir(args.typical_feature_dir):
        raise ValueError(f"Typical feature dir not found: {args.typical_feature_dir}")
    if args.typical_data_dir and not os.path.isdir(args.typical_data_dir):
        raise ValueError(f"Typical data dir not found: {args.typical_data_dir}")

    # DDP setup (or single-GPU fallback)
    distributed = is_ddp()
    if distributed:
        rank, local_rank, world_size = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    set_seed(args.seed + rank)  # different seed per rank for augmentation diversity

    if is_main_process(rank):
        print(f"Using device: {device}" + (f" (DDP: world_size={world_size})" if distributed else ""))

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.exp_name or f"contrastive_pretrain_{timestamp}"
    output_dir = os.path.join(args.output_dir, exp_name)
    if is_main_process(rank):
        os.makedirs(output_dir, exist_ok=True)
    if distributed:
        dist.barrier()  # wait for rank 0 to create dir

    if is_main_process(rank):
        print(f"Output directory: {output_dir}")

    # Save args
    if is_main_process(rank):
        with open(os.path.join(output_dir, "args.json"), 'w') as f:
            json.dump(vars(args), f, indent=2)

    # Initialize wandb
    use_wandb = args.wandb and HAS_WANDB and is_main_process(rank)
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=exp_name,
            config=vars(args),
            dir=output_dir,
        )
    elif args.wandb and not HAS_WANDB and is_main_process(rank):
        print("WARNING: --wandb flag set but wandb is not installed. Skipping.")

    # Load datasets from separate typical and atypical sources
    if is_main_process(rank):
        print("\nLoading datasets...")
    train_datasets = []

    # Atypical (dysarthric) speech — pre-extracted .npy features
    for split in args.atypical_splits:
        feat_dir = os.path.join(args.atypical_feature_dir, split)
        meta_path = os.path.join(args.atypical_data_dir, f"{split}.json")
        if os.path.exists(meta_path) and os.path.exists(feat_dir):
            ds = ContrastiveDataset(
                feat_dir, meta_path, is_typical=False,
                noise_std=args.aug_noise_std,
                max_mask_ratio=args.aug_mask_ratio,
                crop_min_ratio=args.aug_crop_min_ratio,
                label12_as_typical=args.label12_as_typical,
                label12_target=args.label12_target,
                label_typical_threshold=args.label_typical_threshold,
                label_supcon_target=args.label_supcon_target,
                typical_supcon_group=args.label_supcon_typical_group,
            )
            train_datasets.append(ds)
        elif is_main_process(rank):
            print(f"  Skipping atypical split {split} - not found")

    # Typical speech (e.g., LibriSpeech) — pre-extracted .npy features (optional)
    if args.typical_feature_dir and args.typical_data_dir:
        for split in args.typical_splits:
            feat_dir = os.path.join(args.typical_feature_dir, split)
            meta_path = os.path.join(args.typical_data_dir, f"{split}.json")
            if os.path.exists(meta_path) and os.path.exists(feat_dir):
                ds = ContrastiveDataset(
                    feat_dir, meta_path, is_typical=True,
                    noise_std=args.aug_noise_std,
                    max_mask_ratio=args.aug_mask_ratio,
                    crop_min_ratio=args.aug_crop_min_ratio,
                    label_supcon_target=args.label_supcon_target,
                    typical_supcon_group=args.label_supcon_typical_group,
                )
                train_datasets.append(ds)
            elif is_main_process(rank):
                print(f"  Skipping typical split {split} - not found")
    elif is_main_process(rank):
        print("  No typical_feature_dir/typical_data_dir provided; relying on label12_as_typical for typical samples")

    if not train_datasets:
        raise ValueError("No training data found")

    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    # Count per-sample typical/atypical (accounts for label12_as_typical relabeling)
    num_typical = sum(1 for ds in train_datasets for _, t, _ in ds.samples if t)
    num_atypical = sum(1 for ds in train_datasets for _, t, _ in ds.samples if not t)
    if is_main_process(rank):
        print(f"Total training samples: {len(train_dataset)} "
              f"(typical: {num_typical}, atypical: {num_atypical})")

    # Build sampler
    if distributed:
        # DistributedSampler splits the dataset across ranks without overlap,
        # so each GPU processes a disjoint partition of the data per epoch.
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True,
        )
        if is_main_process(rank):
            print(f"DistributedSampler: world_size={world_size}, "
                  f"per-GPU samples={len(train_sampler)}")
    elif args.sampler_type == "random":
        # Option 1: plain random shuffle — no reweighting.
        # Dataset imbalance is left as-is; useful as a baseline.
        train_sampler = None
        if is_main_process(rank):
            print("Sampler: random shuffle (no reweighting)")

    elif args.sampler_type == "binary":
        # Option 2: binary typical/atypical reweighting (original behaviour).
        # Ensures typical_ratio fraction of typical samples per batch on average.
        if num_typical > 0 and num_atypical > 0:
            # Per-sample weights: uniform within each group, scaled so that the
            # expected fraction of typical samples in the draw equals typical_ratio.
            w_typical = args.typical_ratio / num_typical
            w_atypical = (1.0 - args.typical_ratio) / num_atypical

            sample_weights = []
            for ds in train_datasets:
                for _, t, _ in ds.samples:
                    sample_weights.append(w_typical if t else w_atypical)
            sample_weights = torch.tensor(sample_weights, dtype=torch.float64)

            # Cap epoch size so neither group is over-sampled beyond its capacity
            num_samples_per_epoch = int(min(
                num_atypical / (1.0 - args.typical_ratio),
                num_typical / args.typical_ratio,
            ))
            train_sampler = torch.utils.data.WeightedRandomSampler(
                sample_weights, num_samples=num_samples_per_epoch, replacement=True
            )
            if is_main_process(rank):
                print(f"Sampler: binary WeightedRandom  typical_ratio={args.typical_ratio:.2f}, "
                      f"samples_per_epoch={num_samples_per_epoch}")
        else:
            train_sampler = None
            if is_main_process(rank):
                print("WARNING: Only one speech type found, falling back to random shuffle")

    elif args.sampler_type == "group":
        # Option 3: fine-grained group-based reweighting.
        # Each severity group (as defined by rating_to_group) gets equal total weight,
        # so every group contributes equally to each batch regardless of its size.
        # This is critical for label_supcon: without it, sparse severity groups rarely
        # appear in a batch, making the within-group positive gradient near-zero.
        #
        # Group assignment:
        #   - True typical speech (LibriSpeech): rating_value = typical_supcon_group (0.0)
        #   - Atypical with valid rating: rating_to_group(rating_value)
        #   - NaN rating_value (unlabeled): treated as group 0 (same as typical)
        all_ratings = []
        for ds in train_datasets:
            for _, _, rv in ds.samples:
                all_ratings.append(rv)

        # Map each sample to its severity group index
        sample_groups = []
        for rv in all_ratings:
            if np.isnan(rv):
                sample_groups.append(0)  # NaN → group 0 (typical-like)
            else:
                sample_groups.append(rating_to_group(rv))

        # Count samples per group
        from collections import Counter
        group_counts = Counter(sample_groups)
        num_groups = len(group_counts)
        if is_main_process(rank):
            print(f"Sampler: group-based WeightedRandom  groups={dict(sorted(group_counts.items()))}")

        # Weight = 1 / group_count so that each group contributes equally in expectation
        sample_weights = torch.tensor(
            [1.0 / group_counts[g] for g in sample_groups], dtype=torch.float64
        )

        # Epoch size matches random sampling: same number of steps as iterating the full dataset.
        # With replacement=True and uniform group weights, each group still contributes equally
        # in expectation, but the total draws per epoch equals the full dataset size.
        num_samples_per_epoch = len(train_dataset)
        train_sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, num_samples=num_samples_per_epoch, replacement=True
        )
        if is_main_process(rank):
            min_count = min(group_counts.values())
            print(f"  min_group={min_count}, num_groups={num_groups}, "
                  f"samples_per_epoch={num_samples_per_epoch}")
    else:
        raise ValueError(f"Unknown sampler_type: {args.sampler_type}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=contrastive_collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # Create eval dataset for visualization (rank 0 only)
    eval_loader = None
    if args.eval_every > 0 and is_main_process(rank):
        eval_feat_dir = os.path.join(args.atypical_feature_dir, args.eval_split)
        eval_meta_path = os.path.join(args.atypical_data_dir, f"{args.eval_split}.json")
        # Use eval_typical_splits (dev-clean, dev-other) for eval typical samples
        typical_eval_feat_dirs = []
        typical_eval_meta_paths = []
        if args.typical_feature_dir and args.typical_data_dir:
            for split in args.eval_typical_splits:
                typical_eval_feat_dirs.append(os.path.join(args.typical_feature_dir, split))
                typical_eval_meta_paths.append(os.path.join(args.typical_data_dir, f"{split}.json"))

        if os.path.exists(eval_meta_path) and os.path.exists(eval_feat_dir):
            eval_dataset = EvalDataset(
                atypical_feature_dir=eval_feat_dir,
                atypical_metadata_path=eval_meta_path,
                typical_feature_dirs=typical_eval_feat_dirs,
                typical_metadata_paths=typical_eval_meta_paths,
                typical_max_samples=args.eval_typical_samples,
            )
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=eval_collate_fn,
                pin_memory=True,
            )
        else:
            print(f"WARNING: Eval data not found at {eval_meta_path} or {eval_feat_dir}, skipping visualization")

    # Create model
    ModelClass = ContrastiveModelV2 if args.model_version == "v2" else ContrastiveModel
    model_kwargs = dict(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        proj_dim=args.proj_dim,
    )
    if args.model_version == "v2":
        model_kwargs["dropout"] = args.dropout
    model = ModelClass(**model_kwargs).to(device)

    if is_main_process(rank):
        print(model)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {num_params:,}")

    # Wrap model with DDP
    if distributed:
        model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Get the underlying model for saving weights (unwrap DDP)
    raw_model = model.module if distributed else model

    # Run initial evaluation before training (epoch 0)
    if eval_loader is not None and is_main_process(rank):
        print("\nRunning initial evaluation (epoch 0)...")
        evaluate_and_visualize(raw_model, eval_loader, device, 0, output_dir, use_wandb)

    # Training loop
    if is_main_process(rank):
        print(f"\nStarting training for {args.epochs} epochs...")
        print(f"Loss weights: contrast={args.w_contrast}, supcon={args.w_supcon}, label_supcon={args.w_label_supcon}, continuous_supcon={args.w_continuous_supcon}, rnc={args.w_rnc}, var={args.w_var}, anchor={args.w_anchor}, margin={args.w_margin}, mixup={args.w_mixup}")
        if args.w_label_supcon > 0:
            print(f"Label supcon: target={args.label_supcon_target}, typical_group={args.label_supcon_typical_group}")
        if args.label12_as_typical:
            print(f"Label-as-typical: enabled (target={args.label12_target}, threshold={args.label_typical_threshold})")
        if distributed:
            print(f"Effective batch size: {args.batch_size} x {world_size} = {args.batch_size * world_size}")
    history = []
    best_loss = float("inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        # Set epoch for DistributedSampler (ensures proper shuffling)
        if distributed:
            train_sampler.set_epoch(epoch)

        model.train()
        total_loss = 0
        total_contrast = 0
        total_supcon = 0
        total_label_supcon = 0
        total_continuous_supcon = 0
        total_rnc = 0
        total_var = 0
        total_anchor = 0
        total_margin = 0
        total_mixup = 0
        total_center_sep = 0
        total_typical_spread = 0
        total_atypical_spread = 0
        total_aug_consistency = 0
        total_emb_var = 0
        total_typical_count = 0
        total_atypical_count = 0
        metric_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", disable=not is_main_process(rank))
        for batch in pbar:
            padded1, lengths1, padded2, lengths2, is_typical, ratings = batch
            padded1 = padded1.to(device)
            lengths1 = lengths1.to(device)
            padded2 = padded2.to(device)
            lengths2 = lengths2.to(device)
            is_typical = is_typical.to(device)
            ratings = ratings.to(device)

            n_typical = is_typical.sum().item()
            n_atypical = len(is_typical) - n_typical
            total_typical_count += n_typical
            total_atypical_count += n_atypical

            # Forward pass for both views
            repr1, proj1 = model(padded1, lengths1)
            repr2, proj2 = model(padded2, lengths2)

            # Compute losses (skip if weight is zero)
            zero = torch.tensor(0.0, device=device)
            l_contrast = simclr_nt_xent_loss(proj1, proj2, tau=args.tau) \
                if args.w_contrast > 0 else zero
            l_supcon = supervised_contrastive_loss(proj1, proj2, is_typical, tau=args.tau) \
                if args.w_supcon > 0 else zero
            l_label_supcon = label_supervised_contrastive_loss(
                proj1, proj2, ratings, tau=args.tau) \
                if args.w_label_supcon > 0 else zero
            l_continuous_supcon = continuous_label_supervised_contrastive_loss(
                proj1, proj2, ratings, tau=args.tau, radius=args.continuous_radius) \
                if args.w_continuous_supcon > 0 else zero
            l_rnc = rank_n_contrast_loss(
                proj1, proj2, ratings, tau=args.tau, feature_sim=args.rnc_feature_sim) \
                if args.w_rnc > 0 else zero
            l_var = vicreg_variance_loss(
                torch.cat([proj1, proj2], dim=0), gamma=args.vicreg_gamma
            ) if args.w_var > 0 else zero
            l_anchor = anchor_distance_consistency_loss(repr1, repr2, is_typical) \
                if args.w_anchor > 0 else zero
            l_margin = margin_separation_loss(repr1, repr2, is_typical, margin=args.margin) \
                if args.w_margin > 0 else zero

            # Mixup distance smoothness loss
            if args.w_mixup > 0:
                atypical_mask_loss = ~is_typical
                if is_typical.sum() > 0:
                    typical_centroid = torch.cat([repr1[is_typical], repr2[is_typical]], dim=0).mean(dim=0)
                else:
                    typical_centroid = torch.cat([repr1, repr2], dim=0).mean(dim=0)
                l_mixup = mixup_distance_smoothness_loss(repr1, repr2, atypical_mask_loss, typical_centroid)
            else:
                l_mixup = zero

            # Weighted sum of all losses. Each term is independently scaled so
            # that the contribution of each objective can be tuned without
            # affecting the others:
            #   l_contrast     : self-supervised instance discrimination (SimCLR)
            #   l_supcon       : binary typical/atypical cluster separation
            #   l_label_supcon : fine-grained severity-group clustering (cannot combine with l_supcon)
            #   l_var          : prevents representational collapse (VICReg)
            #   l_anchor       : view-consistency of distance to typical centroid
            #   l_margin       : forces minimum inter-centroid gap between groups
            #   l_mixup        : smoothness of severity manifold via convex interpolation
            loss = (args.w_contrast * l_contrast +
                    args.w_supcon * l_supcon +
                    args.w_label_supcon * l_label_supcon +
                    args.w_continuous_supcon * l_continuous_supcon +
                    args.w_rnc * l_rnc +
                    args.w_var * l_var +
                    args.w_anchor * l_anchor +
                    args.w_margin * l_margin +
                    args.w_mixup * l_mixup)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_contrast += l_contrast.item()
            total_supcon += l_supcon.item()
            total_label_supcon += l_label_supcon.item()
            total_continuous_supcon += l_continuous_supcon.item()
            total_rnc += l_rnc.item()
            total_var += l_var.item()
            total_anchor += l_anchor.item()
            total_margin += l_margin.item()
            total_mixup += l_mixup.item()

            # Compute diagnostic metrics (no grad needed)
            with torch.no_grad():
                atypical_mask = ~is_typical
                proj_cat = torch.cat([proj1, proj2], dim=0)

                # Embedding variance: per-dimension std across batch
                step_emb_var = proj_cat.std(dim=0).mean().item()
                total_emb_var += step_emb_var

                if is_typical.sum() > 0 and atypical_mask.sum() > 0:
                    typical_repr = torch.cat([repr1[is_typical], repr2[is_typical]], dim=0)
                    atypical_repr = torch.cat([repr1[atypical_mask], repr2[atypical_mask]], dim=0)
                    typical_center = typical_repr.mean(dim=0)
                    atypical_center = atypical_repr.mean(dim=0)

                    # Center separation
                    total_center_sep += torch.norm(typical_center - atypical_center).item()

                    # Within-class spread
                    total_typical_spread += torch.norm(typical_repr - typical_center, dim=1).mean().item()
                    total_atypical_spread += torch.norm(atypical_repr - atypical_center, dim=1).mean().item()

                    # Augmentation consistency: |dist(view1, centroid) - dist(view2, centroid)|
                    centroid = typical_repr.mean(dim=0, keepdim=True)
                    d1 = torch.norm(repr1 - centroid, dim=1)
                    d2 = torch.norm(repr2 - centroid, dim=1)
                    total_aug_consistency += (d1 - d2).abs().mean().item()

                    metric_count += 1

            # Per-step wandb logging (every 100 steps)
            if use_wandb and is_main_process(rank) and global_step % 100 == 0:
                step_log = {
                    "train/loss": loss.item(),
                    "train/contrast": l_contrast.item(),
                    "train/supcon": l_supcon.item(),
                    "train/label_supcon": l_label_supcon.item(),
                    "train/continuous_supcon": l_continuous_supcon.item(),
                    "train/rnc": l_rnc.item(),
                    "train/var": l_var.item(),
                    "train/anchor": l_anchor.item(),
                    "train/margin": l_margin.item(),
                    "train/mixup": l_mixup.item(),
                    "train/emb_var": step_emb_var,
                    "data/step_typical": n_typical,
                    "data/step_atypical": n_atypical,
                    "step": global_step,
                }
                if is_typical.sum() > 0 and atypical_mask.sum() > 0:
                    step_log["metrics/center_sep"] = torch.norm(typical_center - atypical_center).item()
                    step_log["metrics/typical_spread"] = torch.norm(typical_repr - typical_center, dim=1).mean().item()
                    step_log["metrics/atypical_spread"] = torch.norm(atypical_repr - atypical_center, dim=1).mean().item()
                    step_log["metrics/aug_consistency"] = (d1 - d2).abs().mean().item()
                wandb.log(step_log)
            global_step += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ctr": f"{l_contrast.item():.4f}",
                "sup": f"{l_supcon.item():.4f}",
                "lsup": f"{l_label_supcon.item():.4f}",
                "csup": f"{l_continuous_supcon.item():.4f}",
                "rnc": f"{l_rnc.item():.4f}",
                "var": f"{l_var.item():.4f}",
                "anc": f"{l_anchor.item():.4f}",
                "mrg": f"{l_margin.item():.4f}",
                "mix": f"{l_mixup.item():.4f}",
                "typ": f"{n_typical}/{len(is_typical)}",
            })

        scheduler.step()
        n = len(train_loader)
        avg_loss = total_loss / n
        avg_contrast = total_contrast / n
        avg_supcon = total_supcon / n
        avg_label_supcon = total_label_supcon / n
        avg_continuous_supcon = total_continuous_supcon / n
        avg_rnc = total_rnc / n
        avg_var = total_var / n
        avg_anchor = total_anchor / n
        avg_margin = total_margin / n
        avg_mixup = total_mixup / n
        avg_emb_var = total_emb_var / n
        mc = max(metric_count, 1)
        avg_center_sep = total_center_sep / mc
        avg_typical_spread = total_typical_spread / mc
        avg_atypical_spread = total_atypical_spread / mc
        avg_aug_consistency = total_aug_consistency / mc

        if is_main_process(rank):
            print(f"Epoch {epoch} - Loss: {avg_loss:.4f} "
                  f"(contrast: {avg_contrast:.4f}, supcon: {avg_supcon:.4f}, "
                  f"label_supcon: {avg_label_supcon:.4f}, "
                  f"continuous_supcon: {avg_continuous_supcon:.4f}, "
                  f"rnc: {avg_rnc:.4f}, "
                  f"var: {avg_var:.4f}, "
                  f"anchor: {avg_anchor:.4f}, margin: {avg_margin:.4f}, "
                  f"mixup: {avg_mixup:.4f}), "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")
            print(f"  Metrics: center_sep={avg_center_sep:.4f}, "
                  f"typical_spread={avg_typical_spread:.4f}, "
                  f"atypical_spread={avg_atypical_spread:.4f}, "
                  f"aug_consistency={avg_aug_consistency:.4f}, "
                  f"emb_var={avg_emb_var:.4f}")
            print(f"  Samples: typical={total_typical_count}, atypical={total_atypical_count}, "
                  f"typical_ratio={total_typical_count / max(total_typical_count + total_atypical_count, 1):.3f}")

            history.append({
                "epoch": epoch,
                "loss": avg_loss,
                "contrast": avg_contrast,
                "supcon": avg_supcon,
                "label_supcon": avg_label_supcon,
                "continuous_supcon": avg_continuous_supcon,
                "rnc": avg_rnc,
                "var": avg_var,
                "anchor": avg_anchor,
                "margin": avg_margin,
                "mixup": avg_mixup,
                "center_separation": avg_center_sep,
                "typical_spread": avg_typical_spread,
                "atypical_spread": avg_atypical_spread,
                "aug_consistency": avg_aug_consistency,
                "embedding_variance": avg_emb_var,
                "typical_count": total_typical_count,
                "atypical_count": total_atypical_count,
                "lr": scheduler.get_last_lr()[0],
            })

            # Log to wandb
            if use_wandb:
                wandb.log({
                    "train/loss": avg_loss,
                    "train/contrast": avg_contrast,
                    "train/supcon": avg_supcon,
                    "train/label_supcon": avg_label_supcon,
                    "train/continuous_supcon": avg_continuous_supcon,
                    "train/rnc": avg_rnc,
                    "train/var": avg_var,
                    "train/anchor": avg_anchor,
                    "train/margin": avg_margin,
                    "train/mixup": avg_mixup,
                    "metrics/center_separation": avg_center_sep,
                    "metrics/typical_spread": avg_typical_spread,
                    "metrics/atypical_spread": avg_atypical_spread,
                    "metrics/spread_ratio": avg_atypical_spread / max(avg_typical_spread, 1e-8),
                    "metrics/aug_consistency": avg_aug_consistency,
                    "metrics/embedding_variance": avg_emb_var,
                    "data/typical_count": total_typical_count,
                    "data/atypical_count": total_atypical_count,
                    "data/typical_ratio": total_typical_count / max(total_typical_count + total_atypical_count, 1),
                    "lr": scheduler.get_last_lr()[0],
                    "epoch": epoch,
                })

            # Eval visualization
            if eval_loader is not None and epoch % args.eval_every == 0:
                evaluate_and_visualize(raw_model, eval_loader, device, epoch, output_dir, use_wandb)

            # Save best
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_dict = {
                    "norm": raw_model.norm.state_dict(),
                    "pre_net": raw_model.pre_net.state_dict(),
                }
                if hasattr(raw_model, "pre_net2"):
                    save_dict["pre_net2"] = raw_model.pre_net2.state_dict()
                torch.save(save_dict, os.path.join(output_dir, "pretrained_prenet.pt"))
                print(f"  -> New best loss: {best_loss:.4f}, saved pretrained_prenet.pt")

            # Save full checkpoint periodically
            if epoch % args.save_every == 0:
                ckpt = {
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "args": vars(args),
                }
                torch.save(ckpt, os.path.join(output_dir, f"checkpoint_epoch_{epoch:03d}.pt"))

        # Synchronize all ranks before next epoch
        if distributed:
            dist.barrier()

    if is_main_process(rank):
        # Save history
        with open(os.path.join(output_dir, "history.json"), 'w') as f:
            json.dump(history, f, indent=2)

        # Save final prenet weights
        save_dict = {
            "norm": raw_model.norm.state_dict(),
            "pre_net": raw_model.pre_net.state_dict(),
        }
        if hasattr(raw_model, "pre_net2"):
            save_dict["pre_net2"] = raw_model.pre_net2.state_dict()
        torch.save(save_dict, os.path.join(output_dir, "pretrained_prenet_final.pt"))

        # Finish wandb run
        if use_wandb:
            wandb.finish()

        print(f"\nTraining complete! Best loss: {best_loss:.4f}")
        print(f"Pre-trained weights saved to: {os.path.join(output_dir, 'pretrained_prenet.pt')}")

    # DDP cleanup
    if distributed:
        cleanup_ddp()

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Contrastive pre-training for pre_net")

    # Config file (JSON) — values are used as defaults, CLI args override
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file. CLI arguments override config values.")

    # Atypical (dysarthric) data
    parser.add_argument("--atypical_feature_dir", type=str,
                        default="/path/to/whisper_features",
                        help="Directory containing pre-extracted atypical features (with split subdirs)")
    parser.add_argument("--atypical_data_dir", type=str,
                        default="./dataset_total",
                        help="Directory containing atypical JSON metadata files")
    parser.add_argument("--atypical_splits", type=str, nargs="+", default=["train"],
                        help="Atypical splits to use (e.g., train dev)")

    # Typical (e.g., LibriSpeech) data — pre-extracted .npy features
    parser.add_argument("--typical_feature_dir", type=str,
                        default=None,
                        help="Directory containing pre-extracted typical features (with split subdirs). "
                             "Optional: omit to rely solely on label12_as_typical for typical samples.")
    parser.add_argument("--typical_data_dir", type=str,
                        default=None,
                        help="Directory containing typical JSON metadata files. "
                             "Optional: omit to rely solely on label12_as_typical for typical samples.")
    parser.add_argument("--typical_splits", type=str, nargs="+",
                        default=["train-clean-100", "train-clean-360", "train-other-500"],
                        help="Typical splits to use")

    # Label-1 as typical
    parser.add_argument("--label12_as_typical", action="store_true",
                        help="Treat atypical samples with severity rating<=2 as typical speech "
                             "for contrastive loss")
    parser.add_argument("--label12_target", type=str, default="Naturalness",
                        choices=["Intelligibility", "Naturalness", "Average"],
                        help="Which rating to check for label12_as_typical")
    parser.add_argument("--label_typical_threshold", type=float, default=2.0,
                        help="Samples with rating <= this value are treated as typical "
                             "(supports float, e.g. 1.5, 2.0, 3.0)")

    # Sampling ratio
    parser.add_argument("--typical_ratio", type=float, default=0.5,
                        help="Target fraction of typical speech per batch (0.0-1.0). "
                             "0.5 = balanced 50/50. Used only by sampler_type=binary.")
    parser.add_argument("--sampler_type", type=str, default="binary",
                        choices=["random", "binary", "group"],
                        help="Sampling strategy: "
                             "'random' = standard shuffle (no reweighting); "
                             "'binary' = WeightedRandomSampler by typical/atypical flag (current default); "
                             "'group' = WeightedRandomSampler by fine-grained severity group "
                             "(1/group_count per sample, ensures equal group representation per batch).")

    # Model
    parser.add_argument("--model_version", type=str, default="v1",
                        choices=["v1", "v2"],
                        help="v1: single pre_net + mean pooling, "
                             "v2: two pre_net layers + statistics pooling (mean+std)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate for ContrastiveModelV2 (ignored for v1)")
    parser.add_argument("--input_dim", type=int, default=1280,
                        help="Input feature dimension (Whisper large = 1280)")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="pre_net output dimension")
    parser.add_argument("--proj_dim", type=int, default=128,
                        help="Projection head output dimension")

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=1)

    # Loss weights
    parser.add_argument("--w_contrast", type=float, default=1.0,
                        help="Weight for SimCLR NT-Xent loss")
    parser.add_argument("--w_supcon", type=float, default=0.0,
                        help="Weight for supervised contrastive loss (typical vs atypical)")
    parser.add_argument("--w_label_supcon", type=float, default=0.0,
                        help="Weight for fine-grained label supervised contrastive loss "
                             "(groups samples by binned severity rating)")
    parser.add_argument("--w_continuous_supcon", type=float, default=0.0,
                        help="Weight for continuous-label supervised contrastive loss. "
                             "Treats pairs as positive if |rᵢ − rⱼ| < --continuous_radius, "
                             "avoiding hard bin-boundary artefacts. "
                             "Cannot be combined with w_supcon.")
    parser.add_argument("--continuous_radius", type=float, default=1.0,
                        help="Rating-difference threshold for continuous_supcon: "
                             "pairs with |rᵢ − rⱼ| < radius are treated as positive "
                             "(default 1.0 = within one rating point on the 1-7 scale)")
    parser.add_argument("--label_supcon_target", type=str, default="Average",
                        choices=["Intelligibility", "Naturalness", "Average"],
                        help="Which rating to use for fine-grained label supcon grouping")
    parser.add_argument("--label_supcon_typical_group", type=int, default=0,
                        help="Group index assigned to typical speech for label_supcon. "
                             "Internally converted to rating_value = group + 1.0 so that "
                             "rating_to_group round-trips correctly (group 0 → rating 1.0, "
                             "group 1 → rating 2.0, etc.). "
                             "-1 = exclude typical speech from this loss entirely.")
    parser.add_argument("--w_rnc", type=float, default=0.0,
                        help="Weight for Rank-N-Contrast (RNC) loss. "
                             "Ranking-based contrastive loss for continuous labels: "
                             "for each (anchor i, candidate j), samples with larger label "
                             "distance to i than j serve as negatives. "
                             "Cannot be combined with w_supcon. "
                             "Uses --tau for temperature (paper default 2.0 with l2 sim).")
    parser.add_argument("--rnc_feature_sim", type=str, default="l2",
                        choices=["l2", "cosine"],
                        help="Feature similarity for RNC: "
                             "'l2' = negative L2 distance (paper default), "
                             "'cosine' = cosine similarity")
    parser.add_argument("--w_var", type=float, default=0.1,
                        help="Weight for VICReg variance loss")
    parser.add_argument("--w_anchor", type=float, default=0.5,
                        help="Weight for anchor-distance consistency loss")
    parser.add_argument("--w_margin", type=float, default=0.5,
                        help="Weight for margin separation loss")
    parser.add_argument("--w_mixup", type=float, default=0.5,
                        help="Weight for mixup distance smoothness loss")
    parser.add_argument("--margin", type=float, default=2.0,
                        help="Margin for typical-atypical centroid separation")

    # Loss hyperparameters
    parser.add_argument("--tau", type=float, default=0.1,
                        help="Temperature for NT-Xent loss")
    parser.add_argument("--vicreg_gamma", type=float, default=1.0,
                        help="Target std for VICReg variance regularization")

    # Augmentation
    parser.add_argument("--aug_noise_std", type=float, default=0.01,
                        help="Gaussian noise std for augmentation")
    parser.add_argument("--aug_mask_ratio", type=float, default=0.2,
                        help="Max ratio of time frames to mask")
    parser.add_argument("--aug_crop_min_ratio", type=float, default=0.7,
                        help="Minimum crop ratio for random cropping")

    # wandb
    parser.add_argument("--wandb", action="store_true",
                        help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="contrastive-pretrain",
                        help="wandb project name")

    # Evaluation / visualization
    parser.add_argument("--eval_every", type=int, default=5,
                        help="Run eval visualization every N epochs (0 to disable)")
    parser.add_argument("--eval_split", type=str, default="dev",
                        help="Atypical split to use for eval visualization (default: dev)")
    parser.add_argument("--eval_typical_splits", type=str, nargs="+",
                        default=["dev-other"],
                        help="Typical splits to use for eval visualization (default: dev-other)")
    parser.add_argument("--eval_typical_samples", type=int, default=500,
                        help="Max total typical samples for eval (subsampled for speed)")

    # Output
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./experiments")
    parser.add_argument("--device", type=str, default="cuda")

    # First parse to get --config path
    args, remaining = parser.parse_known_args()

    # Load config file and set as defaults (CLI args override)
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = json.load(f)
        parser.set_defaults(**config)
        # Re-parse with config defaults applied
        args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
