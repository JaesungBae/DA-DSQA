#!/usr/bin/env python3
"""
Training script for regression probes on pre-extracted Whisper features.

This script trains a regression probe on top of pre-extracted Whisper encoder
features (.npy files) to predict speech quality metrics (intelligibility or
naturalness) for pathological speech.

Uses HuggingFace Trainer with Huber loss and weighted random sampling.
"""

import os
import json
import math
import random
import argparse
import numpy as np
from scipy.stats import spearmanr, pearsonr

# Force single-GPU mode: clear SLURM env vars that trick accelerate
# into launching distributed mode on single-GPU jobs
for _k in list(os.environ):
    if _k.startswith("SLURM"):
        del os.environ[_k]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import TrainingArguments, Trainer
from transformers.modeling_outputs import SequenceClassifierOutput

from utils.losses import (
    huber_loss,
    compute_speaker_level_variance_loss,
    compute_sample_weights,
)
from collections import Counter

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ======================
# Dataset
# ======================

class FeatureRegressionDataset(Dataset):
    """
    Dataset that loads pre-extracted .npy features and regression labels from JSON metadata.

    Each sample is a dict with keys compatible with HuggingFace Trainer:
        - input_values: (T, hidden_dim) feature tensor
        - length: int, sequence length
        - labels: float, regression target
        - filename: str, for speaker ID extraction
    """

    def __init__(self, feature_dir, metadata_path, target_type="intelligibility", task="regression",
                 label_field=None, hc_fallback=None):
        """
        Args:
            feature_dir: Directory containing .npy feature files
            metadata_path: Path to JSON metadata with ratings
            target_type: "intelligibility" or "naturalness"
            task: "regression" (float labels 1-7) or "classification" (int labels 0-6)
            label_field: If set, read label directly from info[label_field] instead of
                         info["ratings"][target_type]. Used for external datasets with
                         their own native label fields (e.g. "mos", "dysarthria_severity").
            hc_fallback: If label_field value is None and info["group"] == "HC", use this
                         value as the label (e.g. 0.0 for zenodo H-Y stage).
        """
        self.feature_dir = feature_dir
        self.target_type = target_type
        self.task = task

        # Map target_type to JSON key(s)
        if target_type == "average":
            self.target_keys = ["Intelligibility", "Naturalness", "Average"]
        else:
            self.target_keys = ["Intelligibility" if target_type == "intelligibility" else "Naturalness"]

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        self.samples = []
        num_missing = 0
        num_no_label = 0
        for filename, info in metadata.items():
            if filename.startswith("_"):
                continue

            feat_name = os.path.splitext(filename)[0] + ".npy"
            feat_path = os.path.join(feature_dir, feat_name)

            if label_field is not None:
                val = info.get(label_field)
                if val is None and hc_fallback is not None and info.get("group") == "HC":
                    val = hc_fallback
                if val is None:
                    num_no_label += 1
                    continue
                label_value = float(val)
            else:
                if "ratings" not in info:
                    continue
                values = [info["ratings"].get(k) for k in self.target_keys]
                available = [v for v in values if v is not None]
                if not available:
                    num_no_label += 1
                    continue
                label_value = sum(available) / len(available)

            if task == "classification":
                label = int(round(label_value)) - 1  # 1-7 → 0-6
            else:
                label = float(label_value)
            self.samples.append({
                "feat_path": feat_path,
                "label": label,
                "filename": filename,
            })

        print(f"Loaded {len(self.samples)} samples from {metadata_path} (target: {target_type})")
        if num_missing > 0:
            print(f"  WARNING: {num_missing} entries skipped (feature files not found in {feature_dir})")
        if num_no_label > 0:
            print(f"  WARNING: {num_no_label} entries skipped (no {target_type} label)")

    def get_labels(self):
        """Return list of labels (for weighted sampling)."""
        return [s["label"] for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = np.load(sample["feat_path"])  # (T, hidden_dim)
        label_dtype = torch.long if self.task == "classification" else torch.float32
        return {
            "input_values": torch.from_numpy(features).float(),
            "labels": torch.tensor(sample["label"], dtype=label_dtype),
            "filename": sample["filename"],
        }


# ======================
# Collate Function
# ======================

class CollateFn:
    """
    Callable class that pads variable-length .npy features into a batch.

    Using a class (instead of a closure) makes it picklable, which is required
    by PyTorch DataLoader when num_workers > 0.
    """

    def __init__(self, extract_speaker_ids=False):
        self.extract_speaker_ids = extract_speaker_ids

    def __call__(self, batch):
        features = [b["input_values"] for b in batch]
        labels = torch.stack([b["labels"] for b in batch])

        # Pad to max length in batch
        max_len = max(f.shape[0] for f in features)
        hidden_dim = features[0].shape[1]
        batch_size = len(features)

        padded = torch.zeros(batch_size, max_len, hidden_dim)
        lengths = torch.zeros(batch_size, dtype=torch.long)
        for i, f in enumerate(features):
            length = f.shape[0]
            padded[i, :length] = f
            lengths[i] = length

        result = {
            "input_values": padded,
            "lengths": lengths,
            "labels": labels,
        }

        if self.extract_speaker_ids:
            speaker_ids = []
            for b in batch:
                filename = b["filename"].replace('.wav', '')
                speaker_ids.append(filename.split('_')[0])
            result["speaker_ids"] = speaker_ids

        return result


def make_collate_fn(extract_speaker_ids=False):
    return CollateFn(extract_speaker_ids=extract_speaker_ids)


# ======================
# Probe Model
# ======================

class WhisperFeatureProbeV2(nn.Module):
    """
    Regression probe matching ContrastiveModelV2's backbone.

    Architecture: LayerNorm → Linear → ReLU → Dropout → Linear → ReLU → Dropout
                  → Statistics Pooling (mean+std) → Linear(proj_dim*2, 1)
    """

    def __init__(self, input_dim=1280, proj_dim=256, dropout=0.1, num_classes=1):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.projector = nn.Linear(input_dim, proj_dim)
        self.projector2 = nn.Linear(proj_dim, proj_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(proj_dim * 2, num_classes)

    def forward(self, input_values, lengths=None, **kwargs):
        """
        Args:
            input_values: (batch, time, hidden_dim) pre-extracted features
            lengths: (batch,) actual sequence lengths
        Returns:
            SequenceClassifierOutput with logits of shape (batch, 1)
            and hidden_states of shape (batch, proj_dim*2)
        """
        x = self.norm(input_values)
        x = self.dropout(self.relu(self.projector(x)))   # (B, T, proj_dim)
        x = self.dropout(self.relu(self.projector2(x)))   # (B, T, proj_dim)

        # Statistics pooling (mean + std) with masking
        if lengths is not None:
            batch_size, max_len, _ = x.shape
            mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            mask_f = mask.unsqueeze(-1).float()
            x_masked = x * mask_f
            lengths_f = lengths.unsqueeze(1).float().clamp(min=1)
            mean = x_masked.sum(dim=1) / lengths_f
            var = (x_masked ** 2).sum(dim=1) / lengths_f - mean ** 2
            std = var.clamp(min=1e-8).sqrt()
        else:
            mean = x.mean(dim=1)
            std = x.std(dim=1)

        pooled = torch.cat([mean, std], dim=1)  # (B, proj_dim*2)
        logits = self.classifier(pooled)  # (B, 1)

        return SequenceClassifierOutput(logits=logits, hidden_states=pooled)


# ======================
# Metrics Computation
# ======================

def build_compute_metrics():
    """
    Build a metrics computation function for evaluation.

    Computes SRCC, PCC, MAE, RMSE on the original [1, 7] scale.
    """
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.array(preds).reshape(-1)
        labels = np.array(labels).reshape(-1)
        # Clip predictions to valid range [1.0, 7.0]
        preds = np.clip(preds, 1.0, 7.0)

        srcc = spearmanr(labels, preds).correlation
        pr = pearsonr(labels, preds)
        pcc = getattr(pr, "statistic", pr[0])

        mae = float(np.mean(np.abs(preds - labels)))
        rmse = float(math.sqrt(np.mean((preds - labels) ** 2)))

        return {
            "srcc": float(srcc),
            "pcc": float(pcc),
            "mae": mae,
            "rmse": rmse,
        }
    return compute_metrics


# ======================
# Classification Utilities
# ======================

class LogitAdjustedLoss(nn.Module):
    """Logit-Adjusted Loss for long-tailed recognition."""

    def __init__(self, cls_num_list, tau=1.0):
        super().__init__()
        cls_num_list = torch.tensor(cls_num_list, dtype=torch.float)
        cls_num_ratio = cls_num_list / cls_num_list.sum()
        log_cls_num = torch.log(cls_num_ratio)
        self.register_buffer("log_cls_num", log_cls_num)
        self.tau = tau

    def forward(self, logit, target):
        logit_adjusted = logit + self.tau * self.log_cls_num.unsqueeze(0)
        return F.cross_entropy(logit_adjusted, target)


def get_class_distribution(dataset, num_classes):
    """Get class sample counts from dataset (labels are 0-indexed)."""
    labels = [s["label"] for s in dataset.samples]
    class_counts = Counter(labels)
    return [class_counts.get(i, 1) for i in range(num_classes)]


def build_compute_metrics_classification(num_classes=7):
    """Build metrics for classification: UA (unweighted accuracy) and WA (weighted accuracy)."""
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(np.array(logits), axis=-1).reshape(-1)
        labels = np.array(labels).reshape(-1).astype(int)

        wa = float((preds == labels).mean())

        class_accs = []
        for c in range(num_classes):
            mask = labels == c
            if mask.sum() > 0:
                class_accs.append(float((preds[mask] == c).mean()))
        ua = float(np.mean(class_accs)) if class_accs else 0.0

        return {"ua": ua, "wa": wa}
    return compute_metrics


# ======================
# Custom Trainer
# ======================

class HuberTrainer(Trainer):
    """
    Custom Trainer that uses Huber loss and weighted random sampling.

    - Uses Huber loss instead of MSE for robust regression
    - Implements weighted random sampling to balance class distribution
    - Optionally supports speaker-level regularization
    """

    def __init__(self, *args, sampler_labels=None, speaker_reg_lambda=0.0,
                 task="regression", cls_criterion=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._sampler_labels = sampler_labels
        self.speaker_reg_lambda = speaker_reg_lambda
        self.task = task
        self.cls_criterion = cls_criterion

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        speaker_ids = inputs.pop("speaker_ids", None)
        # Remove non-model keys
        inputs.pop("filename", None)

        outputs = model(**inputs)
        logits = outputs.logits

        if self.task == "classification":
            # Ensure criterion buffers are on the right device
            if self.cls_criterion.log_cls_num.device != logits.device:
                self.cls_criterion = self.cls_criterion.to(logits.device)
            loss = self.cls_criterion(logits, labels)
        else:
            logits = logits.squeeze(-1)  # [B]
            utterance_loss = huber_loss(logits, labels, delta=0.5)

            speaker_loss = torch.tensor(0.0, device=logits.device)
            if speaker_ids is not None and self.speaker_reg_lambda > 0:
                speaker_loss = compute_speaker_level_variance_loss(logits, speaker_ids)

            loss = utterance_loss + self.speaker_reg_lambda * speaker_loss

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        labels = inputs.pop("labels")
        inputs.pop("speaker_ids", None)
        inputs.pop("filename", None)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            if self.task == "classification":
                loss = F.cross_entropy(logits, labels)
            else:
                logits = logits.squeeze(-1)
                loss = huber_loss(logits, labels, delta=0.5)

        return (loss, logits, labels)

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        if self.args.world_size > 1:
            return super().get_train_dataloader()

        # For regression: labels are [1, 7] floats → map to [0, 6]
        # For classification: labels are already [0, 6] ints
        if self.task == "classification":
            labels_all = np.asarray(self._sampler_labels, dtype=np.int64)
        else:
            labels_all = np.asarray(self._sampler_labels, dtype=np.int64) - 1
        _, sample_weights = compute_sample_weights(labels_all, num_classes=7)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=True,
            prefetch_factor=4,
        )


# ======================
# Main Training Function
# ======================

def main():
    parser = argparse.ArgumentParser(
        description="Train a regression probe on pre-extracted Whisper features."
    )

    # Data arguments
    parser.add_argument(
        "--feature_dir",
        type=str,
        default="/path/to/whisper_features",
        help="Root directory containing pre-extracted features (with train/dev subdirs)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./dataset_labeled",
        help="Directory containing JSON metadata files (train.json, dev.json)"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./experiments/probe",
        help="Output directory for saving checkpoints and models"
    )
    parser.add_argument(
        "--target_type",
        type=str,
        default="intelligibility",
        choices=["intelligibility", "naturalness", "average"],
        help="Target metric to predict"
    )

    # Task arguments
    parser.add_argument(
        "--task",
        type=str,
        default="regression",
        choices=["regression", "classification"],
        help="Task type: regression (Huber loss, SRCC metric) or "
             "classification (LA loss, UA metric)"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=7,
        help="Number of classes for classification task"
    )

    # Model arguments
    parser.add_argument(
        "--input_dim",
        type=int,
        default=1280,
        help="Input feature dimension (1280 for whisper-large-v2/v3)"
    )
    parser.add_argument(
        "--proj_dim",
        type=int,
        default=320,
        help="Dimension of the projection layer"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate"
    )

    # Training arguments
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=16,
        help="Micro batch size per device"
    )
    parser.add_argument(
        "--accum_steps",
        type=int,
        default=2,
        help="Gradient accumulation steps (effective batch size = micro_batch_size * accum_steps)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs"
    )

    # Checkpointing arguments
    parser.add_argument(
        "--save_every_steps",
        type=int,
        default=50,
        help="Save checkpoint every N optimization steps"
    )
    parser.add_argument(
        "--eval_every_steps",
        type=int,
        default=50,
        help="Evaluate every N optimization steps"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (or 'latest' for latest checkpoint)"
    )

    # Regularization arguments
    parser.add_argument(
        "--speaker_reg_lambda",
        type=float,
        default=0.0,
        help="Weight for speaker-level regularization (0.0 to disable)"
    )

    # Experiment name
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Experiment name for output subdirectory. If not set, uses target_type only."
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (affects weight init, sampler, etc.)"
    )

    # Save/eval strategy
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="epoch",
        choices=["steps", "epoch"],
        help="Checkpoint saving strategy. 'epoch' saves after every epoch (recommended for "
             "multi-seed comparison); 'steps' uses --save_every_steps and --eval_every_steps."
    )

    # Pre-trained pre_net from contrastive pre-training
    parser.add_argument(
        "--pretrained_prenet",
        type=str,
        default=None,
        help="Path to pretrained_prenet.pt from contrastive pre-training. "
             "Loads norm + projector weights and freezes them by default."
    )
    parser.add_argument(
        "--finetune_prenet",
        action="store_true",
        default=False,
        help="Finetune the loaded norm + projector parameters instead of freezing them."
    )

    # Wandb arguments
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable wandb logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="probe-whisper",
        help="wandb project name"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="wandb run name (defaults to exp_name or auto-generated)"
    )

    # Test mode arguments
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="Only run evaluation on test sets (requires --checkpoint)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory for testing (contains model.safetensors)"
    )
    parser.add_argument(
        "--eval_splits",
        nargs="+",
        default=["dev", "test"],
        help="Splits to evaluate on (default: dev test)"
    )

    args = parser.parse_args()

    if args.test_only:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for --test_only mode")
        test(args)
    else:
        train_model(args)




def _evaluate_splits(model, device, eval_splits, target_type, compute_metrics,
                     task="regression", label_field=None, hc_fallback=None):
    """
    Run inference on a list of (display_name, feature_dir, json_path) tuples.

    Returns:
        results: dict {display_name: metrics}
        split_data_for_plot: dict {display_name: (preds, labels), display_name+"_embeds": embeds,
                                   display_name+"_filenames": filenames}
    """
    collate_fn = make_collate_fn(extract_speaker_ids=False)
    is_cls = task == "classification"

    results = {}
    split_data_for_plot = {}

    for display_name, feat_dir, json_path in eval_splits:
        if not os.path.exists(json_path):
            print(f"Skipping {display_name} - {json_path} not found")
            continue
        if not os.path.exists(feat_dir):
            print(f"Skipping {display_name} - {feat_dir} not found")
            continue

        dataset = FeatureRegressionDataset(
            feat_dir, json_path, target_type=target_type, task=task,
            label_field=label_field, hc_fallback=hc_fallback,
        )
        if len(dataset) == 0:
            print(f"Skipping {display_name} - no samples loaded")
            continue

        loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        all_preds = []
        all_labels = []
        all_embeddings = []
        with torch.no_grad():
            for batch in loader:
                input_values = batch["input_values"].to(device)
                lengths = batch["lengths"].to(device)
                labels = batch["labels"]

                outputs = model(input_values=input_values, lengths=lengths)
                if is_cls:
                    preds = outputs.logits.cpu().numpy()  # (B, num_classes)
                else:
                    preds = outputs.logits.squeeze(-1).cpu().numpy()
                embeds = outputs.hidden_states.cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
                all_embeddings.extend(embeds)

        # Compute metrics
        all_preds_arr = np.array(all_preds)
        all_labels_arr = np.array(all_labels)
        eval_pred = (all_preds_arr, all_labels_arr)
        metrics = compute_metrics(eval_pred)

        print(f"\n{display_name} ({len(dataset)} samples):")
        if is_cls:
            print(f"  UA: {metrics['ua']:.4f}")
            print(f"  WA: {metrics['wa']:.4f}")

            # Per-class breakdown
            pred_classes = np.argmax(all_preds_arr, axis=-1).reshape(-1)
            labels_flat = all_labels_arr.reshape(-1).astype(int)
            unique_labels = sorted(np.unique(labels_flat))

            print(f"\n  Per-class breakdown:")
            print(f"  {'Class':>5}  {'Count':>5}  {'Acc':>6}  {'Predicted':>9}")
            per_label = {}
            for lbl in unique_labels:
                mask = labels_flat == lbl
                n = int(mask.sum())
                acc = float((pred_classes[mask] == lbl).mean()) if n > 0 else 0.0
                n_pred = int((pred_classes == lbl).sum())
                print(f"  {lbl:>5}  {n:>5}  {acc:>6.3f}  {n_pred:>9}")
                per_label[str(lbl)] = {"count": n, "accuracy": acc, "n_predicted": n_pred}
            metrics["per_label"] = per_label
            # For plots, store predicted class
            plot_preds = pred_classes.astype(float)
            plot_labels = labels_flat.astype(float)
        else:
            print(f"  SRCC: {metrics['srcc']:.4f}")
            print(f"  PCC:  {metrics['pcc']:.4f}")
            print(f"  MAE:  {metrics['mae']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.4f}")

            # Per-label metrics
            preds_clipped = np.clip(all_preds_arr.reshape(-1), 1.0, 7.0)
            labels_flat = all_labels_arr.reshape(-1)
            unique_labels = sorted(np.unique(labels_flat).astype(int))

            print(f"\n  Per-label breakdown:")
            print(f"  {'Label':>5}  {'Count':>5}  {'MAE':>6}  {'RMSE':>6}  {'MeanPred':>8}")
            per_label = {}
            for lbl in unique_labels:
                mask = labels_flat.astype(int) == lbl
                n = int(mask.sum())
                p = preds_clipped[mask]
                l = labels_flat[mask]
                lbl_mae = float(np.mean(np.abs(p - l)))
                lbl_rmse = float(math.sqrt(np.mean((p - l) ** 2)))
                lbl_mean_pred = float(np.mean(p))
                print(f"  {lbl:>5}  {n:>5}  {lbl_mae:>6.3f}  {lbl_rmse:>6.3f}  {lbl_mean_pred:>8.3f}")
                per_label[str(lbl)] = {
                    "count": n, "mae": lbl_mae, "rmse": lbl_rmse, "mean_pred": lbl_mean_pred,
                }
            metrics["per_label"] = per_label
            plot_preds = preds_clipped
            plot_labels = labels_flat

        # Filenames in dataset order (shuffle=False, so order matches predictions)
        all_filenames = [s["filename"] for s in dataset.samples]

        results[display_name] = metrics
        split_data_for_plot[display_name] = (plot_preds, plot_labels)
        split_data_for_plot[display_name + "_embeds"] = np.array(all_embeddings)
        split_data_for_plot[display_name + "_filenames"] = all_filenames

    return results, split_data_for_plot


def _save_predictions(split_data_for_plot, results, target_type, output_dir, filename_suffix=""):
    """Save per-sample predictions and aggregate metrics to a JSON file."""
    split_names = [s for s in split_data_for_plot if not s.endswith(("_embeds", "_filenames"))]
    if not split_names:
        return

    output = {}
    for split_name in split_names:
        preds, labels = split_data_for_plot[split_name]
        filenames = split_data_for_plot.get(split_name + "_filenames", [])

        samples = []
        for i in range(len(preds)):
            entry = {
                "prediction": float(preds[i]),
                "label": float(labels[i]),
            }
            if i < len(filenames):
                entry["filename"] = filenames[i]
            samples.append(entry)

        output[split_name] = {
            "metrics": results.get(split_name, {}),
            "predictions": samples,
        }

    suffix = f"_{filename_suffix}" if filename_suffix else ""
    pred_path = os.path.join(output_dir, f"predictions_{target_type}{suffix}.json")
    with open(pred_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Predictions saved to {pred_path}")


def _save_plots(split_data_for_plot, target_type, output_dir, filename_suffix=""):
    """Generate and save boxplot and t-SNE plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    split_names = [s for s in split_data_for_plot if not s.endswith(("_embeds", "_filenames"))]
    n_splits = len(split_names)
    if n_splits == 0:
        return

    # Boxplot
    fig, axes = plt.subplots(1, n_splits, figsize=(6 * n_splits, 5), squeeze=False)
    axes = axes[0]
    for ax, split_name in zip(axes, split_names):
        preds, labels = split_data_for_plot[split_name]
        unique_labels = sorted(np.unique(labels).astype(int))
        data_per_label = [preds[labels.astype(int) == lbl] for lbl in unique_labels]
        counts = [len(d) for d in data_per_label]

        bp = ax.boxplot(data_per_label, positions=unique_labels, widths=0.6,
                        patch_artist=True, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='red', markersize=5))
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')

        ax.plot([unique_labels[0] - 0.5, unique_labels[-1] + 0.5],
                [unique_labels[0] - 0.5, unique_labels[-1] + 0.5],
                'r--', alpha=0.5, label='Perfect')

        ax.set_xlabel("True Label")
        ax.set_ylabel("Predicted Value")
        ax.set_title(f"{split_name} (n={sum(counts)})")
        ax.set_xticks(unique_labels)
        ax.set_xticklabels([f"{lbl}\n(n={c})" for lbl, c in zip(unique_labels, counts)])
        ax.set_ylim(0.5, 7.5)
        ax.legend(loc='upper left')

    fig.suptitle(f"Per-Label Prediction Distribution — {target_type}", fontsize=14)
    fig.tight_layout()
    suffix = f"_{filename_suffix}" if filename_suffix else ""
    plot_path = os.path.join(output_dir, f"boxplot_{target_type}{suffix}.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Boxplot saved to {plot_path}")

    # t-SNE
    from sklearn.manifold import TSNE

    fig, axes = plt.subplots(1, n_splits, figsize=(6 * n_splits, 5), squeeze=False)
    axes = axes[0]
    cmap = plt.cm.get_cmap("RdYlGn", 7)

    for ax, split_name in zip(axes, split_names):
        embeds = split_data_for_plot[split_name + "_embeds"]
        _, labels = split_data_for_plot[split_name]
        labels_int = labels.astype(int)

        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        coords = tsne.fit_transform(embeds)

        scatter = ax.scatter(coords[:, 0], coords[:, 1],
                             c=labels_int, cmap=cmap, vmin=1, vmax=7,
                             s=15, alpha=0.7, edgecolors='none')
        ax.set_title(f"{split_name} (n={len(labels)})")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

    fig.suptitle(f"t-SNE of Intermediate Representations — {target_type}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    fig.colorbar(scatter, cax=cbar_ax, ticks=range(1, 8),
                 label=target_type.capitalize())
    tsne_path = os.path.join(output_dir, f"tsne_{target_type}{suffix}.png")
    fig.savefig(tsne_path, dpi=150)
    plt.close(fig)
    print(f"t-SNE plot saved to {tsne_path}")


def test(args):
    """Evaluate a trained probe on test sets."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_cls = args.task == "classification"
    num_classes = args.num_classes if is_cls else 1

    # Create model and load checkpoint
    print(f"Loading model from {args.checkpoint}...")
    model = WhisperFeatureProbeV2(
        input_dim=args.input_dim,
        proj_dim=args.proj_dim,
        dropout=args.dropout,
        num_classes=num_classes,
    )

    # Load weights (safetensors or pytorch bin)
    safe_path = os.path.join(args.checkpoint, "model.safetensors")
    bin_path = os.path.join(args.checkpoint, "pytorch_model.bin")
    if os.path.isfile(safe_path):
        from safetensors.torch import load_file
        state_dict = load_file(safe_path, device=str(device))
    elif os.path.isfile(bin_path):
        state_dict = torch.load(bin_path, map_location=device)
    else:
        raise FileNotFoundError(
            f"No model.safetensors or pytorch_model.bin found in {args.checkpoint}"
        )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    if is_cls:
        compute_metrics = build_compute_metrics_classification(args.num_classes)
    else:
        compute_metrics = build_compute_metrics()

    # ---- Evaluate on specified splits ----
    eval_splits = []
    for split in args.eval_splits:
        eval_splits.append((
            split,
            os.path.join(args.feature_dir, split),
            os.path.join(args.data_dir, f"{split}.json"),
        ))

    print("\n" + "=" * 60)
    print("Evaluation")
    print("=" * 60)
    results, split_data_for_plot = _evaluate_splits(
        model, device, eval_splits, args.target_type, compute_metrics,
        task=args.task,
    )

    # Save results — use data_dir basename to distinguish datasets
    data_name = os.path.basename(os.path.normpath(args.data_dir))
    results_path = os.path.join(args.checkpoint, f"eval_{data_name}_{args.target_type}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    _save_predictions(split_data_for_plot, results, args.target_type, args.checkpoint, filename_suffix=data_name)
    _save_plots(split_data_for_plot, args.target_type, args.checkpoint, filename_suffix=data_name)

    return results


def train_model(args):
    """Train the probe model."""
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f"Random seed set to {args.seed}")

    is_cls = args.task == "classification"
    num_classes = args.num_classes if is_cls else 1

    # Load datasets
    print("Loading datasets...")
    train_data = FeatureRegressionDataset(
        os.path.join(args.feature_dir, "train"),
        os.path.join(args.data_dir, "train.json"),
        target_type=args.target_type,
        task=args.task,
    )
    dev_data = FeatureRegressionDataset(
        os.path.join(args.feature_dir, "dev"),
        os.path.join(args.data_dir, "dev.json"),
        target_type=args.target_type,
        task=args.task,
    )

    # Create collate function
    extract_speaker_ids = args.speaker_reg_lambda > 0
    collate_fn = make_collate_fn(extract_speaker_ids=extract_speaker_ids)

    # Set up output directory
    if args.exp_name:
        out_dir = os.path.join(args.out_dir, args.exp_name, args.target_type)
    else:
        out_dir = os.path.join(args.out_dir, args.target_type)

    # Create model
    print(f"Creating probe model (input_dim={args.input_dim}, proj_dim={args.proj_dim}, "
          f"task={args.task}, num_classes={num_classes})...")
    model = WhisperFeatureProbeV2(
        input_dim=args.input_dim,
        proj_dim=args.proj_dim,
        dropout=args.dropout,
        num_classes=num_classes,
    )

    # Load pre-trained norm + projector weights if provided
    if args.pretrained_prenet:
        print(f"\nLoading pre-trained pre_net from: {args.pretrained_prenet}")
        prenet_ckpt = torch.load(args.pretrained_prenet, map_location="cpu", weights_only=True)
        if "norm" in prenet_ckpt and "pre_net" in prenet_ckpt:
            # Direct format (pretrained_prenet.pt): norm + pre_net dicts
            model.norm.load_state_dict(prenet_ckpt["norm"])
            model.projector.load_state_dict(prenet_ckpt["pre_net"])
            if "pre_net2" in prenet_ckpt:
                model.projector2.load_state_dict(prenet_ckpt["pre_net2"])
        elif "model_state_dict" in prenet_ckpt:
            # Full checkpoint format (checkpoint_epoch_XXX.pt from train_regression.py)
            full_sd = prenet_ckpt["model_state_dict"]
            norm_sd = {k.replace("norm.", "", 1): v for k, v in full_sd.items() if k.startswith("norm.")}
            pre_net_sd = {k.replace("pre_net.", "", 1): v for k, v in full_sd.items()
                          if k.startswith("pre_net.") and not k.startswith("pre_net2.")}
            model.norm.load_state_dict(norm_sd)
            model.projector.load_state_dict(pre_net_sd)
            pre_net2_sd = {k.replace("pre_net2.", "", 1): v for k, v in full_sd.items()
                           if k.startswith("pre_net2.")}
            if pre_net2_sd:
                model.projector2.load_state_dict(pre_net2_sd)
        else:
            raise ValueError(f"Unrecognized checkpoint format. Keys: {list(prenet_ckpt.keys())}")

        if args.finetune_prenet:
            print("  Loaded norm + projector parameters (finetuning enabled)")
        else:
            for param in model.norm.parameters():
                param.requires_grad = False
            for param in model.projector.parameters():
                param.requires_grad = False
            if hasattr(model, "projector2"):
                for param in model.projector2.parameters():
                    param.requires_grad = False
            print("  Loaded and froze norm + projector parameters")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")
    print(model)

    # Set up classification loss if needed
    cls_criterion = None
    if is_cls:
        cls_num_list = get_class_distribution(train_data, args.num_classes)
        print(f"Class distribution: {cls_num_list}")
        print(f"Imbalance ratio: {max(cls_num_list) / max(min(cls_num_list), 1):.2f}")
        cls_criterion = LogitAdjustedLoss(cls_num_list)

    # Initialize wandb
    use_wandb = args.wandb and HAS_WANDB
    if use_wandb:
        run_name = args.wandb_run_name or args.exp_name or f"{args.task}_{args.target_type}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            dir=out_dir,
        )
    elif args.wandb and not HAS_WANDB:
        print("WARNING: --wandb flag set but wandb is not installed. Skipping.")

    # Training arguments
    best_metric = "ua" if is_cls else "srcc"
    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=args.micro_batch_size,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=args.accum_steps,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        seed=args.seed,

        fp16=torch.cuda.is_available(),

        eval_strategy=args.save_strategy,
        eval_steps=args.eval_every_steps if args.save_strategy == "steps" else None,
        save_strategy=args.save_strategy,
        save_steps=args.save_every_steps if args.save_strategy == "steps" else None,
        save_total_limit=1,

        load_best_model_at_end=True,
        metric_for_best_model=best_metric,
        greater_is_better=True,

        logging_steps=10,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to=["wandb"] if use_wandb else [],
    )

    # Get sampler labels (integer ratings for weighted sampling)
    sampler_labels = train_data.get_labels()

    # Select metrics function
    if is_cls:
        metrics_fn = build_compute_metrics_classification(args.num_classes)
    else:
        metrics_fn = build_compute_metrics()

    # Initialize trainer
    trainer = HuberTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dev_data,
        data_collator=collate_fn,
        compute_metrics=metrics_fn,
        sampler_labels=sampler_labels,
        speaker_reg_lambda=args.speaker_reg_lambda,
        task=args.task,
        cls_criterion=cls_criterion,
    )

    # Train
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            checkpoints = [d for d in os.listdir(out_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                latest_ckpt = max(checkpoints, key=lambda x: int(x.split("-")[1]))
                resume_path = os.path.join(out_dir, latest_ckpt)
            else:
                raise ValueError(f"No checkpoints found in {out_dir}")
        else:
            resume_path = args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=resume_path)
    else:
        trainer.train()

    # Save best model
    trainer.save_model(training_args.output_dir)

    # Save best metrics
    if trainer.state.best_metric is not None:
        metrics_path = os.path.join(training_args.output_dir, "best_metrics.json")
        metadata = {
            "best_metric": float(trainer.state.best_metric),
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "metric_for_best_model": training_args.metric_for_best_model,
            "greater_is_better": training_args.greater_is_better,
            "speaker_reg_lambda": args.speaker_reg_lambda,
        }
        with open(metrics_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Best metrics saved to {metrics_path}")

    # Finish wandb run
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
