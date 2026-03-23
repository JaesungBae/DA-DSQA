#!/usr/bin/env python3
"""
Pseudo-label unlabeled data using a trained probe model.

Loads a trained WhisperFeatureProbeV2 checkpoint, runs inference on
all samples in dataset_total that lack ratings for the target type, and saves
(or merges into) a dataset JSON with pseudo-rated entries.

Run once per target type. The second run merges into the existing output file.

Usage:
    # Naturalness first
    python probe-whisper-pseudo-rating.py \
        --checkpoint ./experiments/probe/baseline_modelv2/naturalness \
        --target_type naturalness --split train

    # Then intelligibility (merges into the same output file)
    python probe-whisper-pseudo-rating.py \
        --checkpoint ./experiments/probe/baseline_modelv2/intelligibility \
        --target_type intelligibility --split train
"""

import os
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers.modeling_outputs import SequenceClassifierOutput
from tqdm import tqdm


# ======================
# Models (copied from probe-whisper.py to avoid import issues)
# ======================

class WhisperFeatureProbeV2(nn.Module):
    def __init__(self, input_dim=1280, proj_dim=256, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.projector = nn.Linear(input_dim, proj_dim)
        self.projector2 = nn.Linear(proj_dim, proj_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(proj_dim * 2, 1)

    def forward(self, input_values, lengths=None, **kwargs):
        x = self.norm(input_values)
        x = self.dropout(self.relu(self.projector(x)))
        x = self.dropout(self.relu(self.projector2(x)))

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

        pooled = torch.cat([mean, std], dim=1)
        logits = self.classifier(pooled)
        return SequenceClassifierOutput(logits=logits, hidden_states=pooled)


# ======================
# Dataset for unlabeled samples
# ======================

class UnlabeledFeatureDataset(Dataset):
    """Loads pre-extracted .npy features for samples without ratings."""

    def __init__(self, feature_dir, metadata_path):
        self.feature_dir = feature_dir

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        self.samples = []
        num_skipped_labeled = 0

        for filename, info in metadata.items():
            if filename.startswith("_"):
                continue
            # Skip samples that already have valid ratings
            ratings = info.get("ratings", {})
            if ratings and any(v is not None for v in ratings.values()):
                num_skipped_labeled += 1
                continue

            feat_name = os.path.splitext(filename)[0] + ".npy"
            feat_path = os.path.join(feature_dir, feat_name)

            self.samples.append({
                "feat_path": feat_path,
                "filename": filename,
                "info": info,
            })

        print(f"Loaded {len(self.samples)} unlabeled samples from {metadata_path}")
        print(f"  Skipped {num_skipped_labeled} labeled samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = np.load(sample["feat_path"])
        return {
            "input_values": torch.from_numpy(features).float(),
            "filename": sample["filename"],
        }


def collate_fn(batch):
    features = [b["input_values"] for b in batch]
    filenames = [b["filename"] for b in batch]

    max_len = max(f.shape[0] for f in features)
    hidden_dim = features[0].shape[1]
    batch_size = len(features)

    padded = torch.zeros(batch_size, max_len, hidden_dim)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    for i, f in enumerate(features):
        length = f.shape[0]
        padded[i, :length] = f
        lengths[i] = length

    return {
        "input_values": padded,
        "lengths": lengths,
        "filenames": filenames,
    }


# ======================
# Model loading
# ======================

def load_probe_model(checkpoint_dir, input_dim, proj_dim, dropout, device):
    """Load a trained probe model from a checkpoint directory."""
    model = WhisperFeatureProbeV2(
        input_dim=input_dim, proj_dim=proj_dim, dropout=dropout,
    )

    safe_path = os.path.join(checkpoint_dir, "model.safetensors")
    bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if os.path.isfile(safe_path):
        from safetensors.torch import load_file
        state_dict = load_file(safe_path, device=str(device))
    elif os.path.isfile(bin_path):
        state_dict = torch.load(bin_path, map_location=device)
    else:
        raise FileNotFoundError(
            f"No model.safetensors or pytorch_model.bin found in {checkpoint_dir}"
        )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


# ======================
# Inference
# ======================

@torch.no_grad()
def predict(model, dataloader, device):
    """Run inference and return {filename: prediction} dict."""
    predictions = {}
    for batch in tqdm(dataloader, desc="Predicting"):
        input_values = batch["input_values"].to(device)
        lengths = batch["lengths"].to(device)
        filenames = batch["filenames"]

        outputs = model(input_values=input_values, lengths=lengths)
        preds = outputs.logits.squeeze(-1).cpu().numpy()

        for fname, pred in zip(filenames, preds):
            # Clip to valid range [1, 7]
            predictions[fname] = float(np.clip(pred, 1.0, 7.0))

    return predictions


# ======================
# Main
# ======================

def main():
    parser = argparse.ArgumentParser(
        description="Pseudo-label unlabeled data using a trained probe model (one target at a time)."
    )

    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained probe checkpoint dir (contains model.safetensors)"
    )
    parser.add_argument(
        "--target_type", type=str, required=True,
        choices=["intelligibility", "naturalness", "average"],
        help="Target type to predict"
    )

    # Data
    parser.add_argument(
        "--feature_dir", type=str,
        default="/path/to/whisper_features",
        help="Root directory containing pre-extracted features"
    )
    parser.add_argument(
        "--data_dir", type=str,
        default="./dataset_total",
        help="Directory containing JSON metadata (dataset_total)"
    )
    parser.add_argument(
        "--split", type=str, default="train",
        help="Split to pseudo-label (e.g., train)"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=None,
        help="Output directory for pseudo-labeled dataset (default: dataset_pseudo_{target_type})"
    )

    # Model
    parser.add_argument("--input_dim", type=int, default=1280)
    parser.add_argument("--proj_dim", type=int, default=320)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Inference
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"dataset_pseudo_{args.target_type}",
        )

    target_key_map = {
        "intelligibility": "Intelligibility",
        "naturalness": "Naturalness",
        "average": "Average",
    }
    target_key = target_key_map[args.target_type]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Target: {args.target_type} ({target_key})")

    # Load source metadata
    metadata_path = os.path.join(args.data_dir, f"{args.split}.json")
    feature_dir = os.path.join(args.feature_dir, args.split)

    print(f"\nLoading source metadata from {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Check if output file already exists (from a previous target_type run)
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.split}.json")
    if os.path.exists(output_path):
        print(f"Found existing output at {output_path}, will merge into it")
        with open(output_path, 'r') as f:
            output = json.load(f)
    else:
        # Start from source metadata
        output = dict(metadata)

    # Build dataset of unlabeled samples
    dataset = UnlabeledFeatureDataset(feature_dir, metadata_path)
    if len(dataset) == 0:
        print("No unlabeled samples found. Exiting.")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Load model and predict
    print(f"\nLoading {args.target_type} model from {args.checkpoint}")
    model = load_probe_model(
        args.checkpoint, args.input_dim, args.proj_dim, args.dropout, device,
    )
    predictions = predict(model, dataloader, device)
    del model
    torch.cuda.empty_cache()
    print(f"  Predicted {args.target_type} for {len(predictions)} samples")

    # Merge predictions into output
    n_updated = 0
    for filename, pred_value in predictions.items():
        if filename not in output:
            continue
        entry = output[filename]
        if "ratings" not in entry:
            entry["ratings"] = {}
        entry["ratings"][target_key] = round(pred_value, 2)
        entry["pseudo_labeled"] = True
        n_updated += 1

    # For "average" target: also compute Average rating for labeled samples
    if args.target_type == "average":
        n_labeled_avg = 0
        for filename, entry in output.items():
            if filename.startswith("_"):
                continue
            ratings = entry.get("ratings", {})
            if "Intelligibility" in ratings and "Naturalness" in ratings:
                avg = (ratings["Intelligibility"] + ratings["Naturalness"]) / 2.0
                ratings["Average"] = round(avg, 2)
                n_labeled_avg += 1
        print(f"  Computed Average rating for {n_labeled_avg} samples with both ratings")

    # Save
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    # Count stats
    n_labeled = sum(1 for k, v in output.items()
                    if not k.startswith("_") and "ratings" in v and not v.get("pseudo_labeled"))
    n_pseudo = sum(1 for k, v in output.items()
                   if not k.startswith("_") and v.get("pseudo_labeled"))

    print(f"\nSaved to {output_path}")
    print(f"  Labeled (original): {n_labeled}")
    print(f"  Pseudo-labeled:     {n_pseudo}")
    print(f"  Updated this run:   {n_updated}")

    # Print prediction statistics
    vals = list(predictions.values())
    print(f"\n{args.target_type} pseudo-label statistics:")
    print(f"  mean={np.mean(vals):.2f}, std={np.std(vals):.2f}, "
          f"min={np.min(vals):.2f}, max={np.max(vals):.2f}")
    print(f"\n  Distribution (rounded to int):")
    for v in range(1, 8):
        count = sum(1 for x in vals if round(x) == v)
        print(f"    {v}: {count}")


if __name__ == "__main__":
    main()
