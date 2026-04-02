---
license: mit
tags:
  - speech
  - dysarthria
  - severity-estimation
  - whisper
  - audio-classification
language:
  - en
pipeline_tag: audio-classification
---

# Dysarthric Speech Severity Level Classifier

A regression probe trained on top of Whisper-large-v3 encoder features for estimating the severity level of dysarthric speech.

**Score scale:** 1.0 (most severe dysarthria) to 7.0 (typical speech)

**GitHub:** [JaesungBae/DA-DSQA](https://github.com/JaesungBae/DA-DSQA)

## Model Description

This model uses a three-stage training pipeline:
1. **Pseudo-labeling** — A baseline probe generates pseudo-labels for unlabeled data
2. **Contrastive pre-training** — Weakly-supervised contrastive learning with typical speech augmentation
3. **Fine-tuning** — Regression probe fine-tuned with the pre-trained projector

**Architecture:** Whisper-large-v3 encoder (frozen) → LayerNorm → 2-layer MLP (proj_dim=320) → Statistics Pooling (mean+std) → Linear → Score

For details, see our paper:
> **Something from Nothing: Data Augmentation for Robust Severity Level Estimation of Dysarthric Speech** [[arXiv]](https://arxiv.org/abs/2603.15988)

## Available Checkpoints

This repository contains **9 checkpoints** trained with different contrastive losses:

| Checkpoint | Contrastive Loss | &tau; |
|---|---|---|
| `proposed_L_coarse_tau0.1` | Proposed (L_coarse) | 0.1 |
| `proposed_L_coarse_tau1.0` | Proposed (L_coarse) | 1.0 |
| `proposed_L_coarse_tau10.0` | Proposed (L_coarse) | 10.0 |
| `proposed_L_coarse_tau50.0` | Proposed (L_coarse) | 50.0 |
| **`proposed_L_coarse_tau100.0`** (default) | Proposed (L_coarse) | 100.0 |
| `proposed_L_cont_tau0.1` | Proposed (L_cont) | 0.1 |
| `proposed_L_dis_tau1.0` | Proposed (L_dis) | 1.0 |
| `rank-n-contrast_tau100.0` | Rank-N-Contrast | 100.0 |
| `simclr_tau0.1` | SimCLR | 0.1 |

## Setup

### 1. Create conda environment

```bash
conda create -n da-dsqa python=3.10 -y
conda activate da-dsqa
```

### 2. Install PyTorch with CUDA

```bash
conda install pytorch torchaudio -c pytorch -y
```

> For a GPU build with a specific CUDA version, see [pytorch.org](https://pytorch.org/get-started/locally/) for the appropriate command.

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

> **Note:** [Silero VAD](https://github.com/snakers4/silero-vad) is loaded automatically at runtime via `torch.hub` — no separate installation needed.

### Runtime Dependencies

This model loads **openai/whisper-large-v3** (~6GB) and **Silero VAD** at initialization time. Ensure sufficient memory is available.

## Usage

### With the custom pipeline

```python
from huggingface_hub import snapshot_download

# Download the model
model_dir = snapshot_download("jaesungbae/da-dsqa")

# Load pipeline (defaults to proposed_L_coarse_tau100.0)
from pipeline import PreTrainedPipeline
pipe = PreTrainedPipeline(model_dir)

# Run inference
result = pipe("/path/to/audio.wav")
print(result)
# {"severity_score": 4.25, "raw_score": 4.2483, "model_name": "proposed_L_coarse_tau100.0"}
```

### Select a specific checkpoint

```python
# Option 1: specify at initialization
pipe = PreTrainedPipeline(model_dir, model_name="simclr_tau0.1")

# Option 2: switch at runtime (Whisper & VAD stay loaded)
pipe.switch_model("rank-n-contrast_tau100.0")
result = pipe("/path/to/audio.wav")

# Option 3: override per call
result = pipe("/path/to/audio.wav", model_name="proposed_L_dis_tau1.0")
```

### Batch inference

```python
results = pipe.batch_inference([
    "/path/to/audio1.wav",
    "/path/to/audio2.wav",
    "/path/to/audio3.wav",
])
for r in results:
    print(f"{r['file']}: {r['severity_score']}")
```

### List available checkpoints

```python
print(pipe.list_models())
# ['proposed_L_coarse_tau0.1', 'proposed_L_coarse_tau1.0', ...]
```

### Compare all checkpoints on a single file

```python
for name in pipe.list_models():
    result = pipe("/path/to/audio.wav", model_name=name)
    print(f"{name}: {result['severity_score']}")
```

### Standalone inference

Clone the [full repository](https://github.com/JaesungBae/DA-DSQA) and run:

```bash
python inference.py \
    --wav /path/to/audio.wav \
    --checkpoint ./checkpoints/stage3/proposed_L_coarse_tau100.0/average
```

## Citation

```bibtex
@misc{bae2026something,
  title         = {Something from Nothing: Data Augmentation for Robust Severity Level Estimation of Dysarthric Speech},
  author        = {Jaesung Bae and Xiuwen Zheng and Minje Kim and Chang D. Yoo and Mark Hasegawa-Johnson},
  year          = {2026},
  eprint        = {2603.15988},
  archivePrefix = {arXiv},
  primaryClass  = {eess.AS},
  url           = {https://arxiv.org/abs/2603.15988}
}
```
