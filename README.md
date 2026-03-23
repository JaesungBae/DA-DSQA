# Something from Nothing: Data Augmentation for Robust Severity Level Estimation of Dysarthric Speech

[Code still in Progress]

This repository contains the code for the paper:

> **Something from Nothing: Data Augmentation for Robust Severity Level Estimation of Dysarthric Speech**
> [[Paper]](https://arxiv.org/abs/2603.15988)

## Overview

Speech severity level classification using Whisper encoder features with VAD preprocessing.

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

---

## Inference

### Checkpoints

The repository includes two-stage checkpoints under `checkpoints/`:

- **`step1/`** — Baseline Checkpoint trained only with SAP labeled dataset
- **`step2/`** — Contrastive pre-training checkpoints (`checkpoint_epoch_002.pt`)
- **`step3/`** — Trained probe checkpoints (`model.safetensors`)

| Checkpoint | Contrastive Loss | τ |
|---|---|---|
| `simclr_tau0.1` | SimCLR | 0.1 |
| `rank-n-contrast_tau100.0` | Rank-N-Contrast | 100.0 |
| `proposed_L_dis_tau1.0` | Proposed ($L_{dis}$) | 1.0 |
| `proposed_L_cont_tau0.1` | Proposed ($L_{cont}$) | 0.1 |
| `proposed_L_coarse_tau{0.1,1.0,10.0,50.0,100.0}` | Proposed ($L_{coarse}$) | 0.1–100.0 |


### Single file inference (`inference.py`)

Predict severity from a single WAV file. Handles VAD preprocessing and Whisper feature extraction internally. You may want to modify this inference code for the batch-wise inferece.:

```bash
python inference.py \
    --wav /path/to/audio.wav \
    --checkpoint ./checkpoints/stage3/proposed_L_coarse_tau10.0/average
```

## Train

### 1. Download dataset
#### Training Dataset
- SAP dataset
[link](https://speechaccessibilityproject.beckman.illinois.edu/conduct-research-through-the-project)
- LibriSpeech dataset [link](https://www.openslr.org/12)
#### Cross-domain Dataset
- UASpeech
[link](https://speechtechnology.web.illinois.edu/uaspeech/)
- DysArinVox [link](https://bit.ly/DysArinVox)
- EasyCall [link](http://neurolab.unife.it/easycallcorpus/)
- EWA-DB [link](https://zenodo.org/records/10952480)
- NeuroVoz [link](https://zenodo.org/records/13647600)
#### Generate Metadata

Each dataset requires a JSON metadata file per split (`train.json`, `dev.json`, `test.json`). The JSON is a dictionary keyed by filename:

**Training dataset (SAP / pseudo-labeled)** — used by `probe-whisper.py` and `pretrain_contrastive.py`:

```json
{
  "speaker1/utterance_001.wav": {
    "ratings": {
      "Intelligibility": 3.5,
      "Naturalness": 4.0,
      "Average": 3.75
    }
  },
  "speaker2/utterance_002.wav": {
    "ratings": {
      "Intelligibility": 6.0,
      "Naturalness": 5.5,
      "Average": 5.75
    }
  }
}
```

Required keys:
- `ratings.Intelligibility` — float, 1.0 (most severe) to 7.0 (typical)
- `ratings.Naturalness` — float, 1.0 to 7.0
- `ratings.Average` — float, mean of Intelligibility and Naturalness (used as default target)

**Typical speech dataset (LibriSpeech)** — used by `pretrain_contrastive.py` for contrastive pre-training. All ratings are set to 1.0 (most typical):

```json
{
  "1234-5678-0001.wav": {
    "ratings": {
      "Intelligibility": 1.0,
      "Naturalness": 1.0,
      "Average": 1.0
    }
  }
}
```

**Cross-domain datasets** — each uses a dataset-specific native label field:

| Dataset | Label field | Type |
|---|---|---|
| DysArinVox | `mos` | float (MOS score) |
| EasyCall | `tom_score` | int (TOM 1–5, raw) |
| UASpeech | `intelligibility_pct` | float (0–100%) |
| EWA-DB | `moca` | float (MoCA score) |
| NeuroVoz | `hy_stadium` | float (Hoehn-Yahr stage, HC fallback: 0.0) |

From here, we will exemplify the case training with open dataset, EasyCall and LibriSpeech dev-other. You can mimic this process with SAP dataset and full LibriSpeech train datasets.

Download the dataset:
```bash
# EasyCall (In our paper, we use SAP dataset)
wget http://neurolab.unife.it/easycallcorpus/EasyCall.zip
unzip EasyCall.zip

# LibriSpeech dev-other (In our paper, we use train-clean-100, train-clean-360, and train-other-500)
wget https://openslr.trmal.net/resources/12/dev-other.tar.gz
tar -xzf dev-other.tar.gz
```

Generate EasyCall metadata:
```bash
python data_prepare/create_metadata_easycall.py \
    --easycall_dir ./EasyCall \
    --output_dir_labeled ./dataset_easycall_labeled \
    --output_dir_unlabeled ./dataset_easycall_unlabeled
```

This produces two output directories:
- **`dataset_easycall_labeled/`** — speakers with TOM scores + healthy controls, split into `{train,dev,test}.json` by speaker (no overlap). To map the TOM score to be similar as SAP datasets' score range, we remapped TOM score to `ratings.Average`: 1.0 = typical (HC), 2.0–7.0 = dysarthric.
- **`dataset_easycall_unlabeled/`** — speakers without TOM scores (e.g., F04), saved as `train.json`.

Generate LibriSpeech metadata:
```bash
python data_prepare/create_metadata_librispeech.py \
    --librispeech_dir ./LibriSpeech/dev-other \
    --output_dir ./dataset_librispeech
```

This creates `dataset_librispeech/dev-other.json` (named after the input directory) with all ratings set to 1.0 (typical speech).

### 2. Feature extraction

Extract Whisper encoder features with VAD preprocessing using `extract_features_with_vad.py`. This applies Silero VAD to strip silence, then saves the last-layer hidden states as `.npy` files (float16).

WARNING: Some files may not generate feature correctly. You may want to filter them out from the metadata files.

The extracted `.npy` files mirror the same directory structure as the source wav files within each split directory.

```bash
# EasyCall (labeled)
python extract_features_with_vad.py \
    --model_name whisper-large-v3 \
    --wav_dir ./EasyCall \
    --data ./dataset_easycall_labeled/train.json \
    --dump_dir ./features/easycall/whisper_large_v3_vad/train \
    --vad_threshold 0.2 \
    --min_speech_duration_ms 100 \
    --min_silence_duration_ms 100 \
    --speech_pad_ms 30 \
    --max_duration 30.0

# Repeat for dev/test splits:
# --data ./dataset_easycall_labeled/dev.json  --dump_dir .../dev
# --data ./dataset_easycall_labeled/test.json --dump_dir .../test

# EasyCall (unlabeled)
python extract_features_with_vad.py \
    --model_name whisper-large-v3 \
    --wav_dir ./EasyCall \
    --data ./dataset_easycall_unlabeled/train.json \
    --dump_dir ./features/easycall/whisper_large_v3_vad/train \
    --vad_threshold 0.2 \
    --min_speech_duration_ms 100 \
    --min_silence_duration_ms 100 \
    --speech_pad_ms 30 \
    --max_duration 30.0

# LibriSpeech dev-other
python extract_features_with_vad.py \
    --model_name whisper-large-v3 \
    --wav_dir ./LibriSpeech/dev-other \
    --data ./dataset_librispeech/dev-other.json \
    --dump_dir ./features/librispeech/whisper_large_v3_vad/dev-other \
    --vad_threshold 0.2 \
    --min_speech_duration_ms 100 \
    --min_silence_duration_ms 100 \
    --speech_pad_ms 30 \
    --max_duration 30.0
```

### Validate features (optional)

Some files may fail during feature extraction. Use `validate_features.py` to remove entries with missing `.npy` files from the metadata:

```bash
# EasyCall (labeled)
python data_prepare/validate_features.py \
    --feature_dir ./features/easycall/whisper_large_v3_vad \
    --data_dir ./dataset_easycall_labeled \
    --splits train dev test

# EasyCall (unlabeled)
python data_prepare/validate_features.py \
    --feature_dir ./features/easycall/whisper_large_v3_vad \
    --data_dir ./dataset_easycall_unlabeled \
    --splits train

# LibriSpeech
python data_prepare/validate_features.py \
    --feature_dir ./features/librispeech/whisper_large_v3_vad \
    --data_dir ./dataset_librispeech \
    --splits dev-other
```

### 3. Stage 1: Pseudo-labeling

#### 3.1. Pre-train regression model

Train a baseline probe on the labeled EasyCall data (no contrastive pre-training):

```bash
python probe-whisper.py \
    --feature_dir ./features/easycall_labeled/whisper_large_v3_vad \
    --data_dir ./dataset_easycall_labeled \
    --out_dir ./experiments/stage1 \
    --exp_name baseline \
    --target_type average \
    --proj_dim 320 \
    --dropout 0.1 \
    --micro_batch_size 16 \
    --accum_steps 2 \
    --lr 1e-4 \
    --epochs 10 \
    --save_strategy epoch \
    --seed 42
```

This trains a `WhisperFeatureProbeV2` model to predict `ratings.Average` from pre-extracted Whisper features. The best checkpoint is saved to `./experiments/stage1/baseline/average/`.

#### 3.2. Pseudo-labeling

Use the trained baseline model to generate pseudo-labels for the unlabeled speakers (e.g., F04):

```bash
python probe-whisper-pseudo-rating.py \
    --checkpoint ./experiments/stage1/baseline/average \
    --target_type average \
    --feature_dir ./features/easycall_unlabeled/whisper_large_v3_vad \
    --data_dir ./dataset_easycall_unlabeled \
    --split train \
    --output_dir ./dataset_easycall_pseudo
```

Merge labeled and pseudo-labeled datasets for use in contrastive pre-training:

```bash
python data_prepare/merge_metadata.py \
    --inputs ./dataset_easycall_labeled/train.json ./dataset_easycall_pseudo/train.json \
    --output ./dataset_easycall_total/train.json
cp ./dataset_easycall_labeled/dev.json ./dataset_easycall_total/dev.json
cp ./dataset_easycall_labeled/test.json ./dataset_easycall_total/test.json
```

### 4. Stage 2: Weakly-supervised contrastive pretraining

Pre-train the feature projector with contrastive losses. Config files for each method are in `configs/stage2/`:

| Config | Loss | τ |
|---|---|---|
| `simclr_tau0.1.json` | SimCLR | 0.1 |
| `rank-n-contrast_tau100.0.json` | Rank-N-Contrast | 100.0 |
| `proposed_L_dis_tau1.0.json` | Proposed L_dis | 1.0 |
| `proposed_L_cont_tau0.1.json` | Proposed L_cont | 0.1 |
| `proposed_L_coarse_tau{0.1,1.0,10.0,50.0,100.0}.json` | Proposed L_coarse | 0.1–100.0 |

```bash
python pretrain_contrastive.py \
    --config ./configs/stage2/proposed_L_coarse_tau10.0.json \
    --atypical_feature_dir ./features/easycall/whisper_large_v3_vad \
    --atypical_data_dir ./dataset_easycall_total \
    --typical_feature_dir ./features/librispeech/whisper_large_v3_vad \
    --typical_data_dir ./dataset_librispeech \
    --typical_splits dev-other \
    --output_dir ./experiments/stage2
```

CLI arguments override config values. The best checkpoint is selected by training loss and saved as `pretrained_prenet.pt` under the experiment output directory. No dev set is used for model selection; the `--eval_split` and `--eval_typical_splits` options control optional t-SNE visualization of the learned representations during training.
### 5. Stage 3: Fine-tuning

Fine-tune the probe with the pre-trained projector from Stage 2:

```bash
python probe-whisper.py \
    --feature_dir ./features/easycall/whisper_large_v3_vad \
    --data_dir ./dataset_easycall_labeled \
    --out_dir ./experiments/stage3 \
    --exp_name proposed_L_coarse_tau10.0 \
    --pretrained_prenet ./experiments/stage2/proposed_L_coarse_tau10.0/checkpoint_epoch_002.pt \
    --target_type average \
    --proj_dim 320 \
    --dropout 0.1 \
    --micro_batch_size 16 \
    --accum_steps 2 \
    --lr 1e-4 \
    --epochs 10 \
    --save_strategy epoch \
    --seed 42
```

By default, the loaded projector weights are frozen. To fine-tune them jointly, add `--finetune_prenet`.

#### Evaluation

Evaluate the trained probe on dev/test splits:

```bash
python probe-whisper.py \
    --test_only \
    --checkpoint ./experiments/stage3/proposed_L_coarse_tau10.0/average \
    --feature_dir ./features/easycall/whisper_large_v3_vad \
    --data_dir ./dataset_easycall_labeled \
    --target_type average \
    --proj_dim 320
```