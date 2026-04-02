"""
Custom inference pipeline for HuggingFace Hub.

Pipeline: WAV -> Silero VAD -> Whisper feature extraction -> Probe -> Severity score

Score scale: 1.0 (most severe) to 7.0 (typical speech)

Supports multiple checkpoints. Pass `model_name` to select which checkpoint to use:

    pipe = PreTrainedPipeline(model_dir)                                    # default
    pipe = PreTrainedPipeline(model_dir, model_name="simclr_tau0.1")        # specific

Available checkpoints:
    - proposed_L_coarse_tau0.1
    - proposed_L_coarse_tau1.0
    - proposed_L_coarse_tau10.0   (default)
    - proposed_L_coarse_tau50.0
    - proposed_L_coarse_tau100.0
    - proposed_L_cont_tau0.1
    - proposed_L_dis_tau1.0
    - rank-n-contrast_tau100.0
    - simclr_tau0.1
"""

import io
import json
import os

import torch
import torch.nn as nn
import soundfile as sf
import torchaudio

SAMPLING_RATE = 16000
WHISPER_MODEL_NAME = "openai/whisper-large-v3"
WHISPER_HIDDEN_DIM = 1280
DEFAULT_CHECKPOINT = "proposed_L_coarse_tau100.0"


class WhisperFeatureProbeV2(nn.Module):
    """
    Regression probe on Whisper encoder features.

    Architecture: LayerNorm -> Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout
                  -> Statistics Pooling (mean+std) -> Linear(proj_dim*2, num_classes)
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
        x = self.norm(input_values)
        x = self.dropout(self.relu(self.projector(x)))
        x = self.dropout(self.relu(self.projector2(x)))

        if lengths is not None:
            batch_size, max_len, _ = x.shape
            mask = (
                torch.arange(max_len, device=x.device).unsqueeze(0)
                < lengths.unsqueeze(1)
            )
            mask_f = mask.unsqueeze(-1).float()
            x_masked = x * mask_f
            lengths_f = lengths.unsqueeze(1).float().clamp(min=1)
            mean = x_masked.sum(dim=1) / lengths_f
            var = (x_masked**2).sum(dim=1) / lengths_f - mean**2
            std = var.clamp(min=1e-8).sqrt()
        else:
            mean = x.mean(dim=1)
            std = x.std(dim=1)

        pooled = torch.cat([mean, std], dim=1)
        logits = self.classifier(pooled)

        return type("Output", (), {"logits": logits, "hidden_states": pooled})()


def _load_vad():
    """Load Silero VAD model."""
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
    )
    model.eval()
    get_speech_timestamps = utils[0]
    return model, get_speech_timestamps


def _apply_vad(wav, vad_model, get_speech_timestamps):
    """Apply VAD and return concatenated speech segments."""
    if wav.dim() > 1:
        wav = wav.squeeze()

    speech_timestamps = get_speech_timestamps(
        wav,
        vad_model,
        threshold=0.5,
        sampling_rate=SAMPLING_RATE,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100,
        speech_pad_ms=30,
    )

    if not speech_timestamps:
        return wav

    segments = [
        wav[max(0, ts["start"]) : min(len(wav), ts["end"])]
        for ts in speech_timestamps
    ]
    return torch.cat(segments)


def _extract_features(wav, whisper_model, processor, device):
    """Extract Whisper encoder last-layer hidden states."""
    if isinstance(wav, torch.Tensor):
        wav_np = wav.cpu().numpy()
    else:
        wav_np = wav

    feat_len = len(wav_np) // 320

    input_features = processor(
        wav_np, sampling_rate=SAMPLING_RATE, return_tensors="pt"
    ).input_features.to(
        device=device, dtype=next(whisper_model.parameters()).dtype
    )

    with torch.no_grad():
        out = whisper_model.encoder(input_features, output_hidden_states=True)

    return out.last_hidden_state[:, :feat_len, :].float()


def _load_probe(checkpoint_dir, device):
    """Load a probe model from a checkpoint directory."""
    probe = WhisperFeatureProbeV2(
        input_dim=WHISPER_HIDDEN_DIM, proj_dim=320, num_classes=1
    )
    safe_path = os.path.join(checkpoint_dir, "model.safetensors")
    bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if os.path.isfile(safe_path):
        from safetensors.torch import load_file

        state_dict = load_file(safe_path, device=str(device))
    elif os.path.isfile(bin_path):
        state_dict = torch.load(
            bin_path, map_location=device, weights_only=True
        )
    else:
        raise FileNotFoundError(
            f"No model.safetensors or pytorch_model.bin in {checkpoint_dir}"
        )
    probe.load_state_dict(state_dict)
    probe.to(device).eval()
    return probe


def _discover_checkpoints(path):
    """Find all available checkpoint subdirectories."""
    checkpoints_dir = os.path.join(path, "checkpoints")
    if not os.path.isdir(checkpoints_dir):
        return []
    names = []
    for name in sorted(os.listdir(checkpoints_dir)):
        ckpt_dir = os.path.join(checkpoints_dir, name)
        if os.path.isdir(ckpt_dir) and (
            os.path.isfile(os.path.join(ckpt_dir, "model.safetensors"))
            or os.path.isfile(os.path.join(ckpt_dir, "pytorch_model.bin"))
        ):
            names.append(name)
    return names


class PreTrainedPipeline:
    """
    HuggingFace custom inference pipeline for dysarthric speech severity estimation.

    Accepts a WAV file path or raw audio bytes and returns a severity score
    on a 1.0 (most severe) to 7.0 (typical speech) scale.

    Supports multiple checkpoints stored under `checkpoints/` in the model repo.
    Use `model_name` to select which checkpoint, or call `switch_model()` to
    change at runtime.

    Args:
        path: Path to the downloaded HuggingFace model directory.
        model_name: Name of the checkpoint to load (e.g., "proposed_L_coarse_tau10.0").
                    If None, uses the default from config.json.
    """

    def __init__(self, path: str, model_name: str = None):
        self.path = path
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Read config
        config_path = os.path.join(path, "config.json")
        if os.path.isfile(config_path):
            with open(config_path) as f:
                self.config = json.load(f)
        else:
            self.config = {}

        # Discover available checkpoints
        self.available_checkpoints = _discover_checkpoints(path)
        if not self.available_checkpoints:
            raise FileNotFoundError(
                f"No checkpoints found under {os.path.join(path, 'checkpoints')}/"
            )

        # Load probe for the selected checkpoint
        if model_name is None:
            model_name = self.config.get("default_checkpoint", DEFAULT_CHECKPOINT)
        self.current_model_name = None
        self.probe = None
        self.switch_model(model_name)

        # Load Whisper encoder (shared across all checkpoints)
        from transformers import WhisperFeatureExtractor, WhisperModel

        self.processor = WhisperFeatureExtractor.from_pretrained(
            WHISPER_MODEL_NAME
        )
        self.whisper = WhisperModel.from_pretrained(WHISPER_MODEL_NAME)
        self.whisper.eval().to(self.device)

        # Load Silero VAD (shared across all checkpoints)
        self.vad_model, self.get_speech_timestamps = _load_vad()

    def switch_model(self, model_name: str):
        """
        Switch to a different checkpoint without reloading Whisper or VAD.

        Args:
            model_name: Name of the checkpoint (e.g., "simclr_tau0.1")
        """
        if model_name == self.current_model_name:
            return

        if model_name not in self.available_checkpoints:
            raise ValueError(
                f"Checkpoint '{model_name}' not found. "
                f"Available: {self.available_checkpoints}"
            )

        checkpoint_dir = os.path.join(self.path, "checkpoints", model_name)
        self.probe = _load_probe(checkpoint_dir, self.device)
        self.current_model_name = model_name

    def list_models(self):
        """Return list of available checkpoint names."""
        return list(self.available_checkpoints)

    def _load_wav(self, inputs):
        """Load and preprocess a single audio input to a 1D waveform tensor."""
        if isinstance(inputs, (bytes, bytearray)):
            data, sr = sf.read(io.BytesIO(inputs), dtype="float32")
        else:
            data, sr = sf.read(inputs, dtype="float32")
        wav = torch.from_numpy(data).float()
        if wav.dim() > 1:
            wav = wav.mean(dim=-1)
        if sr != SAMPLING_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLING_RATE)
        return wav

    def __call__(self, inputs, model_name: str = None):
        """
        Run severity estimation on audio input.

        Args:
            inputs: file path (str) or raw audio bytes
            model_name: optionally override the checkpoint for this call

        Returns:
            dict with "severity_score" (clipped to 1-7), "raw_score",
            and "model_name"
        """
        if model_name is not None:
            self.switch_model(model_name)

        wav = self._load_wav(inputs)

        # VAD
        wav = _apply_vad(wav, self.vad_model, self.get_speech_timestamps)

        # Whisper feature extraction
        features = _extract_features(
            wav, self.whisper, self.processor, self.device
        )

        # Probe inference
        with torch.no_grad():
            output = self.probe(features)
        score = output.logits.item()

        return {
            "severity_score": round(max(1.0, min(7.0, score)), 2),
            "raw_score": round(score, 4),
            "model_name": self.current_model_name,
        }

    def batch_inference(self, input_list, model_name: str = None):
        """
        Run severity estimation on a batch of audio files.

        Whisper processes one file at a time (due to variable-length VAD output),
        but the probe runs as a single padded batch for efficiency.

        Args:
            input_list: list of file paths (str) or raw audio bytes
            model_name: optionally override the checkpoint for this call

        Returns:
            list of dicts, each with "file", "severity_score", "raw_score",
            and "model_name"
        """
        if model_name is not None:
            self.switch_model(model_name)

        # Extract features for each file
        all_features = []
        lengths = []
        for inputs in input_list:
            wav = self._load_wav(inputs)
            wav = _apply_vad(wav, self.vad_model, self.get_speech_timestamps)
            features = _extract_features(
                wav, self.whisper, self.processor, self.device
            )
            feat = features.squeeze(0)  # (T, hidden_dim)
            all_features.append(feat)
            lengths.append(feat.shape[0])

        # Pad and batch
        max_len = max(lengths)
        hidden_dim = all_features[0].shape[1]
        batch_size = len(all_features)

        padded = torch.zeros(batch_size, max_len, hidden_dim, device=self.device)
        for i, feat in enumerate(all_features):
            padded[i, : lengths[i]] = feat
        lengths_tensor = torch.tensor(lengths, device=self.device)

        # Batched probe inference
        with torch.no_grad():
            output = self.probe(padded, lengths=lengths_tensor)
        scores = output.logits.squeeze(-1).cpu().tolist()

        results = []
        for i, inputs in enumerate(input_list):
            score = scores[i]
            results.append({
                "file": inputs if isinstance(inputs, str) else f"input_{i}",
                "severity_score": round(max(1.0, min(7.0, score)), 2),
                "raw_score": round(score, 4),
                "model_name": self.current_model_name,
            })
        return results
