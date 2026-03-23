"""
Inference script for severity level estimation from a single WAV file.

Pipeline: WAV → Silero VAD → Whisper feature extraction → Probe model → Score

Usage:
    python inference.py \
        --wav /path/to/audio.wav \
        --checkpoint ./checkpoints/step3/proposed_L_coarse_tau10.0/average
"""

import os
import argparse

import torch
import torchaudio

from transformers import WhisperFeatureExtractor, WhisperModel

SAMPLING_RATE = 16000

WHISPER_MODEL_NAME = "openai/whisper-large-v3"
WHISPER_HIDDEN_DIM = 1280


def load_vad():
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


def apply_vad(wav, vad_model, get_speech_timestamps):
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

    segments = []
    for ts in speech_timestamps:
        start = max(0, ts["start"])
        end = min(len(wav), ts["end"])
        segments.append(wav[start:end])

    return torch.cat(segments)


def extract_features(wav, whisper_model, processor, device):
    """Extract Whisper encoder last-layer hidden states from waveform."""
    if isinstance(wav, torch.Tensor):
        wav_np = wav.cpu().numpy()
    else:
        wav_np = wav

    feat_len = len(wav_np) // 320

    input_features = processor(
        wav_np, sampling_rate=SAMPLING_RATE, return_tensors="pt"
    ).input_features.to(device=device, dtype=next(whisper_model.parameters()).dtype)

    with torch.no_grad():
        out = whisper_model.encoder(
            input_features, output_hidden_states=True
        )

    last_hidden = out.last_hidden_state[:, :feat_len, :]
    return last_hidden.squeeze(0).float().cpu().numpy()


def load_probe(checkpoint_dir, input_dim, proj_dim, device):
    """Load a trained probe model from checkpoint directory."""
    import importlib
    probe_module = importlib.import_module("probe-whisper")
    WhisperFeatureProbeV2 = probe_module.WhisperFeatureProbeV2

    model = WhisperFeatureProbeV2(
        input_dim=input_dim, proj_dim=proj_dim, num_classes=1,
    )

    # Load weights (safetensors or pytorch bin)
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
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Infer severity level from a WAV file."
    )
    parser.add_argument("--wav", type=str, required=True,
                        help="Path to input WAV file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained probe checkpoint directory "
                             "(contains model.safetensors or pytorch_model.bin)")
    parser.add_argument("--proj_dim", type=int, default=320,
                        help="Projection dimension of the probe")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: cuda if available)")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    input_dim = WHISPER_HIDDEN_DIM

    # 1. Load audio
    wav, sr = torchaudio.load(args.wav)
    if sr != SAMPLING_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLING_RATE)
    wav = wav.squeeze()
    print(f"Loaded: {args.wav} ({len(wav) / SAMPLING_RATE:.2f}s)")

    # 2. VAD
    print("Applying VAD...")
    vad_model, get_speech_timestamps = load_vad()
    wav_speech = apply_vad(wav, vad_model, get_speech_timestamps)
    print(f"Speech after VAD: {len(wav_speech) / SAMPLING_RATE:.2f}s")

    # 3. Whisper feature extraction
    print("Extracting features with whisper-large-v3...")
    processor = WhisperFeatureExtractor.from_pretrained(WHISPER_MODEL_NAME)
    whisper_model = WhisperModel.from_pretrained(WHISPER_MODEL_NAME)
    whisper_model.eval().to(device)
    features = extract_features(wav_speech, whisper_model, processor, device)
    print(f"Features: {features.shape}")

    # Free Whisper model memory
    del whisper_model, processor
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 4. Probe inference
    print(f"Loading probe from {args.checkpoint}...")
    probe = load_probe(args.checkpoint, input_dim, args.proj_dim, device)

    features_tensor = torch.from_numpy(features).unsqueeze(0).to(device)
    with torch.no_grad():
        output = probe(features_tensor)
    score = output.logits.item()
    score_clipped = max(1.0, min(7.0, score))

    print(f"\nSeverity score: {score_clipped:.2f}  (raw: {score:.4f})")


if __name__ == "__main__":
    main()
