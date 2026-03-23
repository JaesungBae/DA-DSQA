"""
Whisper Feature Extraction with VAD Preprocessing

Applies Voice Activity Detection (VAD) to remove silence before extracting
Whisper encoder features, then saves the last layer outputs as .npy files.
"""

import os
import json
import argparse
import torch
import torchaudio
import numpy as np
from tqdm import tqdm

SAMPLING_RATE = 16000
WHISPER_FRAME_RATE = 50  # ~50 frames per second (320 samples per frame)

WHISPER_CONFIGS = {
    "whisper-tiny": {"hidden_dim": 384, "num_layers": 4},
    "whisper-base": {"hidden_dim": 512, "num_layers": 6},
    "whisper-small": {"hidden_dim": 768, "num_layers": 12},
    "whisper-medium": {"hidden_dim": 1024, "num_layers": 24},
    "whisper-large-v2": {"hidden_dim": 1280, "num_layers": 32},
    "whisper-large-v3": {"hidden_dim": 1280, "num_layers": 32},
}


def load_silero_vad(device="cpu"):
    """Load Silero VAD model."""
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )
    model = model.to(device)
    model.eval()
    get_speech_timestamps, _, read_audio, _, _ = utils
    return model, get_speech_timestamps


def apply_vad(wav, sample_rate, vad_model, get_speech_timestamps,
              threshold=0.5, min_speech_duration_ms=250,
              min_silence_duration_ms=100, padding_ms=30, speech_pad_ms=30):
    """
    Apply VAD to audio and return only speech segments.

    Args:
        wav: Audio waveform tensor (1D)
        sample_rate: Sample rate
        vad_model: Silero VAD model
        get_speech_timestamps: Function to get speech timestamps
        threshold: Speech probability threshold (0-1)
        min_speech_duration_ms: Minimum speech segment duration
        min_silence_duration_ms: Minimum silence duration to split
        padding_ms: Padding around speech segments

    Returns:
        speech_wav: Audio with only speech segments concatenated
        speech_timestamps: List of (start, end) tuples in samples
    """
    if wav.dim() > 1:
        wav = wav.squeeze()

    # Get speech timestamps
    speech_timestamps = get_speech_timestamps(
        wav,
        vad_model,
        threshold=threshold,
        sampling_rate=sample_rate,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
    )

    if not speech_timestamps:
        # No speech detected, return original audio
        return wav, [(0, len(wav))]

    # Add padding and extract speech segments
    padding_samples = int(padding_ms * sample_rate / 1000)
    speech_segments = []
    segment_info = []

    for ts in speech_timestamps:
        start = max(0, ts['start'] - padding_samples)
        end = min(len(wav), ts['end'] + padding_samples)
        speech_segments.append(wav[start:end])
        segment_info.append((start, end))

    # Concatenate all speech segments
    speech_wav = torch.cat(speech_segments)

    return speech_wav, segment_info


def load_whisper_model(model_name, model_path, device):
    """Load Whisper model and feature extractor."""
    from transformers import WhisperFeatureExtractor, WhisperForAudioClassification

    source_map = {
        "whisper-large-v3": "openai/whisper-large-v3",
        "whisper-large-v2": "openai/whisper-large-v2",
        "whisper-medium": "openai/whisper-medium",
        "whisper-small": "openai/whisper-small",
        "whisper-base": "openai/whisper-base",
        "whisper-tiny": "openai/whisper-tiny",
    }
    source = source_map.get(model_name, model_name)

    processor = WhisperFeatureExtractor.from_pretrained(source, cache_dir=model_path)
    model = WhisperForAudioClassification.from_pretrained(source, cache_dir=model_path)
    model.eval()
    model.to(device)

    return model, processor


def extract_whisper_features(
    wav,
    model,
    processor,
    device,
    output_norm=False,
):
    """
    Extract Whisper encoder features from audio waveform.

    Args:
        wav: Audio waveform (numpy array or torch tensor)
        model: Whisper model
        processor: Whisper feature extractor
        device: torch device
        output_norm: Apply layer normalization to output

    Returns:
        features: numpy array of shape (num_frames, hidden_dim)
    """
    import torch.nn.functional as F

    if isinstance(wav, torch.Tensor):
        wav_np = wav.cpu().numpy()
    else:
        wav_np = wav

    # Calculate expected feature length
    feat_len = int(len(wav_np) // 320)

    # Process through Whisper feature extractor
    input_features = processor(
        wav_np,
        sampling_rate=SAMPLING_RATE,
        return_tensors="pt"
    ).input_features
    input_features = input_features.to(device)
    # Match model dtype to avoid float/half mismatch on GPU
    input_features = input_features.to(next(model.parameters()).dtype)

    # Extract features - get last layer hidden states
    with torch.no_grad():
        out = model(input_features, output_hidden_states=True)

    # Get last layer hidden states
    last_hidden = out.hidden_states[-1]  # (batch, seq_len, hidden_dim)
    last_hidden = last_hidden[:, :feat_len, :]

    if output_norm:
        last_hidden = F.layer_norm(last_hidden, last_hidden.shape[1:])

    # Remove batch dimension and convert to numpy
    features = last_hidden.squeeze(0).cpu().numpy()

    return features


def process_audio_file(
    wav_path,
    whisper_model,
    whisper_processor,
    vad_model,
    get_speech_timestamps,
    device,
    output_norm=False,
    vad_threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=100,
    speech_pad_ms=30,
):
    """
    Process a single audio file: apply VAD, extract features.

    Args:
        wav_path: Path to audio file
        whisper_model: Whisper model
        whisper_processor: Whisper feature extractor
        vad_model: Silero VAD model
        get_speech_timestamps: VAD utility function
        device: torch device
        output_norm: Apply layer normalization
        vad_threshold: VAD speech probability threshold
        min_speech_duration_ms: Minimum speech segment duration in ms
        min_silence_duration_ms: Minimum silence duration to split in ms
        speech_pad_ms: Padding around speech segments in ms

    Returns:
        features: numpy array of shape (num_frames, hidden_dim)
        metadata: dict with processing info
    """
    # Load audio
    wav, sr = torchaudio.load(wav_path)

    # Convert to mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)

    # Resample if needed
    if sr != SAMPLING_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLING_RATE)

    original_duration = len(wav) / SAMPLING_RATE

    # Apply VAD
    speech_wav, speech_segments = apply_vad(
        wav,
        SAMPLING_RATE,
        vad_model,
        get_speech_timestamps,
        threshold=vad_threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
    )

    vad_duration = len(speech_wav) / SAMPLING_RATE

    # Extract features from VAD-processed audio
    features = extract_whisper_features(
        speech_wav,
        whisper_model,
        whisper_processor,
        device,
        output_norm,
    )

    metadata = {
        "original_duration": original_duration,
        "vad_duration": vad_duration,
        "speech_ratio": vad_duration / original_duration if original_duration > 0 else 0,
        "num_segments": len(speech_segments),
        "feature_shape": features.shape,
    }

    return features, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Extract Whisper features with VAD preprocessing"
    )
    parser.add_argument("--model_name", type=str, default="whisper-large-v3",
                        choices=list(WHISPER_CONFIGS.keys()),
                        help="Whisper model variant")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to cache models")
    parser.add_argument("--dump_dir", type=str, required=True,
                        help="Directory to save extracted features")
    parser.add_argument("--wav_dir", type=str, required=True,
                        help="Base directory containing wav files")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to JSON metadata file")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument("--output_norm", action="store_true",
                        help="Apply layer normalization to output")
    parser.add_argument("--vad_threshold", type=float, default=0.5,
                        help="VAD speech probability threshold (0-1)")
    parser.add_argument("--min_speech_duration_ms", type=int, default=250,
                        help="Minimum speech segment duration in ms")
    parser.add_argument("--min_silence_duration_ms", type=int, default=100,
                        help="Minimum silence duration to split segments in ms")
    parser.add_argument("--speech_pad_ms", type=int, default=30,
                        help="Padding around speech segments in ms")
    parser.add_argument("--max_duration", type=float, default=30.0,
                        help="Skip audio longer than this (seconds)")
    parser.add_argument("--save_metadata", action="store_true",
                        help="Save processing metadata to JSON")
    # Multi-GPU split arguments
    parser.add_argument("--num_splits", type=int, default=1,
                        help="Total number of splits for parallel processing")
    parser.add_argument("--split_idx", type=int, default=0,
                        help="Index of current split (0-indexed, 0 to num_splits-1)")

    args = parser.parse_args()

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load metadata
    print(f"\nLoading metadata from {args.data}...")
    with open(args.data, 'r') as f:
        data = json.load(f)

    seg_ids = [k for k in data.keys() if not k.startswith("_")]
    total_segments = len(seg_ids)
    print(f"Found {total_segments} total segments")

    # Apply split if multi-GPU processing
    if args.num_splits > 1:
        if args.split_idx >= args.num_splits:
            raise ValueError(f"split_idx ({args.split_idx}) must be < num_splits ({args.num_splits})")

        # Sort for consistent splitting across runs
        seg_ids = sorted(seg_ids)

        # Calculate split boundaries
        split_size = len(seg_ids) // args.num_splits
        remainder = len(seg_ids) % args.num_splits

        # Distribute remainder across first few splits
        start_idx = args.split_idx * split_size + min(args.split_idx, remainder)
        end_idx = start_idx + split_size + (1 if args.split_idx < remainder else 0)

        seg_ids = seg_ids[start_idx:end_idx]
        print(f"Split {args.split_idx + 1}/{args.num_splits}: processing {len(seg_ids)} segments (indices {start_idx}-{end_idx-1})")

    # Load VAD model
    print("\nLoading Silero VAD model...")
    vad_model, get_speech_timestamps = load_silero_vad(device="cpu")  # VAD works best on CPU
    print("VAD model loaded")
    print(f"VAD parameters: threshold={args.vad_threshold}, "
          f"min_speech={args.min_speech_duration_ms}ms, "
          f"min_silence={args.min_silence_duration_ms}ms, "
          f"speech_pad={args.speech_pad_ms}ms")

    # Load Whisper model
    print(f"\nLoading {args.model_name}...")
    whisper_model, whisper_processor = load_whisper_model(
        args.model_name, args.model_path, device
    )
    print("Whisper model loaded")

    # Create output directory
    os.makedirs(args.dump_dir, exist_ok=True)

    # Processing statistics
    success_count = 0
    skip_count = 0
    error_count = 0
    total_original_duration = 0.0
    total_vad_duration = 0.0
    all_metadata = {}

    # Process files
    print(f"\nExtracting features to {args.dump_dir}...")
    for seg_id in tqdm(seg_ids, desc="Processing"):
        wav_path = os.path.join(args.wav_dir, seg_id)

        # Check if file exists
        if not os.path.exists(wav_path):
            print(f"\nSKIP (file not found): {wav_path}")
            skip_count += 1
            continue

        # Check if already processed
        save_name = os.path.splitext(seg_id)[0] + ".npy"
        save_path = os.path.join(args.dump_dir, save_name)
        if os.path.exists(save_path):
            success_count += 1
            continue

        try:
            # Check duration
            try:
                info = torchaudio.info(wav_path)
                duration = info.num_frames / info.sample_rate
            except AttributeError:
                wav_tmp, sr_tmp = torchaudio.load(wav_path)
                duration = wav_tmp.shape[-1] / sr_tmp

            if duration > args.max_duration:
                print(f"\nSKIP (too long {duration:.1f}s > {args.max_duration}s): {seg_id}")
                skip_count += 1
                continue

            # Process file
            features, metadata = process_audio_file(
                wav_path,
                whisper_model,
                whisper_processor,
                vad_model,
                get_speech_timestamps,
                device,
                args.output_norm,
                args.vad_threshold,
                args.min_speech_duration_ms,
                args.min_silence_duration_ms,
                args.speech_pad_ms,
            )

            # Save features as float16 to halve storage
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, features.astype(np.float16))

            # Track statistics
            total_original_duration += metadata["original_duration"]
            total_vad_duration += metadata["vad_duration"]

            if args.save_metadata:
                all_metadata[seg_id] = {
                    "original_duration": metadata["original_duration"],
                    "vad_duration": metadata["vad_duration"],
                    "speech_ratio": metadata["speech_ratio"],
                    "feature_shape": list(metadata["feature_shape"]),
                }

            success_count += 1

        except Exception as e:
            print(f"\nERROR {seg_id}: {e}")
            error_count += 1
            continue

    # Print summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Success:  {success_count}")
    print(f"Skipped:  {skip_count}")
    print(f"Errors:   {error_count}")
    print(f"\nTotal original duration: {total_original_duration/60:.2f} min")
    print(f"Total VAD duration:      {total_vad_duration/60:.2f} min")
    if total_original_duration > 0:
        reduction = (1 - total_vad_duration / total_original_duration) * 100
        print(f"VAD reduction:           {reduction:.1f}%")
    print(f"\nFeatures saved to: {args.dump_dir}")

    # Save metadata if requested
    if args.save_metadata and all_metadata:
        metadata_path = os.path.join(args.dump_dir, "extraction_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
