"""
Whisper-based sequence classification model for speech quality assessment.

This module provides a regression/classification model built on top of Whisper encoder
for predicting speech quality metrics (e.g., intelligibility, naturalness) from audio.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
from transformers import WhisperModel, WhisperConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from safetensors.torch import load_file as safe_load_file


class WhisperForSequenceClassification(nn.Module):
    """
    Whisper-based model for sequence classification/regression.
    
    This model:
    - Uses Whisper encoder as feature extractor
    - Applies a projection layer to reduce dimensionality
    - Performs masked mean pooling over time dimension
    - Uses a classifier head for final predictions
    
    The model supports two modes:
    1. Single-layer: Uses only the last encoder layer
    2. Multi-layer: Uses weighted sum of all encoder layers (when use_weighted_layer_sum=True)
    
    Args:
        config: WhisperConfig instance
        whisper_model_name: HuggingFace model identifier for Whisper
        proj_dim: Dimension of the projection layer (default: 256)
    
    Example:
        >>> from transformers import WhisperConfig
        >>> config = WhisperConfig.from_pretrained("openai/whisper-large-v2")
        >>> config.num_labels = 1
        >>> config.problem_type = "regression"
        >>> model = WhisperForSequenceClassification(
        ...     config=config,
        ...     whisper_model_name="openai/whisper-large-v2",
        ...     proj_dim=320
        ... )
    """

    def __init__(
        self,
        config: WhisperConfig,
        whisper_model_name: str,
        proj_dim: int = 256,
    ):
        super().__init__()
        self.config = config

        # Load Whisper encoder
        self.whisper = WhisperModel.from_pretrained(whisper_model_name, config=config)

        hidden_size = config.d_model  # Whisper encoder hidden size

        # Projection layer: reduces dimensionality before classification
        self.projector = nn.Linear(hidden_size, proj_dim)
        
        # Classifier head: final prediction layer
        self.classifier = nn.Linear(proj_dim, config.num_labels)

        # Multi-layer aggregation: learnable weights for combining encoder layers
        self.config.use_weighted_layer_sum = getattr(
            self.config, "use_weighted_layer_sum", False
        )
        if self.config.use_weighted_layer_sum:
            # Learnable weights for each encoder layer (excluding embedding layer)
            self.layer_weights = nn.Parameter(torch.ones(config.encoder_layers))

    def _masked_mean_pool(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Perform masked mean pooling over the time dimension.
        
        Args:
            hidden_states: Tensor of shape (batch_size, sequence_length, hidden_size)
            attention_mask: Optional tensor of shape (batch_size, sequence_length)
                          with 1 for valid tokens, 0 for padding
        
        Returns:
            Pooled tensor of shape (batch_size, hidden_size)
        """
        if attention_mask is None:
            return hidden_states.mean(dim=1)

        # Convert mask to same dtype and add dimension for broadcasting
        mask = attention_mask.to(hidden_states.dtype).unsqueeze(-1)  # (B, T, 1)
        hidden_states = hidden_states * mask
        
        # Sum over time dimension
        summed = hidden_states.sum(dim=1)  # (B, D)
        
        # Normalize by number of valid tokens (avoid division by zero)
        denom = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
        
        return summed / denom

    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[tuple, SequenceClassifierOutput]:
        """
        Forward pass through the model.
        
        Note: The parameter is named `input_values` for compatibility with
        HubertForSequenceClassification, but it should contain log-mel spectrogram
        features (input_features in Whisper terminology).
        
        Args:
            input_values: Log-mel spectrogram features of shape
                        (batch_size, sequence_length, n_mels)
            attention_mask: Attention mask of shape (batch_size, sequence_length)
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return a dict or tuple
            labels: Optional labels for computing loss (not used, loss computed externally)
        
        Returns:
            SequenceClassifierOutput containing logits and optionally loss
        """
        return_dict = return_dict if return_dict is not None else True
        
        # Enable hidden states output if using weighted layer sum
        output_hidden_states = (
            True if self.config.use_weighted_layer_sum else output_hidden_states
        )

        # 1. Pass through Whisper encoder
        outputs = self.whisper.encoder(
            input_features=input_values,  # Log-mel features go directly to encoder
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # 2. Extract hidden states (handle weighted layer sum if enabled)
        if output_hidden_states:
            # outputs.hidden_states: tuple of (layer0, layer1, ..., layerN)
            # Each has shape (B, T, D)
            # Skip layer 0 (embedding layer) and stack the rest
            hs = torch.stack(outputs.hidden_states[1:], dim=1)  # (B, L, T, D)

            # Normalize layer weights with softmax
            norm_weights = F.softmax(self.layer_weights, dim=-1)  # (L,)
            w = norm_weights.view(1, -1, 1, 1)  # (1, L, 1, 1) for broadcasting

            # Weighted sum over layers
            hidden_states = (hs * w).sum(dim=1)  # (B, T, D)
        else:
            # Use only the last hidden state
            hidden_states = outputs.last_hidden_state  # (B, T, D)

        # 3. Apply projection layer
        hidden_states = self.projector(hidden_states)  # (B, T_feat, proj_dim)
        _, T_feat, _ = hidden_states.shape

        # 4. Align attention mask to feature sequence length
        # (Interpolate if log-mel time dimension differs from attention_mask)
        feat_mask = None
        if attention_mask is not None:
            att = attention_mask.to(hidden_states.dtype).unsqueeze(1)  # (B, 1, T_in)
            att_feat = F.interpolate(
                att, size=T_feat, mode="nearest"
            ).squeeze(1)  # (B, T_feat)
            feat_mask = (att_feat > 0.5).to(hidden_states.dtype)

        # 5. Pool over time dimension
        pooled_output = self._masked_mean_pool(hidden_states, attention_mask=feat_mask)

        # 6. Apply classifier
        logits = self.classifier(pooled_output)  # (B, num_labels)

        # Loss is computed externally (e.g., in Trainer)
        loss = None

        if not return_dict:
            output = (logits, outputs.hidden_states, outputs.attentions)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            # hidden_states=outputs.hidden_states,  # Uncomment if needed
            # attentions=outputs.attentions,  # Uncomment if needed
        )

    @classmethod
    def load_from_ckpt(
        cls,
        ckpt_dir: str,
        whisper_model_name: str,
        proj_dim: int = 256,
        map_location: str = "cpu",
        num_labels: int = 1,
        problem_type: str = "regression",
        use_weighted_layer_sum: bool = False,
    ) -> "WhisperForSequenceClassification":
        """
        Load model from a checkpoint directory saved by Trainer.save_model().
        
        This method:
        1. Reconstructs the model configuration
        2. Loads weights from safetensors (preferred) or PyTorch bin file
        3. Handles missing/unexpected keys gracefully
        
        Args:
            ckpt_dir: Directory containing the checkpoint
            whisper_model_name: HuggingFace model identifier for Whisper
            proj_dim: Dimension of the projection layer
            map_location: Device to load the model on (default: "cpu")
            num_labels: Number of output labels (default: 1 for regression)
            problem_type: Type of problem ("regression" or "single_label_classification")
            use_weighted_layer_sum: Whether to use weighted layer sum (default: False)
        
        Returns:
            WhisperForSequenceClassification instance with loaded weights
        
        Raises:
            FileNotFoundError: If neither model.safetensors nor pytorch_model.bin exists
        
        Example:
            >>> model = WhisperForSequenceClassification.load_from_ckpt(
            ...     ckpt_dir="./checkpoints/best_model",
            ...     whisper_model_name="openai/whisper-large-v2",
            ...     proj_dim=320,
            ...     use_weighted_layer_sum=True
            ... )
        """
        # 1. Reconstruct model configuration
        config = WhisperConfig.from_pretrained(whisper_model_name)
        config.num_labels = num_labels
        config.problem_type = problem_type
        config.use_weighted_layer_sum = use_weighted_layer_sum

        # 2. Initialize model
        model = cls(
            config=config,
            whisper_model_name=whisper_model_name,
            proj_dim=proj_dim,
        )

        # 3. Load weights: prefer safetensors, fallback to PyTorch bin
        safe_path = os.path.join(ckpt_dir, "model.safetensors")
        bin_path = os.path.join(ckpt_dir, "pytorch_model.bin")

        if os.path.isfile(safe_path):
            state_dict = safe_load_file(safe_path, device=map_location)
            print(f"[load_from_ckpt] Loaded safetensors from {safe_path}")
        elif os.path.isfile(bin_path):
            state_dict = torch.load(bin_path, map_location=map_location)
            print(f"[load_from_ckpt] Loaded PyTorch bin from {bin_path}")
        else:
            raise FileNotFoundError(
                f"Neither model.safetensors nor pytorch_model.bin found in {ckpt_dir}"
            )

        # 4. Load state dict (non-strict to handle missing/unexpected keys)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[load_from_ckpt] Missing keys: {missing}")
        if unexpected:
            print(f"[load_from_ckpt] Unexpected keys: {unexpected}")

        return model
