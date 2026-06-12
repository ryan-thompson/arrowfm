import os
import torch
import huggingface_hub

from arrowfm.architecture import Arrow
from arrowfm.config import ModelConfig
from arrowfm.decoder import decode_dag
from arrowfm.preprocessing import standardize

#==================================================================================================#
# Checkpoint loading function with optional caching
#==================================================================================================#

_ARROW_MODEL_CACHE: dict[tuple[str, str], Arrow] = {}

def _resolve_cache_key(checkpoint_path: str, device: torch.device | str) -> tuple[str, str]:
    """Build a stable per-process cache key for a checkpoint/device pair."""

    # Resolve paths and devices so equivalent checkpoint requests share a cache entry
    resolved_path = os.path.abspath(checkpoint_path)
    resolved_device = str(torch.device(device))

    return resolved_path, resolved_device

def clear_arrow_model_cache() -> None:
    """Drop all cached inference models for the current process."""

    _ARROW_MODEL_CACHE.clear()

@torch.no_grad()
def load_checkpoint(
    checkpoint_path: str,
    device: torch.device | str = "cpu",
    use_cache: bool = True,
) -> Arrow:
    """Load an Arrow checkpoint for inference on the requested device."""

    # Resolve device and cache key for this checkpoint request
    device = torch.device(device)
    cache_key = _resolve_cache_key(checkpoint_path, device)

    # Reuse a previously loaded model when available
    if use_cache and cache_key in _ARROW_MODEL_CACHE:
        return _ARROW_MODEL_CACHE[cache_key]

    # Load checkpoint file and recover the saved model configuration
    ckpt = torch.load(checkpoint_path, map_location = device)
    model_cfg_dict = ckpt.get("config", {}).get("model")
    if model_cfg_dict is None:
        raise KeyError(
            f"Checkpoint '{checkpoint_path}' does not contain model config "
            "under ckpt['config']['model']"
        )

    # Reconstruct model, load weights, and switch to evaluation mode
    model = Arrow(ModelConfig(**model_cfg_dict)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Cache the loaded model for future calls in this process
    if use_cache:
        _ARROW_MODEL_CACHE[cache_key] = model

    return model

#==================================================================================================#
# Arrow predictor class
#==================================================================================================#

class ArrowPredictor:
    """High-level inference interface for ArrowFM models."""

    def __init__(
        self,
        model: torch.nn.Module | None = None,
        *,
        device: torch.device | str = "cpu",
        repo_id: str = "ryan-thompson/arrowfm-base",
        filename: str = "arrowfm-base.pt",
        revision: str | None = None,
        use_cache: bool = True,
        cache_dir: str | None = None,
        token: bool | str | None = None,
        local_files_only: bool = False,
    ) -> None:
        """Create a predictor, loading the default pretrained model when needed."""

        if model is None:
            checkpoint_path = huggingface_hub.hf_hub_download(
                repo_id = repo_id,
                filename = filename,
                revision = revision,
                cache_dir = cache_dir,
                token = token,
                local_files_only = local_files_only,
            )
            model = load_checkpoint(checkpoint_path, device = device, use_cache = use_cache)

        # Store wrapped model in evaluation mode for repeated predictions
        self.model = model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: torch.device | str = "cpu",
        use_cache: bool = True,
    ) -> "ArrowPredictor":
        """Load an ArrowFM checkpoint and return a predictor."""

        # Reuse the checkpoint loader so predictor construction shares cache behavior
        return cls(load_checkpoint(checkpoint_path, device = device, use_cache = use_cache))

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "ryan-thompson/arrowfm-base",
        filename: str = "arrowfm-base.pt",
        revision: str | None = None,
        device: torch.device | str = "cpu",
        use_cache: bool = True,
        cache_dir: str | None = None,
        token: bool | str | None = None,
        local_files_only: bool = False,
    ) -> "ArrowPredictor":
        """Download a checkpoint from Hugging Face Hub and return a predictor."""

        # Resolve checkpoint file from Hugging Face Hub
        checkpoint_path = huggingface_hub.hf_hub_download(
            repo_id = repo_id,
            filename = filename,
            revision = revision,
            cache_dir = cache_dir,
            token = token,
            local_files_only = local_files_only,
        )

        return cls.from_checkpoint(checkpoint_path, device = device, use_cache = use_cache)

    @property
    def device(self) -> torch.device:
        """Return the device where the wrapped model lives."""

        # Infer device from the first model parameter
        return next(self.model.parameters()).device

    def predict_parameters(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run ArrowFM forward inference on one dataset or a batch of datasets."""

        # Delegate to functional inference helper
        return predict_parameters(self.model, x)

    def predict_adjacency(
        self,
        x: torch.Tensor,
        return_p_hat: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Decode DAG adjacency matrices, optionally returning edge probabilities too."""

        # Delegate to functional inference helper
        return predict_adjacency(self.model, x, return_p_hat = return_p_hat)

#==================================================================================================#
# Predict DAG distribution parameters function
#==================================================================================================#

@torch.inference_mode()
def predict_parameters(
    model: torch.nn.Module,
    x: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Run Arrow forward inference on one dataset or a batch of datasets."""

    # Allow callers to pass either one dataset or a batch of datasets
    squeeze_batch = False
    if x.ndim == 2:
        x = x.unsqueeze(0)
        squeeze_batch = True
    elif x.ndim != 3:
        raise ValueError(f"Expected x to have shape (n, p) or (b, n, p), got {tuple(x.shape)}")

    # Move data to the same device as the model
    device = next(model.parameters()).device
    x = x.to(device)

    # Match the training-time preprocessing applied in task sampling
    x = standardize(x)

    # Predict DAG edge probabilities and node order scores
    p_hat, (a, s) = model(x)

    # Remove the batch dimension again for single-dataset inputs
    if squeeze_batch:
        return p_hat.squeeze(0), (a.squeeze(0), s.squeeze(0))

    return p_hat, (a, s)

#==================================================================================================#
# Predict DAG adjacency matrix function
#==================================================================================================#

@torch.no_grad()
def predict_adjacency(
    model: torch.nn.Module,
    x: torch.Tensor,
    return_p_hat: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Decode DAG adjacency matrices, optionally returning edge probabilities too."""

    # Predict edge probabilities and node order scores
    p_hat, (a, s) = predict_parameters(model, x)

    # Decode either a single task or a batch of tasks
    if a.ndim == 2:
        adj_hat = decode_dag(a, s)
    else:
        adj_hat = torch.stack([decode_dag(a_i, s_i) for a_i, s_i in zip(a, s)], dim = 0)

    if return_p_hat:
        return adj_hat, p_hat

    return adj_hat
