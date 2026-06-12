"""Inference utilities for ArrowFM causal discovery models."""

from arrowfm.architecture import Arrow
from arrowfm.config import ModelConfig
from arrowfm.decoder import decode_dag
from arrowfm.inference import (
    ArrowPredictor,
    clear_arrow_model_cache,
    load_checkpoint,
    predict_adjacency,
    predict_parameters,
)

#==================================================================================================#
# Package metadata
#==================================================================================================#

__version__ = "0.1.0"

#==================================================================================================#
# Public exports
#==================================================================================================#

__all__ = [
    "Arrow",
    "ArrowPredictor",
    "ModelConfig",
    "clear_arrow_model_cache",
    "decode_dag",
    "load_checkpoint",
    "predict_adjacency",
    "predict_parameters",
]
