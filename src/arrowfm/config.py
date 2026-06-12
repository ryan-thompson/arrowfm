import dataclasses

#==================================================================================================#
# Model config
#==================================================================================================#

@dataclasses.dataclass
class ModelConfig:
    """Configuration for the model architecture."""

    d_model: int
    nhead_cols: int
    nhead_rows: int
    nhead_nodes: int
    nlayer_cols: int
    nlayer_rows: int
    nlayer_nodes: int
    ncls_rows: int
    dropout: float
