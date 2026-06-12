import torch

#==================================================================================================#
# Standardize dataset function
#==================================================================================================#

@torch.no_grad()
def standardize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Standardize each variable across dataset rows."""

    # Compute per-variable mean and standard deviation across rows
    mean = x.mean(dim = - 2, keepdim = True)
    std = x.std(dim = - 2, keepdim = True).clamp_min(eps)

    return (x - mean) / std
