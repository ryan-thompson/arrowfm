import torch

#==================================================================================================#
# DAG decoder function
#==================================================================================================#

@torch.no_grad()
def decode_dag(a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Decode a DAG by combining MAP edges and a MAP node ordering."""

    p = a.shape[0]
    device = a.device

    # Create masking matrix from MAP order under Plackett-Luce
    order = torch.argsort(s, descending = True)
    ranks = torch.empty(p, device = device, dtype = torch.long)
    ranks[order] = torch.arange(p, device = device)
    forward = ranks.unsqueeze(1) < ranks.unsqueeze(0)

    # Create pre-DAG edge matrix from MAP edges under Bernoulli
    edge = a > 0.5

    # Combine MAP order and MAP edge matrix
    adj_hat = edge & forward
    adj_hat.fill_diagonal_(False)

    return adj_hat
