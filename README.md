# ArrowFM

ArrowFM is a Python package for using Arrow, a zero-shot foundation model for causal discovery from observational tabular data.

Arrow is described in [Arrow: A Foundation Model for Causal Discovery](https://arxiv.org/abs/2605.07204). Given a dataset, Arrow predicts edge-existence probabilities and node-order scores, then decodes them into a directed acyclic graph (DAG). This package contains the PyTorch model architecture, checkpoint loader, preprocessing, and DAG decoding utilities needed to run that inference workflow.

Model weights are pulled from Hugging Face.

## Installation

Install ArrowFM directly from GitHub:

```bash
python3 -m pip install "git+https://github.com/ryan-thompson/arrowfm.git"
```

## Quickstart

```python
import torch
from arrowfm import ArrowPredictor

predictor = ArrowPredictor()
x = torch.randn(100, 10)
adj = predictor.predict_adjacency(x)
```

`adj` is a boolean DAG adjacency matrix. For a single dataset with shape `(n, p)`, ArrowFM returns an adjacency matrix with shape `(p, p)`, where `adj[j, k]` indicates a predicted directed edge from variable `j` to variable `k`.

Inputs are standardized across rows internally before inference. To run on a GPU, use `ArrowPredictor(device = "cuda")`.

## Example: Predict a DAG from Synthetic Data

The example below generates observations from a small synthetic graph:

```text
x0 -> x2 <- x1
```

Then it asks ArrowFM to infer a DAG and compares the predicted graph with the true graph.

```python
import torch
from arrowfm import ArrowPredictor

torch.manual_seed(123)

n = 100

x0 = torch.randn(n)
x1 = torch.randn(n)
x2 = x0 + x1 + torch.randn(n)

x = torch.stack([x0, x1, x2], dim = 1)
true_adj = torch.tensor([
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 0],
], dtype = torch.bool)

predictor = ArrowPredictor()
pred_adj = predictor.predict_adjacency(x)

print("True adjacency:")
print(true_adj.int())
print("Predicted adjacency:")
print(pred_adj.int())
print(f"Missed edges: {(true_adj & ~ pred_adj).sum().item()}")
print(f"Extra edges:  {(pred_adj & ~ true_adj).sum().item()}")
```

## Inputs and Outputs

`x` may be:

- a single dataset with shape `(n, p)`
- a batch of datasets with shape `(batch, n, p)`

where `n` is the number of rows and `p` is the number of variables.

`predictor.predict_adjacency(x)` returns:

- `(p, p)` for a single dataset
- `(batch, p, p)` for batched input

Pass `return_p_hat = True` to return both the decoded DAG and the model's edge probabilities:

```python
adj, p_hat = predictor.predict_adjacency(x, return_p_hat = True)
```
