import torch

from arrowfm.config import ModelConfig

#==================================================================================================#
# Arrow model class
#==================================================================================================#

class Arrow(torch.nn.Module):
    """Transformer-based causal discovery model for tabular datasets."""

    def __init__(self, cfg: ModelConfig) -> None:
        """Initialize model components from the provided model configuration."""

        super().__init__()

        # Store embedding dimension used throughout the model
        self.d_model = cfg.d_model

        # Store number of learned summary tokens per column
        self.ncls_rows = cfg.ncls_rows

        # Linear projection mapping raw feature values to model embeddings
        self.value_project = torch.nn.Linear(1, cfg.d_model)

        # Learned summary tokens that aggregate information across dataset rows
        self.row_cls_tokens = torch.nn.Parameter(torch.zeros(1, cfg.ncls_rows, cfg.d_model))

        # Define a transformer encoder layer for processing columns within each row
        col_layer = torch.nn.TransformerEncoderLayer(
            d_model = cfg.d_model,
            nhead = cfg.nhead_cols,
            dim_feedforward = 4 * cfg.d_model,
            dropout = cfg.dropout,
            activation = "gelu",
            batch_first = True,
            norm_first = True,
        )
        self.col_encoder = torch.nn.TransformerEncoder(
            col_layer,
            num_layers = cfg.nlayer_cols,
            enable_nested_tensor = False,
        )

        # Define a transformer decoder layer for processing rows within each column
        row_layer = torch.nn.TransformerDecoderLayer(
            d_model = cfg.d_model,
            nhead = cfg.nhead_rows,
            dim_feedforward = 4 * cfg.d_model,
            dropout = cfg.dropout,
            activation = "gelu",
            batch_first = True,
            norm_first = True,
        )
        self.row_encoder = torch.nn.TransformerDecoder(row_layer, num_layers = cfg.nlayer_rows)

        # Define a transformer encoder layer for contextualizing variable/node embeddings
        node_layer = torch.nn.TransformerEncoderLayer(
            d_model = cfg.d_model,
            nhead = cfg.nhead_nodes,
            dim_feedforward = 4 * cfg.d_model,
            dropout = cfg.dropout,
            activation = "gelu",
            batch_first = True,
            norm_first = True,
        )
        self.node_encoder = torch.nn.TransformerEncoder(
            node_layer,
            num_layers = cfg.nlayer_nodes,
            norm = torch.nn.LayerNorm(cfg.d_model, elementwise_affine = False),
            enable_nested_tensor = False,
        )

        # Linear layer that merges multiple summary tokens into a single column embedding
        self.col_merge = torch.nn.Linear(cfg.ncls_rows * cfg.d_model, cfg.d_model)

        # Scalar order score per variable
        self.order_head = torch.nn.Linear(cfg.d_model, 1)

        # Pairwise edge logit from concatenated variable embeddings
        self.edge_head = torch.nn.Sequential(
            torch.nn.Linear(2 * cfg.d_model, 2 * cfg.d_model),
            torch.nn.GELU(),
            torch.nn.Linear(2 * cfg.d_model, 1),
        )

    def embed_cols(self, x: torch.Tensor) -> torch.Tensor:
        """Encode dataset columns into contextualized variable embeddings."""

        b, n, p = x.shape

        # Project raw feature values into model embedding space
        tokens = self.value_project(x.unsqueeze(- 1))

        # Run transformer encoder over columns within each row
        tokens = tokens.view(b * n, p, self.d_model)
        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
            tokens = self.col_encoder(tokens)
        tokens = tokens.view(b, n, p, self.d_model)

        # Reorder so each column becomes its own sequence over rows
        tokens = tokens.permute(0, 2, 1, 3).contiguous()

        # Flatten batch and column so each column is processed over its row sequence
        rows = tokens.view(b * p, n, self.d_model)

        # Expand learned summary tokens so each column gets its own set
        cls = self.row_cls_tokens.expand(b * p, self.ncls_rows, self.d_model)

        # Run transformer decoder with summary tokens attending to row tokens
        cls = self.row_encoder(tgt = cls, memory = rows)

        # Keep only the learned summary-token outputs for each column
        col = cls.reshape(b * p, self.ncls_rows * self.d_model)

        # Merge multiple summary-token embeddings into one embedding per column
        col = self.col_merge(col).view(b, p, self.d_model)

        # Contextualize variable/node embeddings with respect to the other variables
        col = self.node_encoder(col)

        return col

    def forward(
        self,
        x: torch.Tensor,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]] | tuple[torch.Tensor, torch.Tensor]:
        """Compute edge probabilities, or raw logits when requested."""

        b, _, p = x.shape

        # Compute node embeddings
        h = self.embed_cols(x)

        # Compute order scores from node embeddings
        s = self.order_head(h).squeeze(- 1)
        precedence_logits = s.unsqueeze(2) - s.unsqueeze(1)

        # Compute symmetric pairwise edge logits from concatenated node embeddings
        hj = h.unsqueeze(2).expand(b, p, p, self.d_model)
        hk = h.unsqueeze(1).expand(b, p, p, self.d_model)
        edge_logits = self.edge_head(torch.cat([hj, hk], dim = - 1)).squeeze(- 1)
        edge_logits = 0.5 * (edge_logits + edge_logits.transpose(- 1, - 2))

        if return_logits:
            return edge_logits, precedence_logits

        # Combine edge existence and precedence probabilities to get DAG edge probabilities
        m = torch.sigmoid(precedence_logits)
        a = torch.sigmoid(edge_logits)
        p_hat = a * m

        # Remove self edges
        eye = torch.eye(p, device = x.device, dtype = torch.bool).unsqueeze(0)
        p_hat = p_hat.masked_fill(eye, 0.0)
        a = a.masked_fill(eye, 0.0)

        return p_hat, (a, s)
