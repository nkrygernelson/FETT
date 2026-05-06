import torch
import torch.nn as nn


class FidelityTranslationHead(nn.Module):
    """
    MLP that maps two pooled compound representations (at source and target fidelity)
    plus the source-fidelity bandgap to the target-fidelity bandgap.

    Inputs to forward:
        src_embedding: [B, embed_dim]  — base-model pooled embedding at source fidelity
        tgt_embedding: [B, embed_dim]  — base-model pooled embedding at target fidelity
        source_bg:     [B]             — normalized source bandgap
    Output:
        [B] — predicted normalized target bandgap.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        input_dim = 2 * embed_dim + 1
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, src_embedding: torch.Tensor, tgt_embedding: torch.Tensor, source_bg: torch.Tensor) -> torch.Tensor:
        x = torch.cat([src_embedding, tgt_embedding, source_bg.unsqueeze(-1)], dim=-1)
        return self.mlp(x).squeeze(-1)
