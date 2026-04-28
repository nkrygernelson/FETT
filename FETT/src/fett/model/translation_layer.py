import torch
import torch.nn as nn


class FidelityTranslationHead(nn.Module):
    """
    Trainable head for translating between fidelity levels.

    Sits on top of a frozen SetBasedBandgapModel. Given the base model's pooled
    embedding (extracted at the source fidelity), the normalized source bandgap,
    and the target fidelity ID, predicts the target fidelity bandgap.

    Architecture:
        concat([base_embedding, target_fidelity_emb, source_bg]) → MLP → scalar

    Args:
        embed_dim:      Dimension of the base model's pooled embedding
                        (= embedding_dim + fidelity_dim in SetBasedBandgapModel).
        fidelity_dim:   Dimension for the target fidelity embedding.
        num_fidelities: Number of distinct fidelity levels.
        hidden_dim:     Hidden layer size.
        dropout:        Dropout rate.
    """
    def __init__(
        self,
        embed_dim: int,
        fidelity_dim: int,
        num_fidelities: int = 5,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.target_fidelity_embedding = nn.Embedding(num_fidelities, fidelity_dim)
        input_dim = embed_dim + fidelity_dim + 1  # +1 for source_bg scalar
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        base_embedding: torch.Tensor,
        target_fidelity: torch.Tensor,
        source_bg: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            base_embedding:  [batch, embed_dim] — pooled embedding from frozen base model.
            target_fidelity: [batch] long — target fidelity IDs.
            source_bg:       [batch] float — normalized source bandgap.

        Returns:
            [batch] float — predicted normalized target bandgap.
        """
        tgt_emb = self.target_fidelity_embedding(target_fidelity)  # [batch, fidelity_dim]
        src_bg_expanded = source_bg.unsqueeze(-1)                   # [batch, 1]
        x = torch.cat([base_embedding, tgt_emb, src_bg_expanded], dim=-1)
        return self.mlp(x).squeeze(-1)
