import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MovieTransformerEncoder(nn.Module):
    def __init__(
        self,
        feature_dims: list[int],
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dims = feature_dims
        self.n_tokens = len(feature_dims)
        self.d_model = d_model

        # project each feature group into d_model space
        self.projections = nn.ModuleList([          
            nn.Linear(dim, d_model) for dim in feature_dims
        ])

        # learned positional embeddings, one per token
        self.pos_embedding = nn.Embedding(self.n_tokens, d_model)

        # transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # output projection after mean pooling
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ← was outside class
        # split flat vector into feature group tokens
        tokens = torch.split(x, self.feature_dims, dim=1)

        # project each token to d_model and stack
        projected = torch.stack(
            [proj(tok) for proj, tok in zip(self.projections, tokens)],
            # (batch, n_tokens, d_model)
            dim=1,  
        )

        # add positional embeddings
        positions = torch.arange(self.n_tokens, device=x.device)
        projected = projected + self.pos_embedding(positions)

        # self-attention across feature tokens
        # (batch, n_tokens, d_model)
        encoded = self.transformer(projected)  

        # mean pool: single embedding per movie
        # (batch, d_model)
        embedding = encoded.mean(dim=1)       
        return self.output_proj(embedding)

    def encode(self, x: torch.Tensor) -> torch.Tensor:  
        self.eval()
        with torch.no_grad():
            return self.forward(x)


def train_transformer(
    feature_matrix: np.ndarray,
    feature_dims: list[int],
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    verbose: bool = True,
) -> tuple[MovieTransformerEncoder, torch.device]:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(feature_matrix, dtype=torch.float32).to(device)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)

    model = MovieTransformerEncoder(
        feature_dims=feature_dims,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
    ).to(device)

    # reconstruction head, only used during training
    recon_head = nn.Linear(d_model, feature_matrix.shape[1]).to(device)

    optimizer = torch.optim.AdamW(      
        list(model.parameters()) + list(recon_head.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) 
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        recon_head.train()
        total_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            embedding = model(batch)
            recon = recon_head(embedding)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        scheduler.step()                  
        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"  Epoch {epoch:>3}/{epochs}  loss: {total_loss / len(X):.6f}")

    return model, device
