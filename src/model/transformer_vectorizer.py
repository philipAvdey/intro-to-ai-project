from typing import List
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity

from data_utils.movie import Movie
from model.vectorizer import Vectorizer
from model.transformer_encoder import MovieTransformerEncoder, train_transformer

class TVectorizer:
    def __init__(self, df: pd.DataFrame, vectorizer: Vectorizer):
        self.df = df
        self.vectorizer = vectorizer
        self.model: MovieTransformerEncoder | None = None
        self.embeddings: np.ndarray | None = None
        self.device: torch.device | None = None
        self.feature_dims = self._get_feature_dims()

    def _get_feature_dims(self) -> list[int]:
        dims = []
        for feature in self.vectorizer.categorical_features:
            dims.append(self.vectorizer.categorical_matrices[feature.name].shape[1])
        for feature in self.vectorizer.text_features:
            dims.append(self.vectorizer.text_matrices[feature.name].shape[1])
        for _ in self.vectorizer.numeric_features:
            dims.append(1)
        return dims

    def fit(
        self,
        epochs: int = 100,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        batch_size: int = 64,
        lr: float = 1e-3,
        verbose: bool = True,
    ):
        print("Constructing feature matrix...")
        feature_matrix = np.asarray(
            self.vectorizer._build_combined_feature_matrix(),
            dtype=np.float32
        )

        print(f"Training Transformer on {feature_matrix.shape[0]} movies, "
              f"{len(self.feature_dims)} tokens each...")

        self.model, self.device = train_transformer(
            feature_matrix=feature_matrix,
            feature_dims=self.feature_dims,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            verbose=verbose,
        )

        # compute embeddings for each movie
        X = torch.tensor(feature_matrix, dtype=torch.float32).to(self.device)
        self.embeddings = self.model.encode(X).cpu().numpy()
        print(f"Done! Embedding shape: {self.embeddings.shape}")

def recommend(self, user_movies: List[Movie], top_n: int = 10) -> pd.DataFrame:
        if self.model is None or self.embeddings is None:
            raise RuntimeError("Call fit() before recommend().")

  # encode user-dictated movies using existing vectorizer to transformer
 user_vecs = np.vstack([
            self.vectorizer.movie_to_vector(m) for m in user_movies
        ]).astype(np.float32)

   X = torch.tensor(user_vecs, dtype=torch.float32).to(self.device)
        user_embeddings = self.model.encode(X).cpu().numpy()
        profile = user_embeddings.mean(axis=0).reshape(1, -1)

# similarity in learned embedding space
sims = cosine_similarity(profile, self.embeddings).flatten()

# now, exclude input movies
user_ids = {m.movie_id for m in user_movies}
        order = np.argsort(-sims)
        selected = []
        for idx in order:
            if self.df.iloc[idx]["id"] in user_ids:
                continue
            selected.append((idx, sims[idx]))
            if len(selected) >= top_n:
                break

        rec_indices = [i for i, _ in selected]   # ← typo fixed: rec_indcies → rec_indices
        rec_sims = [s for _, s in selected]
        result = self.df.iloc[rec_indices].copy()
        result = result[["title", "release_date"]]
        result["release_year"] = result["release_date"].apply(Movie.get_year)
        result = result[["title", "release_year"]]
        result["similarity"] = rec_sims
        result.reset_index(drop=True, inplace=True)
        return result
