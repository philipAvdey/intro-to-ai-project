# movie class

import numpy as np
import pandas as pd


class Movie:
    def __init__(self, row: pd.Series):
        self.title = row.get("title", "Unknown")
        self.movie_id = row.get("id", None)
        self.genres_raw = row.get("genres", "")
        self.genres = self.parse_genres(self.genres_raw)
        self.keywords = row.get("keywords", "")  # space-separated string of keywords
        self.release_year: float = self.get_year(row.get("release_date", "0-"))
        self.popularity: float = row.get("popularity", 0.0)
        self.vote_average: float = row.get("vote_average", 0.0)
        # Combined feature vector (set by Vectorizer)
        self.feature_vector: np.ndarray | None = None

        # Legacy: kept for backward compatibility if needed
        self.genre_vector: np.ndarray | None = None

    @staticmethod
    def get_year(year_str: str) -> float:
        if year_str == 0:
            return 0
        return float(year_str.split("-")[0])

    def parse_genres(self, genre_string: str) -> list[str]:
        if not isinstance(genre_string, str):
            return []
        genres = genre_string.split(" ")
        if "Science" in genres and "Fiction" in genres:
            genres.remove("Science")
            genres.remove("Fiction")
            genres.append("Science Fiction")
        if "TV" in genres and "Movie" in genres:
            genres.remove("TV")
            genres.remove("Movie")
            genres.append("TV Movie")
        return genres

    def set_genre_vector(self, vector: np.ndarray):
        """Legacy method for backward compatibility."""
        self.genre_vector = vector

    def set_feature_vector(self, vector: np.ndarray):
        """Set the combined feature vector for this movie."""
        self.feature_vector = vector

    def __repr__(self):
        return f"Movie(title={self.title!r}, genres={self.genres}, keywords={self.keywords!r})"
