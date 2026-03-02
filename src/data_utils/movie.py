# movie class

import numpy as np
import pandas as pd

class Movie:
    def __init__(self, row: pd.Series):
        self.title = row.get('title', 'Unknown')
        self.movie_id = row.get('id', None)
        self.genres_raw = row.get('genres', '')
        self.genres = self.parse_genres(self.genres_raw)
        self.genre_vector: np.ndarray | None = None
        # can add more stuff later, just genres for now

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
        self.genre_vector = vector
    
    def __repr__(self):
        return f"Movie(title={self.title!r}, genres={self.genres})"