from typing import List, Dict, Callable, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data_utils.movie import Movie


@dataclass
class CategoricalFeature:
    """Class for categorical features (meaning features which can be defined with discrete labels/groups)"""

    name: str  # e.g., "genres"
    column: str  # column name in DataFrame
    parser: Callable[
        [str], List[str]
    ]  # function to parse string values into a list of categories
    weight: float = (
        1.0  # weight for this feature (will be normalized with all other features)
    )


# TODO: somewhere, we're going to have to document why we chose TF IDF for text classification and how it works
@dataclass
class TextFeature:
    """Class for text feature (uses TF-IDF for text classification to determine which keywords are most important)."""

    name: str  # e.g., "keywords", "summary"
    column: str  # column name in DataFrame
    max_features: int = 100  # limit dimensionality of TF-IDF
    weight: float = (
        1.0  # weight for this feature (will be normalized with all other features)
    )
    # stop_words are automatically handled by TfidfVectorizer, e.g. "the", "and"


@dataclass
class NumericFeature:
    """Class for numeric features (normalized to 0-1 range for consistent scaling)."""

    name: str  # e.g., "popularity", "vote_average"
    column: str  # column name in DataFrame
    weight: float = (
        1.0  # weight for this feature (will be normalized with all other features)
    )


class Vectorizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

        # Define categorical features (one-hot encoded)
        # Weights: relative importance of each feature (will be normalized to sum to 1.0)
        self.categorical_features = [
            CategoricalFeature(
                name="genres", column="genres", parser=self._parse_genres, weight=0.5
            ),
        ]

        # Define text features (TF-IDF encoded)
        self.text_features = [
            TextFeature(
                name="keywords", column="keywords", max_features=100, weight=0.5
            ),
        ]

        # Define numeric features (normalized to 0-1 range)
        # These have lower weights since they're secondary to content-based features
        self.numeric_features = [
            NumericFeature(name="popularity", column="popularity", weight=0.15),
            NumericFeature(name="vote_average", column="vote_average", weight=0.1),
            NumericFeature(name="release_year", column="release_date", weight=0.05),
        ]

        # Normalize weights so they sum to 1.0
        self._normalize_weights()

        # Initialize encoders
        # we're essentially going to set a dictionary of matrices and methods of how we're creating those matrices
        # using categorical and text feature creation
        self.categorical_encoders: Dict[str, MultiLabelBinarizer] = {}
        self.categorical_matrices: Dict[str, np.ndarray] = {}

        self.text_encoders: Dict[str, TfidfVectorizer] = {}
        self.text_matrices: Dict[str, np.ndarray] = {}

        self.numeric_matrices: Dict[str, np.ndarray] = {}
        self.numeric_stats: Dict[str, Dict[str, float]] = (
            {}
        )  # Store min/max for normalization

        # Build all feature encoders and matrices
        self._build_categorical_encoders()
        self._build_text_encoders()
        self._build_numeric_matrices()

    def _normalize_weights(self):
        """Normalize all feature weights to sum to 1.0 for consistent scaling."""
        total_weight = (
            sum(f.weight for f in self.categorical_features)
            + sum(f.weight for f in self.text_features)
            + sum(f.weight for f in self.numeric_features)
        )

        if total_weight == 0:
            raise ValueError("Total feature weight cannot be zero")

        # Normalize categorical feature weights
        for feature in self.categorical_features:
            feature.weight /= total_weight

        # Normalize text feature weights
        for feature in self.text_features:
            feature.weight /= total_weight

        # Normalize numeric feature weights
        for feature in self.numeric_features:
            feature.weight /= total_weight

    def _parse_genres(self, genre_string: str) -> List[str]:
        """Parse genre string into list of genre categories."""
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

    def _build_categorical_encoders(self):
        """Creating categorical encoders, which then build categorical feature matrices."""

        # for each categorical feature which we're using (not sure if we're going to need anything other than genres)
        for feature in self.categorical_features:
            # Parse the feature values for all movies
            parsed_column = f"_{feature.name}_list"
            self.df[parsed_column] = self.df[feature.column].apply(feature.parser)

            # Create and fit encoder
            # in this case, this is a pretty simple use of a one hot encoded matrix of 1's and 0's
            encoder = MultiLabelBinarizer()
            feature_matrix = encoder.fit_transform(self.df[parsed_column])

            # Convert sparse matrix to dense if needed
            if isinstance(feature_matrix, spmatrix):
                feature_matrix = feature_matrix.todense()

            self.categorical_encoders[feature.name] = encoder
            self.categorical_matrices[feature.name] = feature_matrix

    def _build_text_encoders(self):
        """Creating text encoders, which then build categorical feature matrices."""
        for feature in self.text_features:
            # Get text data from all the features. Fill null values w/ strings if needed to avoid errors later.
            # TODO: is this the best way to handle null values?
            text_data = self.df[feature.column].fillna("").astype(str)

            # Create and fit TF-IDF encoder
            # using the TfidfVectorizer library which is useful to classify text
            # we can filter out stop words through this class
            # for max features, we basically just want to limit how many we're considering
            encoder = TfidfVectorizer(
                max_features=feature.max_features,
                stop_words="english",  # removes common words
                lowercase=True,
                token_pattern=r"\b[a-z]+\b",  # only alphanumeric tokens
            )
            feature_matrix = encoder.fit_transform(text_data)

            # convert to dense for consistency with other features if needed
            if isinstance(feature_matrix, spmatrix):
                feature_matrix = feature_matrix.todense()

            self.text_encoders[feature.name] = encoder
            self.text_matrices[feature.name] = feature_matrix

    def _build_numeric_matrices(self):
        """Building numeric feature matrices with normalization to 0-1 range."""
        for feature in self.numeric_features:
            if feature.name == "release_year":
                # Convert release_date strings (e.g., "2001-01-01") to year floats
                col_data = (
                    self.df[feature.column]
                    .fillna("0-01-01")
                    .apply(Movie.get_year)
                    .astype(float)
                )
            else:
                col_data = self.df[feature.column].fillna(0).astype(float)

            # Calculate min and max for normalization
            min_val = col_data.min()
            max_val = col_data.max()

            # Store stats for later use (e.g., when vectorizing new movies)
            self.numeric_stats[feature.name] = {
                "min": min_val,
                "max": max_val,
            }

            # Normalize to 0-1 range
            if max_val == min_val:
                # Handle case where all values are the same
                normalized = np.ones(len(col_data), dtype=float) * 0.5
            else:
                normalized = (col_data - min_val) / (max_val - min_val)

            # Create matrix (each row is one normalized value)
            # Convert to numpy array in case normalized is a Series
            normalized_array = np.asarray(normalized)
            feature_matrix = normalized_array.reshape(-1, 1).astype(float)
            self.numeric_matrices[feature.name] = feature_matrix

    def _get_categorical_vector(
        self, feature_name: str, categories: List[str]
    ) -> np.ndarray:
        """
        Get one-hot encoded vector for a categorical feature.

        Args:
            feature_name: Name of the feature (e.g., "genres")
            categories: List of category values for this movie; e.g. ["Action", "Adventure", "Thriller"]

        Returns:
            One-hot encoded vector for this feature
        """

        # Return zero vector if no categories for this feature
        if not categories:
            encoder = self.categorical_encoders[feature_name]
            return np.zeros(len(encoder.classes_), dtype=float)

        # Use the feature's encoder to transform
        encoder = self.categorical_encoders[feature_name]
        vec = encoder.transform([categories])

        # Convert sparse matrix to dense if needed
        if isinstance(vec, spmatrix):
            vec = vec.todense()

        # Convert to 1D array
        arr = np.asarray(vec).reshape(-1).astype(float)
        return arr

    def _get_text_vector(self, feature_name: str, text: str) -> np.ndarray:
        """
        Get TF-IDF encoded vector for a text feature.

        Args:
            feature_name: Name of the feature (e.g., "keywords", "summary")
            text: Raw text string for this movie (e.g. "competition secret obsession magic dying and death")

        Returns:
            TF-IDF vector for this feature
        """
        if not text or not isinstance(text, str):
            # Return zero vector if no text
            encoder = self.text_encoders[feature_name]
            n_features = encoder.get_feature_names_out().shape[0]
            return np.zeros(n_features, dtype=float)

        # Fill empty strings with empty space
        text = text.strip() if text else ""

        # Use the feature's encoder to transform
        encoder = self.text_encoders[feature_name]
        vec = encoder.transform([text])

        # Convert sparse matrix to dense
        if isinstance(vec, spmatrix):
            vec = vec.todense()

        arr = np.asarray(vec).reshape(-1).astype(float)

        return arr

    def _get_numeric_vector(self, feature_name: str, value: float) -> np.ndarray:
        """
        Get normalized numeric vector for a single numeric feature.

        Args:
            feature_name: Name of the feature (e.g., "popularity", "vote_average")
            value: Raw numeric value for this movie

        Returns:
            Normalized numeric vector (0-1 range) for this feature
        """
        if feature_name not in self.numeric_stats:
            return np.array([0.0])

        stats = self.numeric_stats[feature_name]
        min_val = stats["min"]
        max_val = stats["max"]

        # Normalize to 0-1 range
        if max_val == min_val:
            normalized_value = 0.5
        else:
            normalized_value = (value - min_val) / (max_val - min_val)
            # Clamp to 0-1 range in case value is outside training data range
            normalized_value = np.clip(normalized_value, 0.0, 1.0)

        return np.array([normalized_value], dtype=float)

    def movie_to_vector(self, movie: Movie) -> np.ndarray:
        """
        Get a big matrix of all the combined vectors, as taken from each of the movie's features

        Args:
            movie: Movie object to vectorize

        Returns:
            Matrix representing all the movie's vectorized features (with weights applied)
        """
        feature_vectors = []

        # for each categorical feature, vectorize and add to feature vectors.
        for feature in self.categorical_features:
            # TODO: extend this if statement to include other feature names which are categorical. For now, only genres are.
            if feature.name == "genres":
                categories = movie.genres
            else:
                categories = []

            vec = self._get_categorical_vector(feature.name, categories)
            # Apply weight to feature vector
            weighted_vec = vec * feature.weight
            feature_vectors.append(weighted_vec)

        # for each text feature, vectorize and add to feature vectors.
        for feature in self.text_features:
            # can add description to this as well as needed
            if feature.name == "keywords":
                text = movie.keywords if movie.keywords else ""
            else:
                text = ""

            vec = self._get_text_vector(feature.name, text)
            # Apply weight to feature vector
            weighted_vec = vec * feature.weight
            feature_vectors.append(weighted_vec)

        # for each numeric feature, vectorize and add to feature vectors.
        for feature in self.numeric_features:
            if feature.name == "popularity":
                value = movie.popularity if movie.popularity else 0.0
            elif feature.name == "vote_average":
                value = movie.vote_average if movie.vote_average else 0.0
            elif feature.name == "release_year":
                value = movie.release_year if movie.release_year else 0.0
            else:
                value = 0.0

            vec = self._get_numeric_vector(feature.name, value)
            # Apply weight to feature vector
            weighted_vec = vec * feature.weight
            feature_vectors.append(weighted_vec)

        # combine all the vectors we got and store it in movie object
        combined_vector = np.concatenate(feature_vectors)
        movie.set_feature_vector(combined_vector)
        return combined_vector

    def _build_combined_feature_matrix(self) -> np.ndarray:
        """
        Get the full feature matrix by combining all feature matrices with weights applied.

        Returns:
            Full vector for movie (with weights applied)
        """
        matrices = []

        for feature in self.categorical_features:
            matrix = self.categorical_matrices[feature.name]
            # Apply weight: multiply each row by the feature weight
            weighted_matrix = matrix * feature.weight
            matrices.append(weighted_matrix)

        for feature in self.text_features:
            matrix = self.text_matrices[feature.name]
            # Apply weight: multiply each row by the feature weight
            weighted_matrix = matrix * feature.weight
            matrices.append(weighted_matrix)

        for feature in self.numeric_features:
            matrix = self.numeric_matrices[feature.name]
            # Apply weight: multiply each row by the feature weight
            weighted_matrix = matrix * feature.weight
            matrices.append(weighted_matrix)

        return np.hstack(matrices)

    # function gets top most similar movies from the user profile
    def recommend(self, user_movies: List[Movie], top_n: int = 10) -> pd.DataFrame:
        # Get the combined feature matrix for all movies
        combined_matrix = self._build_combined_feature_matrix()

        # get a list of user movie vectors
        user_vecs = np.vstack([self.movie_to_vector(m) for m in user_movies])
        # if the user doesn't have any watched movies, error
        if user_vecs.size == 0:
            raise ValueError("user_movies must contain at least one movie")
        # create the user profile as an average of the user movie vectors
        # also use reshape to make sure it's a 2d array for ease of use
        profile = user_vecs.mean(axis=0).reshape(1, -1)
        # here, we get the similarity between every movie in the dataset
        sims = cosine_similarity(np.asarray(combined_matrix), profile).flatten()

        # exclude all the movies which the user already has in their history
        # and sort in descending order in similarity
        user_ids = {m.movie_id for m in user_movies}
        order = np.argsort(-sims)
        selected = []
        for idx in order:
            if self.df.iloc[idx]["id"] in user_ids:
                continue
            selected.append((idx, sims[idx]))
            if len(selected) >= top_n:
                break

        # finally we make our return value, which should be a dataframe for ease of printing
        # first make some empty lists
        rec_indices = [i for i, _ in selected]
        rec_sims = [s for _, s in selected]
        # then add the items with title, genre, and similarity amount as the return value
        result = self.df.iloc[rec_indices].copy()
        result = result[["title", "release_date"]]
        # Extract year from release_date
        result["release_year"] = result["release_date"].apply(Movie.get_year)
        result = result[["title", "release_year"]]
        result["similarity"] = rec_sims
        result.reset_index(drop=True, inplace=True)
        return result
