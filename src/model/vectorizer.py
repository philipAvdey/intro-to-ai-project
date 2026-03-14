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

# TODO: somewhere, we're going to have to document why we chose TF IDF for text classification and how it works
@dataclass
class TextFeature:
    """Class for text feature (uses TF-IDF for text classification to determine which keywords are most important)."""
    name: str  # e.g., "keywords", "summary"
    column: str  # column name in DataFrame
    max_features: int = 100  # limit dimensionality of TF-IDF
    # stop_words are automatically handled by TfidfVectorizer, e.g. "the", "and"


class Vectorizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

        # Define categorical features (one-hot encoded)
        # TODO: can add more to this
        # TODO: also, we will add numerical features like popularity, etc
        # TODO: we have to add weight to some of these; like popularity should not matter as much as keywords
        self.categorical_features = [
            # CategoricalFeature(name="genres", column="genres", parser=self._parse_genres),
        ]

        # Define text features (TF-IDF encoded)
        # TODO: add summary as another one of these, should be pretty easy
        self.text_features = [
            TextFeature(name="keywords", column="keywords", max_features=100),
        ]

        # Initialize encoders
        # we're essentially going to set a dictionary of matrices and methods of how we're creating those matrices
        # using categorical and text feature creation
        self.categorical_encoders: Dict[str, MultiLabelBinarizer] = {}
        self.categorical_matrices: Dict[str, np.ndarray] = {}

        self.text_encoders: Dict[str, TfidfVectorizer] = {}
        self.text_matrices: Dict[str, np.ndarray] = {}

        # Build all feature encoders and matrices
        self._build_categorical_encoders()
        self._build_text_encoders()

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

    def movie_to_vector(self, movie: Movie) -> np.ndarray:
        """
        Get a big matrix of all the combined vectors, as taken from each of the movie's features

        Args:
            movie: Movie object to vectorize

        Returns:
            Matrix representing all the movie's vectorized features
        """
        feature_vectors = []

        # for each categorical feature, vectorize and add to feature vectors.
        # TODO: figure out how to add weights to each type of vector as needed
        for feature in self.categorical_features:
            # TODO: extend this if statement to include other feature names which are categorical. For now, only genres are.
            if feature.name == "genres":
                categories = movie.genres
            else:
                categories = []

            vec = self._get_categorical_vector(feature.name, categories)
            feature_vectors.append(vec)

        # for each text feature, vectorize and add to feature vectors.
        for feature in self.text_features:
            # can add description to this as well as needed
            if feature.name == "keywords":
                text = movie.keywords if movie.keywords else ""
            else:
                text = ""

            vec = self._get_text_vector(feature.name, text)
            feature_vectors.append(vec)

        # combine all the vectors we got and store it in movie object
        combined_vector = np.concatenate(feature_vectors)
        movie.set_feature_vector(combined_vector)
        return combined_vector

    def _build_combined_feature_matrix(self) -> np.ndarray:
        """
        Get the full feature matrix by combining all feature matrices.

        Returns:
            Full vector for movie
        """
        matrices = []

        for feature in self.categorical_features:
            matrices.append(self.categorical_matrices[feature.name])

        for feature in self.text_features:
            matrices.append(self.text_matrices[feature.name])

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
        result = result[["title", "genres", "keywords"]]
        result["similarity"] = rec_sims
        result.reset_index(drop=True, inplace=True)
        return result
