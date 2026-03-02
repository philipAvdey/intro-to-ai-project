from typing import List

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

from data_utils.movie import Movie


class Vectorizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        # make a column with list of genre strings so mlb can consume it
        self.df["_genre_list"] = self.df["genres"].apply(self.get_genres)

        self.mlb = MultiLabelBinarizer()
        self.genre_matrix = self.mlb.fit_transform(self.df["_genre_list"])
        if isinstance(self.genre_matrix, spmatrix):
            self.genre_matrix = self.genre_matrix.todense()

    # helper function to get list of genres from a string taken from csv file
    # may have to use this again in the future
    def get_genres(self, genre_string: str):
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

    # get a vector (matrix) based on the movie 
    # TODO: expand this function to include more features beyond genres
    # TODO: add more/less weight to certain categories, add normalization maybe?
    # TODO: add more weight to genres which appear more often in user vectors? each genre is given equal weight currently when averaging
    def movie_to_vector(self, movie: Movie) -> np.ndarray:
        genres = movie.genres
        # if we do not have any genres for this movie, then return an empty vector, make sure it's the right size
        if not genres:
            return np.zeros(len(self.mlb.classes_), dtype=int)
        # get the vector as a MultiLabelBinarizer to turn the genres into a 
        vec = self.mlb.transform([genres])
        # will cause errors if we don't convert it into a proper ndarray with .todense
        if isinstance(vec, spmatrix):
            vec = vec.todense()
        # convert into a 1d array from the matrix we got; makes it easier to deal with 
        arr = np.asarray(vec).reshape(-1)
        # set the vector of the movie to this array we got, representing its vector
        movie.set_genre_vector(arr)
        return arr

    # function gets top most similar movies from the user profile
    def recommend(self, user_movies: List[Movie], top_n: int = 10) -> pd.DataFrame:
        # get a list of user movie vectors
        user_vecs = np.vstack([self.movie_to_vector(m) for m in user_movies])
        # if the user doesn't have any watched movies, error
        if user_vecs.size == 0:
            raise ValueError("user_movies must contain at least one movie")
        # create the user profile as an average of the user movie vectors
        # also use reshape to make sure it's a 2d array for ease of use
        profile = user_vecs.mean(axis=0).reshape(1, -1)
        # here, we get the similarity between every movie in the dataset 
        sims = cosine_similarity(self.genre_matrix, profile).flatten()

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
        result = result[["title", "genres"]]
        result["similarity"] = rec_sims
        result.reset_index(drop=True, inplace=True)
        return result
