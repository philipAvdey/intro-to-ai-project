# TODO: Prompt user for movies
# TODO: Get taste profile
# TODO: Get movies sorted by best match
# TODO: Return output to user

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from data_utils.movie import Movie
from model.vectorizer import Vectorizer

num_start_movies = 3


def get_movie_input(df: pd.DataFrame, num: int) -> Movie:
    # TODO: sanitize & error check input
    # TODO: need a better way of searching, this is too rigid. use regex?
    print(f"Input movie title {num}:")
    movie_input = input()
    movie_row = df[df["title"] == movie_input]
    movie_series = movie_row.iloc[0]
    return Movie(movie_series)


def main():
    filepath: str = "./data/filtered_movies.csv"
    df = pd.read_csv(filepath)
    user_movies: list[Movie] = []
    # get the user movies via input
    for i in range(num_start_movies):
        user_movie = get_movie_input(df, i + 1)
        user_movies.append(user_movie)

    # Vectorizer class is created as our main model which will do recommendations
    vectorizer = Vectorizer(df)
    # get the top 10 most similar movies to the ones the user provided
    recommendations = vectorizer.recommend(user_movies, top_n=10)

    # print the app output
    print("\nYour starting movies:")
    for m in user_movies:
        print(f" - {m.title} ({', '.join(m.genres)})")

    print("\nTop 10 recommendations based on genres:")
    print(recommendations.to_string(index=False))


main()
