# TODO: Prompt user for movies
# TODO: Get taste profile
# TODO: Get movies sorted by best match
# TODO: Return output to user

import pandas as pd

num_start_movies = 3

def get_movie_input(df: pd.DataFrame, num: int) -> pd.DataFrame:
    #TODO: sanitize & error check input
    #TODO: need a better way of searching, this is too rigid. use regex?
    print(f'Input movie title {num}:')
    movie_input_1 = input()
    movie_1 = df[df['title'] == movie_input_1]
    return movie_1

def main():
    filepath: str = './data/filtered_movies.csv'
    df = pd.read_csv(filepath)
    user_movies = []
    for i in range(num_start_movies):
        user_movie = get_movie_input(df, i + 1)
        user_movies.append(user_movie)

main()

