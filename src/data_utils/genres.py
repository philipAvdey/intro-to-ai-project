''' 
in order to recommend based on genres, we have to start by one-hot encoding genres. 
to do this, we have to figure out all the genres in the dataset and label 
'''

from scipy.sparse import spmatrix
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np

# helper function to get list of genres from a string taken from csv file
# may have to use this again in the future
def get_genres(genre_string: str):
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

filepath: str = './data/filtered_movies.csv'

df = pd.read_csv(filepath)

genres_lists: list[list[str]]= []

for genres_string in df["genres"]:
    genres: list[str] = get_genres(genres_string)
    genres_lists.append(genres)

# creating one hot encoding for genres. will use this later for recommendations. 
mlb = MultiLabelBinarizer()
''' this is a matrix of number of movies and number of unique genre titles
 row i in matrix corresponds to row i in original data
 e.g.
 [[0, 1, 0, 0, 1],   ← Interstellar
 [0, 0, 1, 0, 0],   ← The Hangover
 [1, 0, 0, 1, 0]]   ← The Dark Knight
 '''
genres_encoded = mlb.fit_transform(genres_lists)
if isinstance(genres_encoded, spmatrix):
    genres_encoded = genres_encoded.todense()
user_profile = genres_encoded[[0, 1, 2]]
print(user_profile)
